"""Tests for the core FastAPI app + WS link server.

Covers Step 8 wiring:

* app lifespan boots an in-memory store + seeds the ROM + registers the
  built-in tool runnables + loads defs (so ``export_defs`` is non-empty)
* REST surface: ``GET /health``, ``POST /threads`` (provider/model pinning),
  ``GET /threads/{id}`` (404 on missing)
* WS ``/link``:
    - bad/missing bearer token -> 403 (upgrade rejected pre-accept via HTTP)
    - good token + ``Connect`` -> connection stays open; events published on
      the bus fan out to the socket as JSON
    - inbound ``ToolResult`` resolves the matching pending dispatcher future
      held by :class:`LinkRegistry`, so :class:`WSLinkAdhocDispatcher` can
      round-trip an ad-hoc call
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from core.app import AppState, _make_on_user_message, app_factory  # type: ignore[attr-defined]
from core.endpoint_conn.server import WSLinkAdhocDispatcher, handle_link
from core.providers.base import ToolCall, ToolResultMessage
from core.settings import CoreSettings, StorageSettings, ToolsSettings
from shared.auth import bearer_header
from shared.protocol import (
    Connect,
    FinalEvent,
    ToolResult,
    UserMessage,
)

# --------------------------------------------------------------------------- #
# Settings / fixtures
# --------------------------------------------------------------------------- #


def _settings(tmp_path: Path, *, token: str = "test-token") -> CoreSettings:
    return CoreSettings(
        storage=StorageSettings(sqlite_path=":memory:"),
        tools=ToolsSettings(agent_root=str(tmp_path / "ws")),
        link_token=token,
    )


@pytest.fixture
async def app_state(tmp_path: Path) -> AppState:
    state = AppState(_settings(tmp_path))
    await state.start()
    yield state
    await state.stop()


@pytest.fixture
def settings(tmp_path: Path) -> CoreSettings:
    return _settings(tmp_path)


# --------------------------------------------------------------------------- #
# AppState.start()
# --------------------------------------------------------------------------- #


class TestAppStateStart:
    async def test_seeds_rom_and_advertises_builtins(self, app_state: AppState) -> None:
        defs = app_state.registry.export_defs()
        names = {d.name for d in defs}
        # The four ROM tools all have runnables registered in app.start().
        assert names == {"file_read", "file_write", "file_list", "http_fetch"}

    async def test_defs_persisted_in_store(self, app_state: AppState) -> None:
        defs = await app_state.store.list_tool_defs()
        assert {d.name for d in defs} >= {"file_read", "file_write", "file_list", "http_fetch"}


# --------------------------------------------------------------------------- #
# REST via httpx ASGI transport (no real port needed, runs lifespan)
# --------------------------------------------------------------------------- #


class TestREST:
    @pytest.fixture
    async def client(self, tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
        # httpx.ASGITransport does NOT run the FastAPI lifespan, so boot the
        # AppState explicitly and build the app against the started state.
        state = AppState(_settings(tmp_path))
        await state.start()
        try:
            app = app_factory(state)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                yield c
        finally:
            await state.stop()

    async def test_health(self, client: httpx.AsyncClient) -> None:
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    async def test_create_thread_pins_provider_and_model(self, client: httpx.AsyncClient) -> None:
        r = await client.post(
            "/threads",
            json={"provider": "anthropic", "title": "hi"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["provider"] == "anthropic"
        # model pulled from settings.provider.models["anthropic"].
        assert body["model"] == "claude-3-5-sonnet-20241022"
        assert body["title"] == "hi"

    async def test_create_thread_auto_resolves_default(self, client: httpx.AsyncClient) -> None:
        r = await client.post("/threads", json={})
        assert r.status_code == 200, r.text
        assert r.json()["provider"] == "anthropic"  # settings default

    async def test_create_thread_unknown_provider_400(self, client: httpx.AsyncClient) -> None:
        r = await client.post("/threads", json={"provider": "nope"})
        assert r.status_code == 400

    async def test_get_thread_404(self, client: httpx.AsyncClient) -> None:
        r = await client.get("/threads/does-not-exist")
        assert r.status_code == 404

    async def test_get_thread_after_create(self, client: httpx.AsyncClient) -> None:
        r = await client.post("/threads", json={"title": "t"})
        tid = r.json()["id"]
        r2 = await client.get(f"/threads/{tid}")
        assert r2.status_code == 200
        assert r2.json()["id"] == tid
        assert r2.json()["title"] == "t"


# --------------------------------------------------------------------------- #
# WS /link auth: drives the real ASGI app with a synthetic websocket scope
# and asserts the route raises HTTPException(403) BEFORE accepting the socket.
# We avoid Starlette's TestClient (deprecated under -W error::Warning without
# httpx2) and drive the ASGI interface directly so the test stays warning-clean.
# --------------------------------------------------------------------------- #


async def _ws_upgrade(
    app,
    headers: list[tuple[bytes, bytes]],  # type: ignore[no-untyped-def]
) -> list[dict[str, Any]]:
    """Issue a websocket handshake against ``app`` over ASGI, return ASGI events.

    Returns the list of ASGI messages the app sends on the (rejected) upgrade.
    A pre-accept ``HTTPException(403)`` shows up as a
    ``websocket.http.response.start`` with status 403 and is never followed
    by a ``websocket.accept``.
    """
    sent: list[dict[str, Any]] = []

    # Starlette's WebSocket lifecycle expects the first inbound ASGI event to be
    # ``websocket.connect`` (then either the route accepts and frames flow, or
    # it raises). On a rejected upgrade the route never reads past connect, so
    # those don't matter; on an accepted upgrade the route's reader expects a
    # frame next, so we feed a ``websocket.disconnect`` to unblock it.
    recv_q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    recv_q.put_nowait({"type": "websocket.connect"})
    recv_q.put_nowait({"type": "websocket.disconnect", "code": 1006})

    async def receive() -> dict[str, Any]:
        return await recv_q.get()

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope: dict[str, Any] = {
        "type": "websocket",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "scheme": "ws",
        "path": "/link",
        "raw_path": b"/link",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
        "subprotocols": [],
        "state": {},
    }
    await app(scope, receive, send)
    return sent


def _hdr(token: str | None) -> list[tuple[bytes, bytes]]:
    if token is None:
        return []
    return [(b"authorization", f"Bearer {token}".encode())]


class TestWSAuth:
    async def test_bad_token_rejected_403(self, tmp_path: Path) -> None:
        state = AppState(_settings(tmp_path, token="right"))
        await state.start()
        try:
            events = await _ws_upgrade(app_factory(state), _hdr("wrong"))
        finally:
            await state.stop()
        starts = [e for e in events if e["type"] == "websocket.http.response.start"]
        assert starts and starts[0]["status"] == 403
        assert not any(e["type"] == "websocket.accept" for e in events)

    async def test_missing_token_rejected_403(self, tmp_path: Path) -> None:
        state = AppState(_settings(tmp_path, token="right"))
        await state.start()
        try:
            events = await _ws_upgrade(app_factory(state), _hdr(None))
        finally:
            await state.stop()
        starts = [e for e in events if e["type"] == "websocket.http.response.start"]
        assert starts and starts[0]["status"] == 403

    async def test_empty_expected_token_rejected_403(self, tmp_path: Path) -> None:
        # No token configured -> always reject (belt-and-braces).
        state = AppState(_settings(tmp_path, token=""))
        await state.start()
        try:
            events = await _ws_upgrade(app_factory(state), _hdr("anything"))
        finally:
            await state.stop()
        starts = [e for e in events if e["type"] == "websocket.http.response.start"]
        assert starts and starts[0]["status"] == 403

    async def test_good_token_accepts(self, tmp_path: Path) -> None:
        # Sanity: a valid token does NOT get a 403 -- the socket is accepted
        # and the link sits idle until our fake disconnect unblocks the reader.
        state = AppState(_settings(tmp_path, token="right"))
        await state.start()
        try:
            events = await _ws_upgrade(app_factory(state), _hdr("right"))
        finally:
            await state.stop()
        assert any(e["type"] == "websocket.accept" for e in events)
        assert not any(
            e["type"] == "websocket.http.response.start" and e["status"] == 403 for e in events
        )


# --------------------------------------------------------------------------- #
# WS link logic via a fake in-memory WebSocket driving handle_link directly.
# Keeping the link test off Starlette's TestClient avoids its separate-thread
# event loop; here everything runs on one loop we control.
# --------------------------------------------------------------------------- #


class _FakeWebSocket:
    """Minimal async WebSocket double for ``handle_link``.

    ``receive_text`` blocks on an internal queue (fed by the test), so the
    reader loop's ``wait_for`` can idle-timeout-or-receive naturally.
    ``send_text`` records frames for assertions; ``close`` records the code.
    """

    def __init__(self, *, headers: dict[str, str] | None = None) -> None:
        self.headers = headers or {}
        self.sent: list[str] = []
        self.closed: tuple[int, str] | None = None
        self._in: asyncio.Queue[str] = asyncio.Queue()
        self._accept_called = False

    async def accept(self) -> None:
        self._accept_called = True

    async def send_text(self, data: str) -> None:
        self.sent.append(data)

    async def receive_text(self) -> str:
        return await self._in.get()

    async def close(self, *, code: int = 1000, reason: str | None = None) -> None:
        self.closed = (code, reason or "")

    def feed(self, frame_text: str) -> None:
        self._in.put_nowait(frame_text)

    def sent_frames(self) -> list[dict[str, Any]]:
        return [json.loads(s) for s in self.sent]


async def _drain_one(ws: _FakeWebSocket) -> str:
    """Wait for the next frame the link sends to the endpoint."""
    while not ws.sent:
        await asyncio.sleep(0.01)
    return ws.sent[0]


class TestWSLink:
    async def test_connect_and_event_fanout(self, tmp_path: Path) -> None:
        """A connected endpoint receives bus events for any thread.

        We start ``handle_link`` on a task, feed ``Connect``, publish a
        ``FinalEvent`` on the shared bus, and assert the socket sees it.
        """
        settings = _settings(tmp_path)
        state = AppState(settings)
        await state.start()
        try:
            ws = _FakeWebSocket(headers=bearer_header(settings.link_token))
            link_task = asyncio.create_task(
                handle_link(
                    ws,
                    bus=state.bus,
                    registry=state.link_registry,
                    expected_token=settings.link_token,
                    on_user_message=lambda _f: None,
                    idle_timeout=5.0,
                )
            )
            ws.feed(Connect(endpoint_id="e1").model_dump_json())
            # Give the reader a beat to consume Connect, then publish an event.
            await asyncio.sleep(0.05)
            state.bus.publish(FinalEvent(thread_id="t1", text="hello world"))
            try:
                raw = await asyncio.wait_for(_drain_one(ws), timeout=2.0)
                frame = json.loads(raw)
                assert frame["type"] == "final"
                assert frame["text"] == "hello world"
                assert frame["thread_id"] == "t1"
            finally:
                link_task.cancel()
                await _quiet(link_task)
        finally:
            await state.stop()

    async def test_bad_token_closes_1008(self, tmp_path: Path) -> None:
        """handle_link re-checks the token and closes 1008 if it is wrong."""
        settings = _settings(tmp_path, token="right")
        state = AppState(settings)
        await state.start()
        try:
            ws = _FakeWebSocket(headers=bearer_header("wrong"))
            await handle_link(
                ws,
                bus=state.bus,
                registry=state.link_registry,
                expected_token=settings.link_token,
                on_user_message=lambda _f: None,
                idle_timeout=1.0,
            )
            assert ws.closed is not None
            assert ws.closed[0] == 1008
        finally:
            await state.stop()

    async def test_first_frame_not_connect_errors(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        state = AppState(settings)
        await state.start()
        try:
            ws = _FakeWebSocket(headers=bearer_header(settings.link_token))
            link_task = asyncio.create_task(
                handle_link(
                    ws,
                    bus=state.bus,
                    registry=state.link_registry,
                    expected_token=settings.link_token,
                    on_user_message=lambda _f: None,
                    idle_timeout=5.0,
                )
            )
            # Send a ToolResult as the first frame; not a Connect.
            ws.feed(ToolResult(call_id="x", ok=True, output="y").model_dump_json())
            try:
                raw = await asyncio.wait_for(_drain_one(ws), timeout=2.0)
                frame = json.loads(raw)
                assert frame["type"] == "error"
                assert "connect" in frame["message"].lower()
            finally:
                link_task.cancel()
                await _quiet(link_task)
        finally:
            await state.stop()

    async def test_adhoc_round_trip(self, tmp_path: Path) -> None:
        """``WSLinkAdhocDispatcher.dispatch`` resolves via an inbound ToolResult.

        We register a call's future by dispatching it (on a task), then feed
        the matching ``ToolResult`` frame through ``handle_link``; the
        dispatcher's await returns the result content + the original name.
        """
        settings = _settings(tmp_path)
        state = AppState(settings)
        await state.start()
        try:
            ws = _FakeWebSocket(headers=bearer_header(settings.link_token))
            link_task = asyncio.create_task(
                handle_link(
                    ws,
                    bus=state.bus,
                    registry=state.link_registry,
                    expected_token=settings.link_token,
                    on_user_message=lambda _f: None,
                    idle_timeout=5.0,
                )
            )
            ws.feed(Connect(endpoint_id="e1").model_dump_json())
            await asyncio.sleep(0.05)  # let the reader consume Connect

            dispatcher = WSLinkAdhocDispatcher(state.link_registry, timeout=5.0)
            call_id = "call-123"
            call = ToolCall(id=call_id, name="endpoint_tool", arguments={"q": "x"})

            async def _dispatch() -> ToolResultMessage:
                return await dispatcher.dispatch(call)

            dispatch_task = asyncio.create_task(_dispatch())
            # Give the dispatch a tick to register the future.
            await asyncio.sleep(0.05)
            assert state.link_registry.pending == 1

            # Endpoint returns the result; the reader resolves the future.
            ws.feed(ToolResult(call_id=call_id, ok=True, output="42").model_dump_json())

            msg = await asyncio.wait_for(dispatch_task, timeout=2.0)
            assert msg.call_id == call_id
            assert msg.content == "42"
            # Name is preserved from the dispatch call (not carried in the
            # wire ToolResult, which only has call_id + ok + output/error).
            assert msg.name == "endpoint_tool"
            assert msg.is_error is False
            assert state.link_registry.pending == 0
        finally:
            link_task.cancel()
            await _quiet(link_task)
            await state.stop()

    async def test_adhoc_timeout_surfaces_error_result(self, tmp_path: Path) -> None:
        """If the endpoint never answers, dispatch returns an error result."""
        settings = _settings(tmp_path)
        state = AppState(settings)
        await state.start()
        try:
            ws = _FakeWebSocket(headers=bearer_header(settings.link_token))
            link_task = asyncio.create_task(
                handle_link(
                    ws,
                    bus=state.bus,
                    registry=state.link_registry,
                    expected_token=settings.link_token,
                    on_user_message=lambda _f: None,
                    idle_timeout=5.0,
                )
            )
            ws.feed(Connect(endpoint_id="e1").model_dump_json())
            await asyncio.sleep(0.05)

            dispatcher = WSLinkAdhocDispatcher(state.link_registry, timeout=0.2)
            call = ToolCall(id="late", name="slow_tool", arguments={})

            async def _dispatch() -> ToolResultMessage:
                return await dispatcher.dispatch(call)

            msg = await asyncio.wait_for(asyncio.create_task(_dispatch()), timeout=2.0)
            assert msg.call_id == "late"
            assert msg.is_error is True
            assert "0.2" in msg.content
            assert state.link_registry.pending == 0
        finally:
            link_task.cancel()
            await _quiet(link_task)
            await state.stop()

    async def test_user_message_invokes_callback(self, tmp_path: Path) -> None:
        """Inbound UserMessage triggers the on_user_message callback."""
        settings = _settings(tmp_path)
        state = AppState(settings)
        await state.start()
        try:
            ws = _FakeWebSocket(headers=bearer_header(settings.link_token))
            received: list[UserMessage] = []

            def on_msg(frame: UserMessage) -> None:
                received.append(frame)

            link_task = asyncio.create_task(
                handle_link(
                    ws,
                    bus=state.bus,
                    registry=state.link_registry,
                    expected_token=settings.link_token,
                    on_user_message=on_msg,
                    idle_timeout=5.0,
                )
            )
            ws.feed(Connect(endpoint_id="e1").model_dump_json())
            await asyncio.sleep(0.05)
            ws.feed(UserMessage(content="hello", thread_id="t9").model_dump_json())
            await asyncio.sleep(0.1)
            assert len(received) == 1
            assert received[0].content == "hello"
            assert received[0].thread_id == "t9"
        finally:
            link_task.cancel()
            await _quiet(link_task)
            await state.stop()


async def _quiet(task: asyncio.Task[Any]) -> None:
    """Await a cancelled task, swallowing CancelledError/exceptions."""
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


# --------------------------------------------------------------------------- #
# _make_on_user_message: provider resolution + thread creation
# --------------------------------------------------------------------------- #


class TestOnUserMessage:
    async def test_unknown_provider_emits_fatal_error(
        self, app_state: AppState, tmp_path: Path
    ) -> None:
        bus = app_state.bus
        frames: list[Any] = []
        orig = bus.publish

        def spy(frame: Any) -> None:
            frames.append(frame)
            orig(frame)

        bus.publish = spy  # type: ignore[method-assign]
        spawn = _make_on_user_message(app_state)
        # Unknown provider -> fatal ErrorEvent published, no turn started.
        await spawn(UserMessage(content="hi", provider="nope"))
        errs = [f for f in frames if getattr(f, "type", None) == "error"]
        assert errs and errs[0].fatal is True

    async def test_creates_thread_when_no_thread_id(self, app_state: AppState) -> None:
        # Substitute the loop's run_agent_turn with a no-op so we don't need
        # a real provider; we only assert thread creation happens.
        created: list[str] = []

        async def _noop_run(
            self: Any, thread_id: str, content: str, *, provider_name: str = "auto"
        ) -> None:
            created.append(thread_id)

        import core.app as app_mod

        orig_run = app_mod.AgentLoop.run_agent_turn
        app_mod.AgentLoop.run_agent_turn = _noop_run  # type: ignore[assignment]
        try:
            spawn = _make_on_user_message(app_state)
            await spawn(UserMessage(content="hello"))
            # _spawn schedules the turn as a fire-and-forget task; yield so it
            # actually runs (and appends the created thread id) before we check.
            await asyncio.sleep(0.05)
        finally:
            app_mod.AgentLoop.run_agent_turn = orig_run  # type: ignore[assignment]

        assert len(created) == 1
        thread = await app_state.store.get_thread(created[0])
        assert thread is not None
        assert thread.provider == "anthropic"  # settings default
