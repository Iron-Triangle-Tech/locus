"""Tests for :class:`endpoint.endpoint_conn.client.EndpointClient`.

We drive an in-process ``websockets`` server standing in for core, so the client
exercises its real ``connect`` path (auth header on the upgrade handshake, the
``Connect`` frame, ``events()`` async generator, ad-hoc tool round-trip) against
a live socket on an ephemeral port -- no mocking of the WS layer.

The fake core here intentionally does NOT run core's ``handle_link``; it just
speaks the wire protocol well enough to assert the client's behavior. Round-trip
coverage *through* core lives in ``tests/core/test_app.py`` already.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from endpoint.endpoint_conn.client import AuthError, EndpointClient
from endpoint.settings import CoreSettings, EndpointSettings, UISettings
from shared.protocol import load_endpoint

# --------------------------------------------------------------------------- #
# Fake core WS server
# --------------------------------------------------------------------------- #


class FakeCore:
    """A minimal stand-in for core's /link endpoint.

    On connect it reads the client's ``Connect`` frame, then hands control to
    the per-test ``handler`` coroutine, which receives the server-side
    connection (a ``websockets`` ServerConnection) and can ``recv``/``send``.
    The handler can assert on what the client sent.
    """

    def __init__(
        self,
        *,
        token: str = "right",
        handler: Callable[[Any], Awaitable[None]],
    ) -> None:
        self._token = token
        self._handler = handler
        self._server: Any = None
        self.port: int = 0

    async def start(self) -> None:
        from websockets.asyncio.server import serve

        self._server = await serve(self._serve, "127.0.0.1", 0, max_size=None)
        # websockets Server exposes the bound sockets under ._socks (server
        # attribute) -- pull the actual port from there.
        socks = self._server.sockets
        self.port = socks[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _serve(self, conn: Any) -> None:
        # The HTTP upgrade headers arrive before the WS is established; check
        # the bearer token the same way core does (reject by closing).
        presented = conn.request.headers.get("Authorization", "")
        if presented != f"Bearer {self._token}":
            await conn.close(code=1008, reason="unauthorized")
            return
        # Honor the connect-time handshake by reading the client's Connect.
        await self._handler(conn)


def _settings(core_url: str, *, token: str = "right") -> EndpointSettings:
    return EndpointSettings(
        core=CoreSettings(url=core_url, connect_timeout=5.0, idle_timeout=10.0),
        ui=UISettings(),
        link_token=token,
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestEndpointClient:
    async def test_connect_sends_connect_frame(self) -> None:
        """The client authenticates and emits a ``Connect`` frame first."""
        seen: list[Any] = []

        async def handler(conn: Any) -> None:
            raw = await conn.recv()
            seen.append(load_endpoint(raw))
            await conn.close()

        core = FakeCore(handler=handler)
        await core.start()
        try:
            async with EndpointClient(_settings(f"ws://127.0.0.1:{core.port}/link")):
                await asyncio.sleep(0.1)  # let the handler read the Connect
        finally:
            await core.stop()

        assert seen and isinstance(seen[0], type(seen[0]))  # not-None frame
        assert seen[0].endpoint_id == "endpoint-1"
        assert seen[0].adhoc_tools == []

    async def test_send_user_message_round_trips(self) -> None:
        """send_user_message emits a UserMessage frame core can read back."""
        received: list[Any] = []

        async def handler(conn: Any) -> None:
            # consume Connect
            await conn.recv()
            # then the UserMessage
            received.append(load_endpoint(await conn.recv()))
            await conn.close()

        core = FakeCore(handler=handler)
        await core.start()
        try:
            async with EndpointClient(_settings(f"ws://127.0.0.1:{core.port}/link")) as client:
                await client.send_user_message("hello there")
                await asyncio.sleep(0.1)
        finally:
            await core.stop()

        assert len(received) == 1
        assert received[0].content == "hello there"
        assert received[0].thread_id is None
        assert received[0].provider == "auto"

    async def test_send_user_message_with_thread_and_provider(self) -> None:
        received: list[Any] = []

        async def handler(conn: Any) -> None:
            await conn.recv()
            received.append(load_endpoint(await conn.recv()))
            await conn.close()

        core = FakeCore(handler=handler)
        await core.start()
        try:
            async with EndpointClient(_settings(f"ws://127.0.0.1:{core.port}/link")) as client:
                await client.send_user_message("hi", thread_id="t-7", provider="openai")
                await asyncio.sleep(0.1)
        finally:
            await core.stop()

        assert len(received) == 1
        assert received[0].thread_id == "t-7"
        assert received[0].provider == "openai"

    async def test_events_yields_parsed_frames(self) -> None:
        """events() parses core->endpoint frames into pydantic models."""
        from shared.protocol import FinalEvent, TokenEvent, dump

        async def handler(conn: Any) -> None:
            await conn.recv()  # Connect
            # stream a token then a final
            await conn.send(dump(TokenEvent(thread_id="t1", delta="Hel")))
            await conn.send(dump(TokenEvent(thread_id="t1", delta="lo")))
            await conn.send(dump(FinalEvent(thread_id="t1", text="Hello")))
            await conn.close()

        core = FakeCore(handler=handler)
        await core.start()
        frames: list[Any] = []
        try:
            async with EndpointClient(_settings(f"ws://127.0.0.1:{core.port}/link")) as client:
                async for frame in client.events():
                    frames.append(frame)
        finally:
            await core.stop()

        deltas = "".join(f.delta for f in frames if f.type == "token")
        assert deltas == "Hello"
        finals = [f for f in frames if f.type == "final"]
        assert finals and finals[0].text == "Hello"

    async def test_adhoc_tool_invokes_local_runnable_and_replies_toolresult(self) -> None:
        """A ToolCallEvent for an advertised ad-hoc tool runs locally and the
        client replies with a matching ToolResult frame core can parse."""
        from shared.protocol import ToolCallEvent, dump, load_endpoint

        # Ad-hoc tool: returns the echoed query string.
        async def echo(args: dict[str, Any]) -> str:
            return f"echo:{args.get('q')}"

        # Attach a schema so _adhoc_defs advertises it. The client only forwards
        # whatever schema its tool carries; for this test we don't need core to
        # validate it. But to exercise the advertisement path, give it one.
        echo.schema = {
            "name": "echo",
            "description": "echo the query",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        }

        result_seen: list[Any] = []

        async def handler(conn: Any) -> None:
            connect = load_endpoint(await conn.recv())
            assert connect.adhoc_tools, "client should advertise its ad-hoc tools"
            assert connect.adhoc_tools[0].tool.name == "echo"
            # Tell the client to run its tool.
            await conn.send(
                dump(
                    ToolCallEvent(
                        thread_id="t", call_id="c1", name="echo", arguments={"q": "hi"}, local=False
                    )
                )
            )
            # Expect a ToolResult frame back.
            result_seen.append(load_endpoint(await conn.recv()))
            await conn.close()

        core = FakeCore(handler=handler)
        await core.start()
        try:
            async with EndpointClient(
                _settings(f"ws://127.0.0.1:{core.port}/link"),
                adhoc_tools={"echo": echo},
            ) as client:
                async for _ in client.events():
                    pass  # iterate so the tool gets dispatched
                await asyncio.sleep(0.05)
        finally:
            await core.stop()

        assert result_seen
        r = result_seen[0]
        assert r.type == "tool_result"
        assert r.call_id == "c1"
        assert r.ok is True
        assert r.output == "echo:hi"

    async def test_unknown_adhoc_tool_replies_error(self) -> None:
        """If core asks for a tool we never registered, we still reply (error)
        so core's dispatcher future doesn't time out."""
        from shared.protocol import ToolCallEvent, dump, load_endpoint

        result_seen: list[Any] = []

        async def handler(conn: Any) -> None:
            await conn.recv()  # Connect
            await conn.send(
                dump(
                    ToolCallEvent(
                        thread_id="t", call_id="c2", name="nope_local", arguments={}, local=False
                    )
                )
            )
            result_seen.append(load_endpoint(await conn.recv()))
            await conn.close()

        core = FakeCore(handler=handler)
        await core.start()
        try:
            async with EndpointClient(
                _settings(f"ws://127.0.0.1:{core.port}/link"),
            ) as client:
                async for _ in client.events():
                    pass
                await asyncio.sleep(0.05)
        finally:
            await core.stop()

        assert result_seen
        r = result_seen[0]
        assert r.ok is False
        assert "nope_local" in r.error
        assert r.call_id == "c2"

    async def test_local_tool_call_is_not_executed_endpoint_side(self) -> None:
        """ToolCallEvent with local=True is just rendered (iterated), not
        dispatched to an ad-hoc runnable -- core ran it itself."""
        from shared.protocol import ToolCallEvent, dump

        executed: list[dict[str, Any]] = []

        async def my_tool(args: dict[str, Any]) -> str:
            executed.append(args)
            return "should-not-happen"

        my_tool.schema = {
            "name": "my_tool",
            "description": "",
            "parameters": {"type": "object", "properties": {}},
        }

        async def handler(conn: Any) -> None:
            await conn.recv()  # Connect
            await conn.send(
                dump(
                    ToolCallEvent(
                        thread_id="t", call_id="c3", name="my_tool", arguments={}, local=True
                    )
                )
            )
            await conn.close()

        core = FakeCore(handler=handler)
        await core.start()
        frames: list[Any] = []
        try:
            async with EndpointClient(
                _settings(f"ws://127.0.0.1:{core.port}/link"),
                adhoc_tools={"my_tool": my_tool},
            ) as client:
                async for frame in client.events():
                    frames.append(frame)
        finally:
            await core.stop()

        assert any(f.type == "tool_call" and f.call_id == "c3" for f in frames)
        assert executed == []  # local tool was NOT run on the endpoint

    async def test_auth_failure_raises_autherror(self) -> None:
        """A rejected upgrade (wrong token) surfaces as AuthError, not a hang."""

        async def handler(conn: Any) -> None:
            await conn.close()

        core = FakeCore(token="right", handler=handler)
        await core.start()
        try:
            client = EndpointClient(_settings(f"ws://127.0.0.1:{core.port}/link", token="wrong"))
            with pytest.raises(AuthError):
                await client.connect()
            await client.close()
        finally:
            await core.stop()

    async def test_send_before_connect_raises(self) -> None:
        client = EndpointClient(_settings("ws://127.0.0.1:1/link"))
        with pytest.raises(RuntimeError):
            await client.send_user_message("x")
