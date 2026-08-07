"""Tests for :class:`endpoint.ui.repl.REPL`.

We stand up an in-process ``websockets`` server that plays core's side of the
link: on each inbound ``UserMessage`` it streams a scripted set of frames and
closes the turn with a ``FinalEvent``. The REPL reads scripted stdin lines and
writes rendered output to captured ``StringIO`` streams so we can assert what
the user would see, including tool calls/results and conversational thread
continuity across turns.
"""

from __future__ import annotations

import asyncio
import io
import json
from typing import Any

import pytest

from endpoint.endpoint_conn.client import EndpointClient
from endpoint.settings import CoreSettings, EndpointSettings, UISettings
from endpoint.ui import repl as repl_mod
from shared.protocol import dump, load_endpoint


# --------------------------------------------------------------------------- #
# Fake core WS server (a compact copy of the one in test_client.py; kept local
# so the REPL test module is self-contained and readable).
# --------------------------------------------------------------------------- #


class FakeCore:
    def __init__(self, *, token: str, handler) -> None:  # type: ignore[no-untyped-def]
        self._token = token
        self._handler = handler
        self._server: Any = None
        self.port = 0

    async def start(self) -> None:
        from websockets.asyncio.server import serve

        self._server = await serve(self._serve, "127.0.0.1", 0, max_size=None)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _serve(self, conn: Any) -> None:
        if conn.request.headers.get("Authorization", "") != f"Bearer {self._token}":
            await conn.close(code=1008, reason="unauthorized")
            return
        await self._handler(conn)


def _settings(core_url: str, *, token: str, stream_tokens: bool = True) -> EndpointSettings:
    return EndpointSettings(
        core=CoreSettings(url=core_url, connect_timeout=5.0, idle_timeout=10.0),
        ui=UISettings(stream_tokens=stream_tokens),
        link_token=token,
    )


def _patch_stdin(lines: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Feed ``lines`` to the REPL's ``_read_line`` one per call; EOF after."""
    it = iter(lines)

    def fake_read_line(_prompt: str) -> str | None:
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(repl_mod, "_read_line", fake_read_line)


async def _run_repl(  # type: ignore[no-untyped-def]
    settings, *, endpoint_id: str | None = None, out: io.StringIO, err: io.StringIO
) -> int:
    """Drive run_repl with captured stdout/stderr."""
    # We bypass run_repl's own connect-error path by constructing the client
    # and REPL directly so we can point the REPL's streams at our buffers.
    client = EndpointClient(settings, endpoint_id=endpoint_id)
    try:
        await client.connect()
    except Exception as e:  # pragma: no cover - connect failure is test infra
        print(f"error: could not connect to core: {e}", file=err)
        return 1
    repl = repl_mod.REPL(client, settings)
    repl._out = out  # type: ignore[assignment]
    repl._err = err  # type: ignore[assignment]
    try:
        await repl.run()
        return 0
    finally:
        await client.close()


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestREPL:
    async def test_streams_tokens_and_marks_final(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A turn renders streamed tokens inline and ends with a newline."""
        from shared.protocol import FinalEvent, TokenEvent

        async def handler(conn: Any) -> None:
            await conn.recv()  # Connect
            await conn.recv()  # UserMessage (turn 1)
            await conn.send(dump(TokenEvent(thread_id="t1", delta="Hel")))
            await conn.send(dump(TokenEvent(thread_id="t1", delta="lo")))
            await conn.send(dump(FinalEvent(thread_id="t1", text="Hello")))
            # Keep the socket open until the client closes after EOF; any further
            # reads will block -- the REPL never iterates again once it saw final.
            await conn.recv()  # blocks; server task ends when client closes

        core = FakeCore(token="right", handler=handler)
        await core.start()
        out, err = io.StringIO(), io.StringIO()
        _patch_stdin(["hi", "/exit"], monkeypatch)
        try:
            rc = await _run_repl(
                _settings(f"ws://127.0.0.1:{core.port}/link", token="right"),
                out=out, err=err,
            )
        finally:
            await core.stop()

        assert rc == 0
        text = out.getvalue()
        assert "Hello" in text  # streamed tokens (Hel+lo) form "Hello"
        # The final token followed by a newline means "Hello\n" appears.
        assert "Hello\n" in text

    async def test_tool_call_and_result_rendered(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from shared.protocol import FinalEvent, ToolCallEvent, ToolResultEvent

        async def handler(conn: Any) -> None:
            await conn.recv()  # Connect
            await conn.recv()  # UserMessage
            await conn.send(
                dump(ToolCallEvent(thread_id="t", call_id="c1", name="http_fetch",
                                    arguments={"url": "https://x"}, local=True))
            )
            await conn.send(
                dump(ToolResultEvent(thread_id="t", call_id="c1", ok=True,
                                      output="some bytes"))
            )
            await conn.send(dump(FinalEvent(thread_id="t", text="done")))
            await conn.recv()  # block until client closes

        core = FakeCore(token="right", handler=handler)
        await core.start()
        out, err = io.StringIO(), io.StringIO()
        _patch_stdin(["go", "/exit"], monkeypatch)
        try:
            rc = await _run_repl(
                _settings(f"ws://127.0.0.1:{core.port}/link", token="right"),
                out=out, err=err,
            )
        finally:
            await core.stop()

        text = out.getvalue()
        assert rc == 0
        assert "tool_call core" in text and "http_fetch" in text
        # show_tool_args default True -> the args JSON shows up.
        assert "https://x" in text
        assert "tool_result ok" in text and "some bytes" in text
        assert "done" in text

    async def test_no_stream_prints_final_text_once(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from shared.protocol import FinalEvent, TokenEvent

        async def handler(conn: Any) -> None:
            await conn.recv()
            await conn.recv()
            await conn.send(dump(TokenEvent(thread_id="t", delta="ignored")))
            await conn.send(dump(FinalEvent(thread_id="t", text="FINALBODY")))
            await conn.recv()

        core = FakeCore(token="right", handler=handler)
        await core.start()
        out, err = io.StringIO(), io.StringIO()
        _patch_stdin(["q", "/exit"], monkeypatch)
        try:
            rc = await _run_repl(
                _settings(f"ws://127.0.0.1:{core.port}/link", token="right",
                          stream_tokens=False),
                out=out, err=err,
            )
        finally:
            await core.stop()

        text = out.getvalue()
        assert rc == 0
        assert "ignored" not in text  # tokens suppressed in no-stream mode
        assert "FINALBODY" in text  # final body printed once

    async def test_thread_continuity_across_turns(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The second turn reuses the thread_id pinned from the first turn."""
        from shared.protocol import FinalEvent

        seen_thread_ids: list[str | None] = []

        async def handler(conn: Any) -> None:
            await conn.recv()  # Connect
            # Turn 1: client sends with thread_id=None; core allocates "fixed".
            seen_thread_ids.append(load_endpoint(await conn.recv()).thread_id)
            await conn.send(dump(FinalEvent(thread_id="fixed", text="a")))
            # Turn 2: client should now send thread_id == "fixed".
            seen_thread_ids.append(load_endpoint(await conn.recv()).thread_id)
            await conn.send(dump(FinalEvent(thread_id="fixed", text="b")))
            await conn.recv()  # block until close

        core = FakeCore(token="right", handler=handler)
        await core.start()
        out, err = io.StringIO(), io.StringIO()
        _patch_stdin(["one", "two", "/exit"], monkeypatch)
        try:
            await _run_repl(
                _settings(f"ws://127.0.0.1:{core.port}/link", token="right"),
                out=out, err=err,
            )
        finally:
            await core.stop()

        assert seen_thread_ids == [None, "fixed"]

    async def test_error_event_renders_to_stderr(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from shared.protocol import ErrorEvent, FinalEvent

        async def handler(conn: Any) -> None:
            await conn.recv()
            await conn.recv()
            await conn.send(dump(ErrorEvent(thread_id="t", message="oops", fatal=False)))
            await conn.send(dump(FinalEvent(thread_id="t", text="ok")))
            await conn.recv()

        core = FakeCore(token="right", handler=handler)
        await core.start()
        out, err = io.StringIO(), io.StringIO()
        _patch_stdin(["x", "/exit"], monkeypatch)
        try:
            rc = await _run_repl(
                _settings(f"ws://127.0.0.1:{core.port}/link", token="right"),
                out=out, err=err,
            )
        finally:
            await core.stop()

        assert rc == 0
        assert "oops" in err.getvalue()
        # Non-fatal error keeps the turn going to final.
        assert "ok" in out.getvalue()

    async def test_eof_exits_cleanly(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from shared.protocol import FinalEvent

        async def handler(conn: Any) -> None:
            await conn.recv()
            await conn.recv()
            await conn.send(dump(FinalEvent(thread_id="t", text="hi")))
            await conn.recv()

        core = FakeCore(token="right", handler=handler)
        await core.start()
        out, err = io.StringIO(), io.StringIO()
        # Only one input line, then EOF (no /exit). _read_line raises EOFError.
        _patch_stdin(["hello"], monkeypatch)
        try:
            rc = await _run_repl(
                _settings(f"ws://127.0.0.1:{core.port}/link", token="right"),
                out=out, err=err,
            )
        finally:
            await core.stop()

        assert rc == 0
        assert "hi" in out.getvalue()
        assert "Bye." in out.getvalue()

    async def test_blank_lines_are_ignored(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from shared.protocol import FinalEvent

        sent: list[str] = []

        async def handler(conn: Any) -> None:
            await conn.recv()
            msg = load_endpoint(await conn.recv())
            sent.append(msg.content)
            await conn.send(dump(FinalEvent(thread_id="t", text="ok")))
            await conn.recv()

        core = FakeCore(token="right", handler=handler)
        await core.start()
        out, err = io.StringIO(), io.StringIO()
        _patch_stdin(["", "   ", "real", "/exit"], monkeypatch)
        try:
            rc = await _run_repl(
                _settings(f"ws://127.0.0.1:{core.port}/link", token="right"),
                out=out, err=err,
            )
        finally:
            await core.stop()

        assert rc == 0
        # Only the non-blank line became a UserMessage.
        assert sent == ["real"]
