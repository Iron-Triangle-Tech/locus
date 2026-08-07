"""stdin/stdout REPL that drives an :class:`EndpointClient`.

The REPL is the human-facing layer; it owns rendering only. It does not know
the wire protocol beyond the pydantic event types, and it delegates all
transport to :class:`endpoint.endpoint_conn.client.EndpointClient`.

Interaction model (v1):
* One connection to core lives for the whole session.
* Each non-empty input line starts an agent turn on the **current thread**
  (or a fresh one if we haven't seen any events yet). The echoed ``thread_id``
  on the first event of the turn pins ``self._thread_id`` so the next line
  continues the conversation.
* A turn's frames are rendered inline as they stream: tokens appended to the
  current line, tool calls/results on their own indented lines, and the
  ``final`` frame closes the turn with a newline.
* ``/exit`` or Ctrl-D (EOF) ends the session cleanly.

The REPL is intentionally synchronous-looking under the hood: we run one event
loop, read stdin with :func:`asyncio.to_thread` so we don't block the loop, and
drive the client's ``events()`` async generator per turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import TYPE_CHECKING, Any

from shared.protocol import (
    ErrorEvent,
    FinalEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)

if TYPE_CHECKING:
    from endpoint.endpoint_conn.client import EndpointClient
    from endpoint.settings import EndpointSettings

__all__ = ["REPL", "run_repl"]

_log = logging.getLogger(__name__)


class REPL:
    """A minimal line-oriented REPL over an :class:`EndpointClient`."""

    PROMPT = "> "

    def __init__(self, client: "EndpointClient", settings: "EndpointSettings") -> None:
        self._client = client
        self._settings = settings
        # Thread continuity: pinned from the first event's thread_id, then reused
        # for subsequent sends so a conversation stays in one thread.
        self._thread_id: str | None = None
        self._out = sys.stdout
        self._err = sys.stderr

    async def run(self) -> None:
        """Read input lines, drive turns, until /exit or EOF."""
        self._print("Connected to core. Type a message; /exit or Ctrl-D to quit.")
        loop = asyncio.get_running_loop()
        while True:
            try:
                line = await loop.run_in_executor(None, _read_line, self.PROMPT)
            except EOFError:
                self._print()  # newline after the prompt
                break
            if line is None:
                # EOF from the executor fallback path.
                break
            text = line.strip()
            if not text:
                continue
            if text in {"/exit", "/quit"}:
                break
            try:
                await self._run_turn(text)
            except Exception as e:  # surface turn failures, keep the session up
                self._render_error(ErrorEvent(thread_id=self._thread_id, message=str(e)))
        self._print("Bye.")

    async def _run_turn(self, content: str) -> None:
        """Send a user message and render every frame core streams back.

        Stops iterating once a ``FinalEvent`` (or a fatal ``ErrorEvent``) lands,
        which is how the loop signals end-of-turn.
        """
        await self._client.send_user_message(
            content, thread_id=self._thread_id
        )
        started_line = False
        async for frame in self._client.events():  # noqa: B007 -- break handled below
            # Pin the thread id from the first frame that carries one.
            tid = getattr(frame, "thread_id", None)
            if tid and self._thread_id is None:
                self._thread_id = tid
            if isinstance(frame, TokenEvent):
                if self._settings.ui.stream_tokens:
                    self._out.write(frame.delta)
                    self._out.flush()
                    started_line = True
            elif isinstance(frame, ToolCallEvent):
                if started_line:
                    self._print()
                    started_line = False
                self._render_tool_call(frame)
            elif isinstance(frame, ToolResultEvent):
                if started_line:
                    self._print()
                    started_line = False
                self._render_tool_result(frame)
            elif isinstance(frame, FinalEvent):
                if started_line:
                    # Tokens already rendered the assistant text inline; the
                    # final frame just closes the line.
                    self._print()
                    started_line = False
                elif frame.text:
                    # No tokens were shown inline this turn (no-stream mode,
                    # or a tool-only turn that produced zero tokens); print the
                    # assembled final text so the assistant's answer isn't lost.
                    self._print(frame.text)
                return
            elif isinstance(frame, ErrorEvent):
                if started_line:
                    self._print()
                    started_line = False
                self._render_error(frame)
                if frame.fatal:
                    return
            else:  # pragma: no cover - defensive: future frame types
                if started_line:
                    self._print()
                    started_line = False
                self._print(f"[unhandled frame: {frame!r}]")
        # Stream ended without a FinalEvent (link dropped). Ensure we newline.
        if started_line:
            self._print()

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render_tool_call(self, frame: ToolCallEvent) -> None:
        who = "core" if frame.local else "endpoint"
        if self._settings.ui.show_tool_args:
            args = json.dumps(frame.arguments, default=str)
            self._print(f"  [tool_call {who}] {frame.name}({args})")
        else:
            self._print(f"  [tool_call {who}] {frame.name}")

    def _render_tool_result(self, frame: ToolResultEvent) -> None:
        if frame.ok:
            out = (frame.output or "").rstrip()
            self._print(f"  [tool_result ok] {out}")
        else:
            self._print(f"  [tool_result ERROR] {frame.error or 'unknown error'}")

    def _render_error(self, frame: ErrorEvent) -> None:
        prefix = "FATAL" if frame.fatal else "error"
        self._eprint(f"  [{prefix}] {frame.message}")

    # Thin wrappers so tests can capture output by monkeypatching the REPL's
    # _out/_err streams (kept as instance attrs for that reason).
    def _print(self, *args: Any) -> None:
        print(*args, file=self._out, flush=True)

    def _eprint(self, *args: Any) -> None:
        print(*args, file=self._err, flush=True)


def _read_line(prompt: str) -> str | None:
    """Read one line from stdin (blocking, run in an executor). Returns None on
    EOF, raises EOFError if input() returns empty at EOF (Python's input raises
    EOFError on Ctrl-D). """
    try:
        return input(prompt)
    except EOFError:
        raise
    except OSError:
        return None


async def run_repl(
    settings: "EndpointSettings",
    *,
    endpoint_id: str | None = None,
    adhoc_tools: dict[str, Any] | None = None,
) -> int:
    """Connect to core and run the REPL. Returns a process exit code."""
    from endpoint.endpoint_conn.client import EndpointClient

    client = EndpointClient(
        settings, endpoint_id=endpoint_id, adhoc_tools=adhoc_tools
    )
    try:
        try:
            await client.connect()
        except Exception as e:
            print(f"error: could not connect to core: {e}", file=sys.stderr)
            return 1
        repl = REPL(client, settings)
        await repl.run()
        return 0
    finally:
        await client.close()
