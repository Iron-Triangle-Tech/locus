"""Core side of the core <-> endpoint WebSocket link.

This is where the agent loop's transport-agnostic ad-hoc dispatch meets the
network. Two pieces live here:

* :class:`LinkRegistry` -- a process-wide table of pending endpoint tool calls
  keyed by ``call_id``. :class:`WSLinkAdhocDispatcher` registers a future for a
  call when the loop asks it to resolve an ad-hoc call, and waits on that
  future. When the matching :class:`shared.protocol.ToolResult` frame arrives
  over the WS link, :func:`handle_link` resolves the future and the loop's
  await returns.

* :func:`handle_link` -- the per-connection coroutine run by the FastAPI WS
  endpoint. It:

    1. Authenticates the endpoint with the shared bearer token (rejects with a
       ``1008`` close if the token is missing or wrong).
    2. Waits for the endpoint's :class:`Connect` frame (rejects any other
       first frame; the endpoint must announce itself before doing anything
       else). The ad-hoc tools it advertises are accepted for protocol
       validity; merging them into the agent loop's offered toolset is a
       Step 9/10 refinement (the loop's ``_adhoc_defs`` returns ``[]`` for
       now), so in v1 the loop can only *invoke* an ad-hoc tool if something
       else already put its def into the registry.
    3. Spawns a bus-pump task that subscribes a fresh :class:`EventBus`
       subscription and forwards each agent event to the socket as JSON.
       All agent events for any thread on this core are visible to every
       authenticated endpoint in v1; per-connection thread filtering is a
       later refinement.
    4. Reads inbound frames in a loop. :class:`UserMessage` starts an agent
       turn on a background task so the link can interleave streaming to the
       client while still draining results the endpoint might send mid-turn.
       :class:`ToolResult` resolves the matching pending future.
       :class:`Disconnect` ends the connection cleanly.

The handler never blocks the loop on a single endpoint: agent turns run as
background tasks, and the reader loop only resolves futures / spawns work.

Why a registry and not a single dispatcher: the core can host many endpoints
at once, and the loop is shared. The registry lets any endpoint's
``ToolResult`` resolve the right future regardless of which connection
happened to carry the call; in v1 actual routing is implicit (the call's
future is process-global by ``call_id``), and a later refinement will pin a
call to the endpoint that advertised the tool.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from shared.auth import check_token
from shared.protocol import (
    Connect,
    Disconnect,
    ErrorEvent,
    ToolResult,
    UserMessage,
    dump,
    load_endpoint,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from fastapi import WebSocket

    from core.bus import EventBus
    from core.providers.base import ToolCall, ToolResultMessage

__all__ = [
    "LinkRegistry",
    "WSLinkAdhocDispatcher",
    "handle_link",
]

_log = logging.getLogger(__name__)

# Seconds to wait for an endpoint's ToolResult once the loop asks us to
# dispatch an ad-hoc call. The loop sets its own max_iters guard; this is a
# per-call network timeout so a dead endpoint doesn't pin a turn forever. Kept
# generous because ad-hoc tools can legitimately be slow (browser, long jobs).
DEFAULT_ADHOC_TIMEOUT = 120.0


class LinkRegistry:
    """Process-wide table of pending endpoint tool calls keyed by ``call_id``.

    Each pending call owns an :class:`asyncio.Future`. The loop's dispatcher
    awaits it; an inbound :class:`ToolResult` resolves it. Resolving a future
    that isn't pending is a no-op log (the endpoint sent a result we're not
    waiting for -- late dupe, or a call we already timed out / cancelled).

    Thread-safety: every method runs on the event loop. No locks needed.
    """

    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Future[ToolResult]] = {}

    def register(self, call_id: str) -> "asyncio.Future[ToolResult]":
        """Create + remember a future for ``call_id``.

        Raises if a call with this id is already pending -- call ids are
        Core-generated UUIDs, so a collision means a bug.
        """
        if call_id in self._pending:
            raise ValueError(f"ad-hoc call already pending: {call_id!r}")
        fut: asyncio.Future[ToolResult] = asyncio.Future()
        self._pending[call_id] = fut
        return fut

    def resolve(self, result: ToolResult) -> bool:
        """Resolve the pending future for ``result.call_id`` if there is one.

        Returns True iff a future was actually resolved (i.e. the endpoint
        sent a result we were still waiting for). Late / unknown results are
        dropped with a debug log.
        """
        fut = self._pending.pop(result.call_id, None)
        if fut is None or fut.done():
            _log.debug(
                "ad-hoc result with no pending call (late/dupe?): call_id=%s",
                result.call_id,
            )
            return False
        if not fut.cancelled():
            fut.set_result(result)
        return True

    def cancel(self, call_id: str) -> bool:
        """Cancel a pending future (e.g. the owning turn errored out)."""
        fut = self._pending.pop(call_id, None)
        if fut is None:
            return False
        if not fut.done():
            fut.cancel()
        return True

    def cancel_all(self) -> None:
        """Cancel every pending future. Used on connection teardown."""
        for call_id in list(self._pending):
            self.cancel(call_id)

    @property
    def pending(self) -> int:
        return len(self._pending)


class WSLinkAdhocDispatcher:
    """Implements :class:`core.agent.loop.AdhocDispatcher` over a registry.

    One instance per core app is fine: it never holds per-connection state --
    it just borrows the shared registry. The loop calls
    :meth:`dispatch`; this registers a future, awaits it, and translates the
    resolved :class:`ToolResult` into the loop-internal
    :class:`ToolResultMessage`.

    If the endpoint never answers within ``timeout`` seconds, the future is
    cancelled and the loop gets an error result it can feed back to the model.
    """

    def __init__(
        self,
        registry: LinkRegistry,
        *,
        timeout: float = DEFAULT_ADHOC_TIMEOUT,
    ) -> None:
        self._registry = registry
        self._timeout = timeout

    async def dispatch(self, call: "ToolCall") -> "ToolResultMessage":
        fut = self._registry.register(call.id)
        try:
            result = await asyncio.wait_for(fut, timeout=self._timeout)
        except asyncio.TimeoutError:
            _log.warning("ad-hoc tool timed out: call_id=%s name=%s", call.id, call.name)
            return _to_message(
                ToolResult(
                    call_id=call.id,
                    ok=False,
                    error=f"endpoint did not return a result within {self._timeout:g}s",
                ),
                name=call.name,
            )
        except asyncio.CancelledError:
            # The turn was cancelled (connection torn down, app shutting down).
            # Surface as an error so the loop's caller sees a clean result.
            return _to_message(
                ToolResult(
                    call_id=call.id,
                    ok=False,
                    error="ad-hoc dispatch cancelled",
                ),
                name=call.name,
            )
        finally:
            # wait_for cancels the future on timeout; let the registry forget
            # any stragglers. resolve/cancel is idempotent.
            self._registry.cancel(call.id)

        return _to_message(result, name=call.name)


def _to_message(result: ToolResult, *, name: str) -> "ToolResultMessage":
    """Translate a wire :class:`ToolResult` into the loop-internal type."""
    from core.providers.base import ToolResultMessage

    return ToolResultMessage(
        call_id=result.call_id,
        name=name,
        content=result.output if result.ok and result.output is not None
        else (result.error or ""),
        is_error=not result.ok,
    )


async def handle_link(
    websocket: "WebSocket",
    *,
    bus: "EventBus",
    registry: LinkRegistry,
    expected_token: str,
    on_user_message: "UserMessageHandler",
    idle_timeout: float,
) -> None:
    """Drive one endpoint's WS connection to completion (or close).

    Parameters
    ----------
    websocket
        The accepted FastAPI/Starlette WebSocket (already accepted by the
        caller; we just own the read/write loop). We do NOT accept it here so
        the caller can decide close codes on auth failure cleanly.
    bus
        The in-process :class:`EventBus` to subscribe for agent events.
    registry
        Where inbound :class:`ToolResult` frames resolve pending dispatcher
        futures. One per core app; shared across endpoints.
    expected_token
        The bearer token the endpoint must present.
    on_user_message
        Callback invoked for each inbound :class:`UserMessage`. Should start a
        background turn and return promptly -- the link's reader loop must not
        block waiting on the agent to finish (we need to keep draining results
        the endpoint might send mid-turn).
    idle_timeout
        Seconds with no inbound frame before we ping-close the link.
    """
    # Fast pre-accept auth: Starlette exposes the upgrade headers before the
    # WS is "accepted". We can't send a close *after* accept here, so the
    # caller checks the token and rejects before calling us; this is a
    # belt-and-braces re-check on the opened socket in case a misconfigured
    # caller forgets.
    headers = {k: v for k, v in websocket.headers.items()}
    if not expected_token or not check_token(headers, expected_token):
        _log.warning("endpoint link rejected: bad/missing token")
        try:
            await websocket.close(code=1008, reason="unauthorized")
        except Exception:  # pragma: no cover - defensive
            pass
        return

    sub = bus.subscribe()
    pump_task = asyncio.create_task(
        _pump_events(websocket, sub), name="ws-link-event-pump"
    )
    reader_task = asyncio.create_task(
        _reader(websocket, registry, on_user_message, idle_timeout),
        name="ws-link-reader",
    )
    try:
        # Either side can end the connection; whichever finishes first wins,
        # and we cancel the other.
        done, pending = await asyncio.wait(
            {pump_task, reader_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
    finally:
        bus.unsubscribe(sub)
        # Don't leave the loop's dispatcher waiting forever if the endpoint
        # vanished mid-turn. The loop will see cancelled futures and emit its
        # own error events.
        _log.debug("endpoint link closed; link pending=%d", registry.pending)


async def _pump_events(websocket: "WebSocket", sub) -> None:  # type: ignore[no-untyped-def]
    """Forward bus frames -> websocket as JSON, until the bus closes."""
    try:
        async for frame in sub.events():
            await websocket.send_text(dump(frame))
    except Exception as e:  # pragma: no cover - defensive; endpoint went away
        _log.debug("event pump ended: %s", e)


async def _reader(
    websocket: "WebSocket",
    registry: LinkRegistry,
    on_user_message: "UserMessageHandler",
    idle_timeout: float,
) -> None:
    """Inbound frame loop: handle Connect once, then UserMessage/ToolResult.

    Sends an :class:`ErrorEvent` back over the socket (not just logs) on
    malformed frames so the endpoint has a chance to recover rather than
    silently starve. After ``idle_timeout`` with no inbound frame, the reader
    returns and the connection tears down (the pump is cancelled by the
    caller).
    """
    # First frame must be Connect.
    try:
        first = await asyncio.wait_for(websocket.receive_text(), timeout=idle_timeout)
    except (asyncio.TimeoutError, Exception):
        _log.warning("endpoint link: no Connect frame within %gs", idle_timeout)
        return

    try:
        frame = load_endpoint(first)
    except Exception as e:
        await _send_error(websocket, f"invalid first frame: {e}")
        return
    if not isinstance(frame, Connect):
        await _send_error(websocket, "expected a 'connect' frame first")
        return
    _log.info(
        "endpoint connected: id=%s adhoc_tools=%d",
        frame.endpoint_id,
        len(frame.adhoc_tools),
    )

    while True:
        try:
            raw = await asyncio.wait_for(
                websocket.receive_text(), timeout=idle_timeout
            )
        except asyncio.TimeoutError:
            _log.debug("endpoint link idle-timeout after %gs", idle_timeout)
            return
        except Exception:
            # Socket closed / errored; let the caller tear down.
            return

        try:
            frame = load_endpoint(raw)
        except Exception as e:
            await _send_error(websocket, f"invalid frame: {e}")
            continue

        if isinstance(frame, UserMessage):
            # Off-load the turn so the reader can keep drinking results.
            try:
                on_user_message(frame)
            except Exception as e:  # pragma: no cover - defensive
                _log.exception("on_user_message callback failed: %s", e)
                await _send_error(
                    websocket,
                    f"failed to start agent turn: {type(e).__name__}: {e}",
                    thread_id=frame.thread_id,
                )
        elif isinstance(frame, ToolResult):
            registry.resolve(frame)
        elif isinstance(frame, Disconnect):
            _log.info("endpoint disconnected: reason=%s", frame.reason)
            return
        else:
            await _send_error(websocket, f"unexpected frame type: {frame.type!r}")


async def _send_error(
    websocket: "WebSocket", message: str, *, thread_id: str | None = None
) -> None:
    """Best-effort push an :class:`ErrorEvent` to the endpoint."""
    try:
        await websocket.send_text(
            dump(ErrorEvent(thread_id=thread_id, message=message, fatal=False))
        )
    except Exception:  # pragma: no cover - defensive
        pass


# A callable that takes an inbound UserMessage and kicks off an agent turn on
# the caller's side (typically spawned as a background task by app.py). It
# must return promptly and not raise into the link reader. Type-only.
if TYPE_CHECKING:
    UserMessageHandler = Callable[[UserMessage], Any]
