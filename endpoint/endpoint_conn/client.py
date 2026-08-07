"""Endpoint side of the core <-> endpoint WebSocket link.

:class:`EndpointClient` is the reusable connection a CLI REPL (or a future
webui) drives. It owns one outbound WS to core and the small piece of policy
that turns core's stream of agent events into either rendered output or, for
ad-hoc tools the endpoint advertised, local execution + a ``ToolResult`` back.

Responsibilities split from the UI deliberately: this module knows about the
wire protocol and the event loop; it does not touch stdin/stdout. ``ui/repl.py``
subscribes to :meth:`EndpointClient.events` and renders.

Design notes:

* **Auth** happens on the WS upgrade via the shared bearer token in the
  ``Authorization`` header (matching core's pre-accept 403 check). Failed
  handshakes raise :class:`AuthError` so the UI can surface "unauthorized"
  rather than retry forever.

* **Ad-hoc tools** are endpoint-local runnables registered up front; their
  :class:`shared.protocol.ToolSchema` defs are advertised in the opening
  ``Connect`` frame so core can offer them to the model. When core routes a
  ``ToolCallEvent`` back with a name we registered, we run it here and send a
  ``ToolResult`` frame; core's :class:`WSLinkAdhocDispatcher` resolves the
  loop's pending future. Built-in tool calls (``local == True``) are just
  rendered; core ran them itself.

* **Reconnect** is intentionally NOT automatic in v1: the first feature is a
  CLI driven by a human, and silently resuming mid-conversation after a link
  drop is surprising. We expose :meth:`close` and let the caller decide to
  re-:meth:`connect`. The underlying ``websockets`` library already handles
  ping/pong keepalive, so we don't re-implement that.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeAlias

from shared.protocol import (
    Connect,
    ErrorEvent,
    ToolCallEvent,
    ToolResult,
    UserMessage,
    load_core,
)

if TYPE_CHECKING:
    from endpoint.settings import EndpointSettings

__all__ = ["AdhocTool", "AuthError", "EndpointClient"]

_log = logging.getLogger(__name__)


class AuthError(RuntimeError):
    """Core rejected the WS upgrade (bad/missing token)."""


# An ad-hoc tool runnable the endpoint registers: takes the parsed arguments
# dict and returns either a string (ok) or raises (the result is an error).
AdhocTool: TypeAlias = Callable[[dict[str, Any]], Awaitable[str]]


class EndpointClient:
    """One persistent WS connection to core, plus ad-hoc tool dispatch.

    Use as an async context manager::

        async with EndpointClient(settings) as client:
            await client.send_user_message("hello")
            async for frame in client.events():
                ...

    Or call :meth:`connect` / :meth:`close` manually when the UI wants full
    control over the lifecycle (e.g. a REPL that reconnects per command).

    The client is single-threaded by construction: one task reads frames and
    yields them from :meth:`events`; the caller dispatches. Tool execution
    runs on the same loop so a slow ad-hoc tool naturally backpressures the
    read loop (frame iteration pauses while we await the tool). That's fine
    for v1; if we ever host genuinely concurrent turns we'll fan tool work
    out onto tasks.
    """

    def __init__(
        self,
        settings: EndpointSettings,
        *,
        endpoint_id: str | None = None,
        adhoc_tools: dict[str, AdhocTool] | None = None,
    ) -> None:
        self._settings = settings
        self._endpoint_id = endpoint_id or "endpoint-1"
        # name -> (runnable). Defs are derived from the runnables' declared
        # schemas at connect time, so callers register concrete callables.
        self._adhoc: dict[str, AdhocTool] = dict(adhoc_tools or {})
        self._ws: Any | None = None  # websockets.ClientConnection once connected
        self._closed = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def __aenter__(self) -> EndpointClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def connect(self) -> None:
        """Open the WS to core, authenticate, and send the ``Connect`` frame.

        Raises :class:`AuthError` if core rejects the upgrade (we can't tell a
        403 from any other non-101 status with the ``websockets`` client's
        default exception types, so we treat any handshake failure that isn't
        a clean close as auth/config trouble to surface to the user).
        """
        from websockets.asyncio.client import connect as ws_connect
        from websockets.exceptions import ConnectionClosed, InvalidStatus

        # str (not bytes) header tuples: websockets' Headers normalizes these
        # so core's case-insensitive ``Authorization`` lookup matches.
        headers = [("authorization", f"Bearer {self._settings.link_token}")]
        try:
            self._ws = await ws_connect(
                self._settings.core.url,
                additional_headers=headers,
                open_timeout=self._settings.core.connect_timeout,
                ping_interval=20.0,
                ping_timeout=20.0,
                max_size=None,  # agent turns can stream large bodies
                proxy=self._settings.core.proxy,  # default None -> never proxy
            )
        except InvalidStatus as e:
            # A 4xx error on the upgrade handshake.
            raise AuthError(
                f"core rejected the link handshake "
                f"(status {e.response.status_code if e.response else '?'})"
            ) from e
        # Announce ourselves + the ad-hoc tools we can run. Core merges these
        # into the toolset it offers the model; when one is invoked, core
        # routes the ToolCallEvent back here and we run it. Some servers (and
        # core's belt-and-braces re-check) close the link *after* the handshake
        # with code 1008 policy-violation on a bad token; that surfaces as a
        # ConnectionClosed on the first send, so we translate it to AuthError.
        connect_frame = Connect(
            endpoint_id=self._endpoint_id,
            adhoc_tools=_adhoc_defs(self._adhoc),
        )
        try:
            await self._send_raw(connect_frame.model_dump_json())
        except ConnectionClosed as e:
            await self.close()
            raise AuthError("core closed the link on connect (auth rejected?)") from e
        _log.info(
            "endpoint connected to core: id=%s adhoc_tools=%d",
            self._endpoint_id,
            len(connect_frame.adhoc_tools),
        )

    async def close(self) -> None:
        """Close the link cleanly. Idempotent; safe to call more than once."""
        self._closed = True
        if self._ws is None:
            return
        try:
            await self._ws.close()
        except Exception:  # pragma: no cover - defensive on teardown
            pass
        self._ws = None

    # ------------------------------------------------------------------ #
    # Outbound: UserMessage
    # ------------------------------------------------------------------ #

    async def send_user_message(
        self,
        content: str,
        *,
        thread_id: str | None = None,
        provider: str = "auto",
    ) -> None:
        """Start an agent turn by sending a ``UserMessage``.

        ``thread_id`` omitted starts a fresh thread on core. The created/used
        id echoes back on every event's ``thread_id`` so the UI can label it.
        """
        if self._ws is None:
            raise RuntimeError("EndpointClient not connected (call connect() first)")
        await self._send_raw(
            UserMessage(content=content, thread_id=thread_id, provider=provider).model_dump_json()
        )

    # ------------------------------------------------------------------ #
    # Inbound: event stream
    # ------------------------------------------------------------------ #

    async def events(self) -> AsyncIterator[Any]:
        """Yield parsed core->endpoint frames until the link closes.

        ``ToolCallEvent`` frames targeting an ad-hoc tool the endpoint
        registered are *executed* here and their ``ToolResult`` is sent back;
        the caller still sees the event (to render "running tool X") but the
        result round-trip is handled internally so the UI doesn't have to know
        the dispatch protocol.

        Yields plain pydantic model instances (``TokenEvent`` etc.). A
        ``ErrorEvent`` with ``fatal`` ends iteration after being yielded.
        """
        if self._ws is None:
            raise RuntimeError("EndpointClient not connected (call connect() first)")
        async for raw in self._ws:  # type: ignore[union-attr]
            frame = load_core(raw)
            yield frame
            if isinstance(frame, ToolCallEvent) and not frame.local:
                await self._maybe_run_adhoc_or_reject(frame)
            if isinstance(frame, ErrorEvent) and frame.fatal:
                _log.warning("fatal error from core: %s", frame.message)
                break

    # ------------------------------------------------------------------ #
    # Ad-hoc tool dispatch
    # ------------------------------------------------------------------ #

    async def _maybe_run_adhoc_or_reject(self, frame: ToolCallEvent) -> None:
        """Run a named ad-hoc tool if we registered it; else reject with error.

        Core's dispatcher awaits a future keyed by ``call_id``; if we never
        send a ``ToolResult`` that future times out, which stalls the whole
        turn. So we always reply -- error if the tool is unknown to us too.
        """
        runnable = self._adhoc.get(frame.name)
        if runnable is None:
            await self._send_raw(
                ToolResult(
                    call_id=frame.call_id,
                    ok=False,
                    error=f"endpoint has no ad-hoc tool named {frame.name!r}",
                ).model_dump_json()
            )
            return
        try:
            # frame.arguments is already a parsed dict (shared.protocol parses
            # it via pydantic), not a JSON string; pass it straight through.
            output = await runnable(frame.arguments or {})
            # ToolResult carries only a string output; coerce non-str results.
            if not isinstance(output, str):
                output = json.dumps(output)
            result = ToolResult(call_id=frame.call_id, ok=True, output=output)
        except Exception as e:  # surface tool failure to core as a ToolResult
            _log.exception("ad-hoc tool %s failed: %s", frame.name, e)
            result = ToolResult(
                call_id=frame.call_id,
                ok=False,
                error=f"{type(e).__name__}: {e}",
            )
        await self._send_raw(result.model_dump_json())

    async def _send_raw(self, text: str) -> None:
        assert self._ws is not None
        await self._ws.send(text)


# ----------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------- #


def _adhoc_defs(adhoc: dict[str, AdhocTool]) -> list:
    """Build :class:`shared.protocol.AdhocTool` defs from registered runnables.

    In v1 the endpoint's ad-hoc tools are simple callables; they declare their
    own schema by carrying attributes the caller populated when registering.
    The first feature ships with no built-in ad-hoc tools on the endpoint (the
    browser/sandbox tools are out of scope), so this returns ``[]`` unless a
    caller registers one with a ``schema`` attribute.
    """
    from shared.protocol import AdhocTool, ToolSchema

    defs: list[AdhocTool] = []
    for name, runnable in adhoc.items():
        schema = getattr(runnable, "schema", None)
        if schema is None:
            # No declared schema -> skip rather than fabricate one; core
            # would offer the model a tool it can't call meaningfully.
            _log.warning("ad-hoc tool %r has no `schema` attribute; not advertising it", name)
            continue
        # ``schema`` is expected to be a ToolSchema-shaped dict; build the
        # pydantic model defensively so a bad registration is caught here,
        # not deep inside core's merge.
        defs.append(AdhocTool(tool=ToolSchema(**schema)))
    return defs
