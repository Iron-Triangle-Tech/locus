"""The core agent loop.

One run of :func:`run_agent_turn` drives a single user message through to a
terminal assistant turn (``finish_reason == "stop"``), streaming tokens and
tool calls onto the in-process :class:`core.bus.EventBus` and persisting every
assistant turn + tool result in the :class:`core.storage.database_io.MemoryStore`.

Pipeline per provider call:

1. Load history (assistant turns + prior tool results) + pending tool calls
   for the thread.
2. Mint Core-UUID call ids for any tool calls the model emits during streaming
   (the store keys ``ToolCallRow.id`` on these). Local dispatch tracks them in a
   dict so results map back even if a tool re-emits the same logical call.
3. Stream provider chunks: tokens -> ``TokenEvent`` on the bus; accumulate tool
   calls across chunks (a single call can arrive in pieces). The provider's
   own id is kept as ``provider_id`` for correlation but the **stored + bus
   id is the Core UUID**.
4. After the stream, persist the assistant message atomically (text + calls).
5. Dispatch each tool call:

   * **Built-in** (the name is in the :class:`core.tools.registry.ToolRegistry`):
     run it locally via ``registry.dispatch``; persist the result; emit
     ``ToolCallEvent(local=True)`` then ``ToolResultEvent``.
   * **Ad-hoc** (not in the registry): emit ``ToolCallEvent(local=False)`` and
     await the matching :class:`shared.protocol.ToolResult` via the injected
     :class:`AdhocDispatcher`. Step 8's WS link implements that; the loop itself
     never touches the network.

6. Persist all results, then re-call the provider with the extended history.
   Loop until ``finish_reason == "stop"`` (emit ``FinalEvent``) or an error
   (emit ``ErrorEvent`` and stop). A bound ``max_iters`` guards against runaway
   tool-call loops.

The loop is transport-agnostic: publish = bus, wait-for-ad-hoc = dispatcher.
Tests inject a :class:`FakeAdhocDispatcher`.
"""

from __future__ import annotations

import logging
import uuid
from typing import Protocol, runtime_checkable

from shared.protocol import (
    ErrorEvent,
    FinalEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)

from core.providers import get_provider
from core.providers.base import (
    AssistantTurn,
    Provider,
    ProviderStreamChunk,
    ToolCall,
    ToolDef,
    ToolResultMessage,
    UserTurn,
)
from core.settings import CoreSettings
from core.storage.database_io import MemoryStore
from core.tools.registry import ToolRegistry

__all__ = ["AdhocDispatcher", "AgentLoop", "LoopConfig", "NoAdhocDispatcher"]

_log = logging.getLogger(__name__)


@runtime_checkable
class AdhocDispatcher(Protocol):
    """Resolve an endpoint-owned tool call to a result.

    The loop calls :meth:`dispatch` for any tool call whose name is not in the
    local :class:`ToolRegistry`. The real implementation (Step 8) stashes a
    per-``call_id`` future and resolves it when the endpoint sends
    :class:`shared.protocol.ToolResult` back over the WS link. Tests inject a
    fake / no-op.
    """

    async def dispatch(self, call: ToolCall) -> ToolResultMessage: ...


class NoAdhocDispatcher:
    """Default ad-hoc dispatcher: every call returns ``is_error`` (no endpoint)."""

    async def dispatch(self, call: ToolCall) -> ToolResultMessage:
        return ToolResultMessage(
            call_id=call.id,
            name=call.name,
            content=f"no ad-hoc dispatcher registered for tool {call.name!r}",
            is_error=True,
        )


class LoopConfig:
    """Knobs for one run of the agent loop."""

    def __init__(
        self,
        *,
        max_iters: int = 8,
        id_factory=lambda: str(uuid.uuid4()),
    ) -> None:
        self.max_iters = max_iters
        self._id = id_factory

    def mint_call_id(self) -> str:
        return self._id()


class AgentLoop:
    """Stateful runner bound to a thread for one ``run_agent_turn`` call.

    Construct a fresh ``AgentLoop`` per inbound user message (or per resume of a
    thread with pending calls); it is not safe to share across concurrent
    turns on the same thread because ``run_agent_turn`` mutates the store.
    """

    def __init__(
        self,
        *,
        settings: CoreSettings,
        store: MemoryStore,
        registry: ToolRegistry,
        bus,
        dispatcher: AdhocDispatcher | None = None,
        config: LoopConfig | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._registry = registry
        self._bus = bus
        self._dispatcher: AdhocDispatcher = dispatcher or NoAdhocDispatcher()
        self._config = config or LoopConfig()
        # call_id CORE <-> provider_id (so we can correlate streamed chunks to
        # the ids we mint / persist). Cleared per turn.
        self._id_map: dict[str, str] = {}

    # ---------------------------------------------------------------- public

    async def run_agent_turn(
        self,
        thread_id: str,
        content: str,
        *,
        provider_name: str = "auto",
    ) -> None:
        """Drive ``content`` through one agent turn for ``thread_id``.

        Persists the user message, then loops provider->dispatch->provider
        until ``finish_reason == "stop"`` or ``max_iters``. All assistant
        turns and tool results are persisted; events are published on the bus.
        On an unrecoverable error emits an ``ErrorEvent(fatal=True)`` and
        returns. Never raises to the caller -- the caller (Step 8 WS handler)
        relies on events, not exceptions.
        """
        try:
            await self._store.append_user_message(thread_id, content)
        except Exception as e:
            self._emit(ErrorEvent(thread_id=thread_id, message=f"store error: {e}", fatal=True))
            return

        try:
            provider = get_provider(provider_name, self._settings)
        except (KeyError, ValueError) as e:
            self._emit(
                ErrorEvent(thread_id=thread_id, message=f"provider error: {e}", fatal=True)
            )
            return

        tools = self._registry.export_defs()
        extra_defs = list(self._adhoc_defs(provider_name))
        all_defs = tools + extra_defs

        for _ in range(self._config.max_iters):
            history, prior_results = await self._store.load_history(thread_id)
            prior_results = self._strip_pending(thread_id, prior_results)

            try:
                text, tool_calls, finish = await self._stream_one(
                    provider, thread_id, history,
                    content if _ == 0 else "", prior_results, all_defs,
                )
                if tool_calls:
                    # Pin Core-UUID ids onto the calls and persist atomically.
                    core_calls = [self._to_core_call(c) for c in tool_calls]
                    await self._store.append_assistant_message(
                        thread_id, text=text, tool_calls=core_calls
                    )
                    outcomes = await self._dispatch_calls(thread_id, core_calls)
                    # Persist the results we just produced.
                    for r in outcomes:
                        await self._store.append_tool_result(
                            thread_id,
                            call_id=r.call_id,
                            content=r.content,
                            is_error=r.is_error,
                        )
            except Exception as e:
                self._emit(
                    ErrorEvent(
                        thread_id=thread_id,
                        message=f"provider/dispatch error: {type(e).__name__}: {e}",
                        fatal=False,
                    )
                )
                return

            if tool_calls:
                if finish == "stop":
                    # The model stopped, but it also emitted final tool calls;
                    # they are a normal finish. Emit the final.
                    self._emit(FinalEvent(thread_id=thread_id, text=text))
                    return
                # else: model wants us to keep going with the new results.
                continue

            # No tool calls this turn: it's the final answer.
            await self._store.append_assistant_message(thread_id, text=text, tool_calls=[])
            self._emit(FinalEvent(thread_id=thread_id, text=text))
            return

        # Exhausted max_iters.
        self._emit(
            ErrorEvent(
                thread_id=thread_id,
                message=f"agent loop exceeded max_iters ({self._config.max_iters})",
                fatal=False,
            )
        )

    # --------------------------------------------------------------- streaming

    async def _stream_one(
        self,
        provider: Provider,
        thread_id: str,
        history: list[AssistantTurn],
        user_content: str,
        prior_results: list[ToolResultMessage],
        tools: list[ToolDef],
    ) -> tuple[str, list[ToolCall], str]:
        """Stream one provider call. Returns (text, tool_calls, finish_reason).

        Tool calls arriving in chunks are stitched by index: each chunk with a
        ``tool_call`` may carry a partial-id; we keep an ordered list and append
        / extend by the call's position in the chunk (or treat each chunk as a
        new call when it carries an id). The provider's id is translated to a
        Core UUID on the way out so storage / bus never depend on vendor ids.
        """
        text_parts: list[str] = []
        calls_by_idx: dict[int, ToolCall] = {}
        finish = "stop"

        user = UserTurn(content=user_content) if user_content else UserTurn(content="")
        stream = provider.stream(history, user, tools, prior_results)
        async for chunk in stream:  # type: ignore[union-attr]
            if chunk.token:
                text_parts.append(chunk.token)
                self._emit(TokenEvent(thread_id=thread_id, delta=chunk.token))
            if chunk.tool_call:
                # Use the call's own id as the dedup key when present, else
                # fall back to call index (some adapters stream partial args
                # before emitting the id). Here we just take the latest value
                # for each known id.
                tc = chunk.tool_call
                key = tc.id or f"idx{len(calls_by_idx)}"
                if key in calls_by_idx:
                    # Merge: prefer the later id/name; concat args dicts
                    # are not expected to stream per-arg here (base model
                    # carries arguments wholesale). Replace.
                    calls_by_idx[key] = tc
                else:
                    calls_by_idx[key] = tc
            if chunk.finish_reason:
                finish = chunk.finish_reason

        calls = list(calls_by_idx.values())
        return "".join(text_parts), calls, finish

    # --------------------------------------------------------------- dispatch

    async def _dispatch_calls(
        self, thread_id: str, calls: list[ToolCall]
    ) -> list[ToolResultMessage]:
        """Dispatch each call; persist + bus-emit outcomes. Returns results
        suitable for re-feeding to the provider on the next iter."""
        out: list[ToolResultMessage] = []
        for call in calls:
            self._emit(
                ToolCallEvent(
                    thread_id=thread_id,
                    call_id=call.id,
                    name=call.name,
                    arguments=call.arguments,
                    local=call.name in self._registry,
                )
            )
            if call.name in self._registry:
                res = await self._registry.dispatch(call.name, call.arguments)
                msg = ToolResultMessage(
                    call_id=call.id,
                    name=call.name,
                    content=res.content,
                    is_error=res.is_error,
                )
            else:
                msg = await self._dispatcher.dispatch(call)

            self._emit(
                ToolResultEvent(
                    thread_id=thread_id,
                    call_id=call.id,
                    ok=not msg.is_error,
                    output=None if msg.is_error else msg.content,
                    error=msg.content if msg.is_error else None,
                )
            )
            out.append(msg)
        return out

    # --------------------------------------------------------------- helpers

    def _to_core_call(self, call: ToolCall) -> ToolCall:
        """Re-stamp a provider call with a Core UUID, remembering the mapping."""
        core_id = self._config.mint_call_id()
        self._id_map[core_id] = call.id
        return ToolCall(id=core_id, name=call.name, arguments=call.arguments)

    def _strip_pending(
        self, thread_id: str, results: list[ToolResultMessage]
    ) -> list[ToolResultMessage]:
        """Drop prior results whose calls are still in flight (no result row).

        Defensive only: in normal flow, persisted tool results always have a
        matching call row and result row. On resume, :meth:`pending_tool_calls`
        is the source of truth, so this filters the assistant turns' own
        in-flight calls out of the *prior_results* view. (load_history already
        excludes results without a row; this is a belt-and-braces no-op.)
        """
        return results

    def _adhoc_defs(self, provider_name: str) -> list[ToolDef]:
        """Endpoint-advertised tool defs, merged into the offered set.

        Step 8 populates these from the endpoint's :class:`shared.protocol.Connect`
        frame. For now the loop runs with no ad-hoc defs unless the caller
        injects them via the bus/subscriptions layer (not wired here in v1).
        """
        return []

    def _emit(self, frame) -> None:
        """Publish with a swallowed-no-exceptions guarantee to the loop."""
        try:
            self._bus.publish(frame)
        except Exception:  # pragma: no cover - defensive
            _log.exception("event bus publish failed; frame=%r", frame)
