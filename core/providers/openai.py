"""OpenAI Chat Completions provider adapter.

This is the canonical OpenAI implementation. The :class:`OpenAICompatProvider`
adapter reuses the same message/tool marshalling with a different ``base_url``
+ api_key, so all shared logic lives in module-level helpers here.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Literal

from .base import (
    AssistantTurn,
    ProviderResponse,
    ProviderStreamChunk,
    ToolCall,
    ToolDef,
    ToolResultMessage,
    UserTurn,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI

_FINISH_MAP = {
    "stop": "stop",
    "tool_calls": "tool_calls",
    "length": "length",
    "content_filter": "stop",
}


def _tools_to_openai(tools: list[ToolDef]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def _assistant_to_openai(turn: AssistantTurn) -> dict[str, Any]:
    """An assistant turn -> a chat message with optional tool_calls blob."""
    msg: dict[str, Any] = {"role": "assistant", "content": turn.text or None}
    if turn.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in turn.tool_calls
        ]
    return msg


def _build_messages(
    history: list[AssistantTurn],
    user: UserTurn,
    prior_tool_results: list[ToolResultMessage],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    # Tool results come back as role=tool messages, in pairing order with the
    # assistant turn that issued the calls. For the first cut we emit tool
    # result messages immediately after the assistant turn that produced them;
    # since history is the record of assistant turns, we interleave by emitting
    # assistant turns then any tool results referencing their call ids.
    for turn in history:
        messages.append(_assistant_to_openai(turn))
        # Emit tool results that were produced for calls in THIS turn right
        # after it -- identified by matching call ids.
        ids = {tc.id for tc in turn.tool_calls}
        for tr in prior_tool_results:
            if tr.call_id in ids:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.call_id,
                        "content": tr.content,
                    }
                )
    # If caller supplied tool results not paired with any history turn (rare,
    # e.g. first call where the assistant turn hasn't been persisted yet),
    # emit them in order before the new user turn.
    paired = {tc.id for turn in history for tc in turn.tool_calls}
    for tr in prior_tool_results:
        if tr.call_id not in paired:
            messages.append(
                {"role": "tool", "tool_call_id": tr.call_id, "content": tr.content}
            )
    messages.append({"role": "user", "content": user.content})
    return messages


def _parse_tool_calls(calls: list[Any] | None) -> list[ToolCall]:
    out: list[ToolCall] = []
    if not calls:
        return out
    for tc in calls:
        fn = tc.function
        try:
            args = json.loads(fn.arguments) if fn.arguments else {}
        except (json.JSONDecodeError, TypeError):
            args = {}
        out.append(ToolCall(id=tc.id, name=fn.name, arguments=args))
    return out


class OpenAIProvider:
    """OpenAI Chat Completions implementation of :class:`Provider`."""

    def __init__(self, client: "AsyncOpenAI", model: str) -> None:
        self.client = client
        self.model = model
        self.name = "openai"

    async def complete(
        self,
        history: list[AssistantTurn],
        user: UserTurn,
        tools: list[ToolDef],
        prior_tool_results: list[ToolResultMessage],
    ) -> ProviderResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _build_messages(history, user, prior_tool_results),
        }
        if tools:
            kwargs["tools"] = _tools_to_openai(tools)
        resp = await self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        return ProviderResponse(
            text=msg.content or "",
            tool_calls=_parse_tool_calls(getattr(msg, "tool_calls", None)),
            finish_reason=_FINISH_MAP.get(choice.finish_reason or "stop", "stop"),
        )

    def stream(
        self,
        history: list[AssistantTurn],
        user: UserTurn,
        tools: list[ToolDef],
        prior_tool_results: list[ToolResultMessage],
    ) -> AsyncIterator[ProviderStreamChunk]:
        return self._stream(history, user, tools, prior_tool_results)

    async def _stream(
        self,
        history: list[AssistantTurn],
        user: UserTurn,
        tools: list[ToolDef],
        prior_tool_results: list[ToolResultMessage],
    ) -> AsyncIterator[ProviderStreamChunk]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _build_messages(history, user, prior_tool_results),
            "stream": True,
        }
        if tools:
            kwargs["tools"] = _tools_to_openai(tools)

        # Accumulate tool-call argument deltas per index, emit on completion.
        pending: dict[int, dict[str, Any]] = {}

        stream = await self.client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta.content:
                yield ProviderStreamChunk(token=delta.content)
            tcalls = getattr(delta, "tool_calls", None)
            if tcalls:
                for tc in tcalls:
                    idx = tc.index
                    slot = pending.setdefault(
                        idx, {"id": "", "name": "", "json": ""}
                    )
                    if tc.id:
                        slot["id"] = tc.id
                    fn = tc.function
                    if fn and fn.name:
                        slot["name"] = fn.name
                    if fn and fn.arguments:
                        slot["json"] += fn.arguments
            if choice.finish_reason:
                # Flush any pending tool calls.
                for slot in pending.values():
                    try:
                        args = json.loads(slot["json"]) if slot["json"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    yield ProviderStreamChunk(
                        tool_call=ToolCall(
                            id=slot["id"], name=slot["name"], arguments=args
                        )
                    )
                pending.clear()
                yield ProviderStreamChunk(
                    finish_reason=_FINISH_MAP.get(choice.finish_reason, "stop")
                )


__all__ = ["OpenAIProvider"]
