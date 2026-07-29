"""Anthropic Claude provider adapter.

Translates neutral :class:`ToolDef` to the Anthropic Messages-API tool shape,
maps neutral tool results into ``tool_result`` content blocks, and parses
``tool_use`` blocks back to neutral :class:`ToolCall` objects.

Streaming uses the SDK's async stream to emit text deltas and tool_use deltas.
Tool-call argument JSON is accumulated and emitted once the tool_use block
closes (arguments arrive as incremental strings in Anthropic streams).
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
    from anthropic import AsyncAnthropic

_FINISH_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}


def _tools_to_anthropic(tools: list[ToolDef]) -> list[dict[str, Any]]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]


def _history_to_anthropic(
    history: list[AssistantTurn],
    prior_tool_results: list[ToolResultMessage],
) -> list[dict[str, Any]]:
    """Build the messages array *excluding* the imminent new user turn.

    Assistant turns carry their text and any tool_use blocks; tool results are
    added as ``user``-role messages containing tool_result blocks, in order.
    """
    messages: list[dict[str, Any]] = []
    # Interleave tool results with assistant turns. For the first cut we append
    # any pending tool results in order right before the assistant turn that
    # will consume them; the loop keeps pairing simple.
    for tr in prior_tool_results:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tr.call_id,
                        "content": tr.content,
                        "is_error": tr.is_error,
                    }
                ],
            }
        )
    for turn in history:
        content: list[dict[str, Any]] = []
        if turn.text:
            content.append({"type": "text", "text": turn.text})
        for tc in turn.tool_calls:
            content.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                }
            )
        messages.append({"role": "assistant", "content": content})
    return messages


def _parse_tool_calls(content: list[dict[str, Any]]) -> tuple[list[ToolCall], str]:
    tool_calls: list[ToolCall] = []
    text_parts: list[str] = []
    for block in content:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=block.get("input") or {},
                )
            )
    return tool_calls, "".join(text_parts)


class AnthropicProvider:
    """``anthropic`` Claude Messages-API implementation of :class:`Provider`."""

    def __init__(self, client: "AsyncAnthropic", model: str) -> None:
        self.client = client
        self.model = model
        self.name = "anthropic"

    async def complete(
        self,
        history: list[AssistantTurn],
        user: UserTurn,
        tools: list[ToolDef],
        prior_tool_results: list[ToolResultMessage],
    ) -> ProviderResponse:
        messages = _history_to_anthropic(history, prior_tool_results)
        messages.append({"role": "user", "content": user.content})
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = _tools_to_anthropic(tools)
        resp = await self.client.messages.create(**kwargs)
        tool_calls, text = _parse_tool_calls(list(resp.content))
        return ProviderResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason=_FINISH_MAP.get(
                getattr(resp, "stop_reason", "end_turn"), "stop"
            ),
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
        messages = _history_to_anthropic(history, prior_tool_results)
        messages.append({"role": "user", "content": user.content})
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = _tools_to_anthropic(tools)

        # Accumulate tool_use input deltas; emit a ToolCall on input_json_delta
        # completion by buffering the JSON string and parsing at end.
        pending: dict[str, dict[str, Any]] = {}

        async with self.client.messages.stream(**kwargs) as stream:
            async for event in stream:
                etype = getattr(event, "type", "")
                if etype == "content_block_delta":
                    delta = event.delta  # type: ignore[attr-defined]
                    dtype = getattr(delta, "type", "")
                    if dtype == "text_delta":
                        yield ProviderStreamChunk(token=getattr(delta, "text", ""))
                    elif dtype == "input_json_delta":
                        # Buffer partial JSON keyed by the current tool_use id.
                        # Find the active tool_use block index from content_block.
                        bindex = getattr(event, "index", 0)
                        slot = pending.setdefault(
                            str(bindex), {"id": "", "name": "", "json": ""}
                        )
                        slot["json"] += getattr(delta, "partial_json", "")
                elif etype == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if block is not None and getattr(block, "type", "") == "tool_use":
                        bindex = getattr(event, "index", 0)
                        pending[str(bindex)] = {
                            "id": getattr(block, "id", ""),
                            "name": getattr(block, "name", ""),
                            "json": "",
                        }
                elif etype == "content_block_stop":
                    bindex = str(getattr(event, "index", -1))
                    if bindex in pending:
                        slot = pending.pop(bindex)
                        try:
                            args = json.loads(slot["json"]) if slot["json"] else {}
                        except json.JSONDecodeError:
                            args = {}
                        yield ProviderStreamChunk(
                            tool_call=ToolCall(
                                id=slot["id"], name=slot["name"], arguments=args
                            )
                        )
            final = await stream.get_final_message()
            reason: Literal["stop", "tool_calls", "length", "error"] = _FINISH_MAP.get(
                getattr(final, "stop_reason", "end_turn"), "stop"
            )
            yield ProviderStreamChunk(finish_reason=reason)


__all__ = ["AnthropicProvider"]
