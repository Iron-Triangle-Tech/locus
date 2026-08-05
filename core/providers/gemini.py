"""Google Gemini provider adapter (``google-genai`` SDK).

Translates neutral :class:`ToolDef` to Gemini ``FunctionDeclaration`` / ``Tool``
objects, maps neutral tool results into ``functionResponse`` parts, and parses
``functionCall`` parts back to neutral :class:`ToolCall` objects.

The google-genai SDK exposes an async client at ``google.genai.Client`` with an
``aio`` property for async methods; we use ``client.aio.models.generate_content``
+ ``generate_content_stream``.
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
    from google import genai  # noqa: F401


def _tools_to_gemini(tools: list[ToolDef]) -> list[dict[str, Any]]:
    """Gemini expects a list of Tool objects each carrying function declarations.

    We build plain dicts; the google-genai SDK accepts dict shapes that match
    its proto via ``from_response``/pydantic-like coercion in ``generate_content``.
    """
    decls = [
        {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        }
        for t in tools
    ]
    return [{"function_declarations": decls}]


def _turn_to_gemini_contents(turn: AssistantTurn) -> dict[str, Any]:
    """An assistant turn -> Gemini ``model`` role contents entry."""
    parts: list[dict[str, Any]] = []
    if turn.text:
        parts.append({"text": turn.text})
    for tc in turn.tool_calls:
        parts.append({"function_call": {"name": tc.name, "args": tc.arguments}})
    return {"role": "model", "parts": parts}


def _tool_result_to_gemini(tr: ToolResultMessage) -> dict[str, Any]:
    return {
        "role": "user",
        "parts": [
            {
                "function_response": {
                    "name": tr.name,
                    "response": {"content": tr.content, "error": tr.is_error},
                }
            }
        ],
    }


def _build_contents(
    history: list[AssistantTurn],
    user: UserTurn,
    prior_tool_results: list[ToolResultMessage],
) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    # Tool results must follow the assistant turn that produced them; we pair
    # by call name (Gemini identifies function calls by name).
    for turn in history:
        contents.append(_turn_to_gemini_contents(turn))
        names = {tc.name for tc in turn.tool_calls}
        for tr in prior_tool_results:
            if tr.name in names:
                contents.append(_tool_result_to_gemini(tr))
    # Unpaired tool results (rare) -> emitted inline before the new user turn.
    paired = {tc.name for turn in history for tc in turn.tool_calls}
    for tr in prior_tool_results:
        if tr.name not in paired:
            contents.append(_tool_result_to_gemini(tr))
    if user.content:
        contents.append({"role": "user", "parts": [{"text": user.content}]})
    return contents


def _parse_response(resp: Any) -> tuple[list[ToolCall], str]:
    tool_calls: list[ToolCall] = []
    text_parts: list[str] = []
    # google-genai response objects expose .candidates[].content.parts
    for cand in getattr(resp, "candidates", []) or []:
        content = getattr(cand, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", []) or []:
            fc = getattr(part, "function_call", None)
            if fc is not None:
                args = getattr(fc, "args", None)
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args else {}
                    except json.JSONDecodeError:
                        args = {}
                elif args is None:
                    args = {}
                tool_calls.append(
                    ToolCall(
                        id=getattr(fc, "id", "") or getattr(fc, "name", ""),
                        name=getattr(fc, "name", ""),
                        arguments=dict(args) if isinstance(args, dict) else {},
                    )
                )
            elif getattr(part, "text", None):
                text_parts.append(part.text)
    return tool_calls, "".join(text_parts)


class GeminiProvider:
    """``google-genai`` Gemini implementation of :class:`Provider`."""

    def __init__(self, client: Any, model: str) -> None:
        # ``client`` is a google.genai.Client; we keep it Any-typed to avoid a
        # hard import dependency from this module's top level.
        self.client = client
        self.model = model
        self.name = "gemini"

    async def complete(
        self,
        history: list[AssistantTurn],
        user: UserTurn,
        tools: list[ToolDef],
        prior_tool_results: list[ToolResultMessage],
    ) -> ProviderResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "contents": _build_contents(history, user, prior_tool_results),
        }
        if tools:
            kwargs["tools"] = _tools_to_gemini(tools)
        resp = await self.client.aio.models.generate_content(**kwargs)
        tool_calls, text = _parse_response(resp)
        finish: Literal["stop", "tool_calls", "length", "error"] = (
            "tool_calls" if tool_calls else "stop"
        )
        return ProviderResponse(text=text, tool_calls=tool_calls, finish_reason=finish)

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
            "contents": _build_contents(history, user, prior_tool_results),
        }
        if tools:
            kwargs["tools"] = _tools_to_gemini(tools)
        had_tool_call = False
        async for chunk in await self.client.aio.models.generate_content_stream(
            **kwargs
        ):
            for cand in getattr(chunk, "candidates", []) or []:
                content = getattr(cand, "content", None)
                if content is None:
                    continue
                for part in getattr(content, "parts", []) or []:
                    fc = getattr(part, "function_call", None)
                    if fc is not None:
                        had_tool_call = True
                        args = getattr(fc, "args", None)
                        if isinstance(args, str):
                            try:
                                args = json.loads(args) if args else {}
                            except json.JSONDecodeError:
                                args = {}
                        elif args is None:
                            args = {}
                        yield ProviderStreamChunk(
                            tool_call=ToolCall(
                                id=getattr(fc, "id", "") or getattr(fc, "name", ""),
                                name=getattr(fc, "name", ""),
                                arguments=dict(args) if isinstance(args, dict) else {},
                            )
                        )
                    elif getattr(part, "text", None):
                        yield ProviderStreamChunk(token=part.text)
        yield ProviderStreamChunk(
            finish_reason="tool_calls" if had_tool_call else "stop"
        )


__all__ = ["GeminiProvider"]
