"""Neutral Provider protocol + internal message/tool types.

Agent loop talks only to :class:`Provider`. Adapters translate to/from vendor
native shapes. Canonical tool form is OpenAI-style JSON-schema (``ToolDef``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Internal canonical types
# ---------------------------------------------------------------------------


class UserTurn(BaseModel):
    """A single user message (content already in canonical form)."""

    model_config = ConfigDict(extra="forbid")

    content: str


class AssistantTurn(BaseModel):
    """A stored assistant turn (text + any tool calls it made)."""

    model_config = ConfigDict(extra="forbid")

    text: str = ""
    tool_calls: list[ToolCall] = []


class ToolCall(BaseModel):
    """A tool call requested by the model (provider-agnostic)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict


class ToolResultMessage(BaseModel):
    """The result of a tool call, fed back to the model as a tool message."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    name: str
    content: str
    is_error: bool = False


class ToolDef(BaseModel):
    """OpenAI-style function-calling tool definition (neutral canonical form).

    Adapters translate this to their vendor's native tool shape.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict  # JSON Schema describing the arguments object


# A chat history entry expressed in the neutral form. Order matters; the first
# item is the oldest.
History = list  # type alias used in type hints


class ProviderResponse(BaseModel):
    """The non-streaming result of a provider call."""

    model_config = ConfigDict(extra="forbid")

    text: str = ""
    tool_calls: list[ToolCall] = []
    finish_reason: Literal["stop", "tool_calls", "length", "error"] = "stop"


class ProviderStreamChunk(BaseModel):
    """One chunk emitted while streaming a provider call.

    Exactly one of ``token`` / ``tool_call`` is set per chunk (or neither, for
    a final marker). ``done`` True means the stream has ended.
    """

    model_config = ConfigDict(extra="forbid")

    token: str | None = None
    tool_call: ToolCall | None = None
    finish_reason: Literal["stop", "tool_calls", "length", "error"] | None = None


@runtime_checkable
class Provider(Protocol):
    """The neutral interface every adapter implements.

    A ``Provider`` is keyed by (name, model). It converts a neutral history +
    new user turn plus an optional toolset into model output -- either as a
    single accumulated response or a stream of chunks.
    """

    name: str
    model: str

    async def complete(
        self,
        history: list[AssistantTurn],
        user: UserTurn,
        tools: list[ToolDef],
        prior_tool_results: list[ToolResultMessage],
    ) -> ProviderResponse: ...

    def stream(
        self,
        history: list[AssistantTurn],
        user: UserTurn,
        tools: list[ToolDef],
        prior_tool_results: list[ToolResultMessage],
    ) -> AsyncIterator[ProviderStreamChunk]: ...


__all__ = [
    "AssistantTurn",
    "History",
    "Provider",
    "ProviderResponse",
    "ProviderStreamChunk",
    "ToolCall",
    "ToolDef",
    "ToolResultMessage",
    "UserTurn",
]
