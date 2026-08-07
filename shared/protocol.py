"""Wire protocol for the core <-> endpoint link.

Single JSON objects with discriminated ``type`` field over WebSocket.
Tool-call/result shapes are provider-agnostic OpenAI-style JSON-schema.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

# ---------------------------------------------------------------------------
# Common, reused types
# ---------------------------------------------------------------------------

#: Role on a stored message thread.
Role = Literal["user", "assistant", "tool", "system"]

#: Provider identifier used to route a turn. Opaque string name resolved by
#: core's provider registry; the reserved value ``"auto"`` picks the configured
#: default. Built-ins include ``anthropic``/``openai``/``gemini``; any name
#: configured under ``[providers.<name>]`` in core's config also works.
ProviderName = str


class ToolSchema(BaseModel):
    """OpenAI-style function-calling tool definition.

    This is the neutral, internal canonical form. Provider adapters translate
    to/from each vendor's native tool shape at the edge.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict  # JSON Schema describing the arguments object


class AdhocTool(BaseModel):
    """An endpoint-declared tool, advertised at connect time.

    The endpoint tells core which tools it can run; core passes them to the
    provider alongside its own built-in tools. When the model invokes one,
    core forwards the call over the link and waits for a ToolResult.
    """

    model_config = ConfigDict(extra="forbid")

    tool: ToolSchema


# ---------------------------------------------------------------------------
# Endpoint -> core frames
# ---------------------------------------------------------------------------


class Connect(BaseModel):
    """First frame from endpoint after the WebSocket opens.

    Carries the ad-hoc tools the endpoint can execute (may be empty) so core
    can merge them into the toolset it offers the provider.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["connect"] = "connect"
    endpoint_id: str
    adhoc_tools: list[AdhocTool] = Field(default_factory=list)


class UserMessage(BaseModel):
    """A user turn to run through the agent loop.

    ``thread_id`` is optional so the endpoint can start a brand-new thread by
    omitting it; core allocates an id and echoes it back on subsequent events.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["user_message"] = "user_message"
    thread_id: str | None = None
    content: str
    provider: ProviderName = "auto"


class ToolResult(BaseModel):
    """Result of an endpoint-executed ad-hoc tool, returned to core."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_result"] = "tool_result"
    call_id: str
    ok: bool
    output: str | None = None
    error: str | None = None


class Disconnect(BaseModel):
    """Endpoint signalling it is leaving cleanly."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["disconnect"] = "disconnect"
    reason: str | None = None


# ---------------------------------------------------------------------------
# Core -> endpoint frames (agent events)
# ---------------------------------------------------------------------------


class TokenEvent(BaseModel):
    """A streaming assistant text chunk."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["token"] = "token"
    thread_id: str
    delta: str


class ToolCallEvent(BaseModel):
    """The agent is invoking a tool.

    ``local`` is True when core is executing a built-in tool itself; False when
    the endpoint is expected to execute it and return a ToolResult. Endpoint
    rendering treats both the same (show the user what the agent is doing).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_call"] = "tool_call"
    thread_id: str
    call_id: str
    name: str
    arguments: dict
    local: bool = False


class ToolResultEvent(BaseModel):
    """Outcome of a tool call, streamed to the endpoint for display."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_result_event"] = "tool_result_event"
    thread_id: str
    call_id: str
    ok: bool
    output: str | None = None
    error: str | None = None


class FinalEvent(BaseModel):
    """Terminal assistant turn complete (no more tokens coming)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["final"] = "final"
    thread_id: str
    text: str


class ErrorEvent(BaseModel):
    """Something went wrong during the turn."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["error"] = "error"
    thread_id: str | None = None
    message: str
    fatal: bool = False


# ---------------------------------------------------------------------------
# Discriminated unions + helpers
# ---------------------------------------------------------------------------

EndpointFrame = Annotated[
    Connect | UserMessage | ToolResult | Disconnect,
    Field(discriminator="type"),
]
CoreFrame = Annotated[
    TokenEvent | ToolCallEvent | ToolResultEvent | FinalEvent | ErrorEvent,
    Field(discriminator="type"),
]

# Direction sets, useful for cheap server-side dispatch before full parsing.
ENDPOINT_TAGS: frozenset[str] = frozenset({"connect", "user_message", "tool_result", "disconnect"})
CORE_TAGS: frozenset[str] = frozenset({"token", "tool_call", "tool_result_event", "final", "error"})

_endpoint_adapter: TypeAdapter[EndpointFrame] = TypeAdapter(EndpointFrame)
_core_adapter: TypeAdapter[CoreFrame] = TypeAdapter(CoreFrame)


def dump(frame: BaseModel) -> str:
    """Serialize a frame to a single JSON string for the wire."""
    return frame.model_dump_json()


def load_endpoint(text: str) -> Connect | UserMessage | ToolResult | Disconnect:
    """Parse a frame known to be endpoint->core."""
    return _endpoint_adapter.validate_json(text)  # type: ignore[return-value]


def load_core(text: str) -> TokenEvent | ToolCallEvent | ToolResultEvent | FinalEvent | ErrorEvent:
    """Parse a frame known to be core->endpoint."""
    return _core_adapter.validate_json(text)  # type: ignore[return-value]


__all__ = [
    "CORE_TAGS",
    "ENDPOINT_TAGS",
    "AdhocTool",
    "Connect",
    "CoreFrame",
    "EndpointFrame",
    "ErrorEvent",
    "FinalEvent",
    "ProviderName",
    "Role",
    "TokenEvent",
    "ToolCallEvent",
    "ToolResult",
    "ToolResultEvent",
    "ToolSchema",
    "UserMessage",
    "dump",
    "load_core",
    "load_endpoint",
]
