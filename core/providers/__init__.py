"""Provider registry.

``get_provider`` resolves a provider name (opaque string) to a concrete
:class:`Provider` instance, lazily constructing the vendor SDK client. Built-in
names (``anthropic``/``openai``/``gemini``) use the official SDKs + their
``*_API_KEY`` env vars; any name configured under ``[providers.<name>]`` uses
the OpenAI-compatible adapter against the configured ``base_url``/``api_key``.

Vendor SDKs are imported lazily so a service without, say, ``anthropic``
installed still runs as long as it only requests providers it has.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import (
    AssistantTurn,
    Provider,
    ProviderResponse,
    ProviderStreamChunk,
    ToolCall,
    ToolDef,
    ToolResultMessage,
    UserTurn,
)

if TYPE_CHECKING:
    from core.settings import CoreSettings

__all__ = [
    "AssistantTurn",
    "Provider",
    "ProviderResponse",
    "ProviderStreamChunk",
    "ToolCall",
    "ToolDef",
    "ToolResultMessage",
    "UserTurn",
    "get_provider",
    "resolve_provider_name",
]


def resolve_provider_name(name: str, settings: CoreSettings) -> str:
    """Map a wire-protocol ``provider`` value to a concrete provider name.

    ``"auto"`` resolves to the configured default; any other name is returned
    as-is. Raises ``KeyError`` if the name is neither built-in nor configured.
    """
    if name == "auto":
        name = settings.provider.default
    if not settings.is_known_provider(name):
        raise KeyError(f"Unknown provider: {name!r}")
    return name


def get_provider(name: str, settings: CoreSettings) -> Provider:
    """Construct a :class:`Provider` for ``name`` (``"auto"`` resolves to default)."""
    raw_name = name
    if raw_name == "auto":
        raw_name = settings.provider.default
    if not settings.is_known_provider(raw_name):
        raise KeyError(f"Unknown provider: {name!r}")

    model = settings.resolved_model(raw_name)
    if not model:
        raise ValueError(f"No model configured for provider {raw_name!r}")

    if raw_name in settings.providers:
        # Named OpenAI-compatible gateway.
        entry = settings.providers[raw_name]
        return _build_openai_compat(raw_name, entry.base_url, entry.api_key, model)

    if raw_name == "anthropic":
        return _build_anthropic(model)
    if raw_name == "openai":
        return _build_openai(model)
    if raw_name == "gemini":
        return _build_gemini(model)

    raise KeyError(f"Unknown provider: {name!r}")


def _build_anthropic(model: str) -> Provider:
    import os

    from anthropic import AsyncAnthropic

    from .anthropic import AnthropicProvider

    api_key = os.environ.get("ANTHROPIC_API_KEY") or None
    client = AsyncAnthropic(api_key=api_key)
    return AnthropicProvider(client, model)


def _build_openai(model: str) -> Provider:
    from openai import AsyncOpenAI

    from .openai import OpenAIProvider

    client = AsyncOpenAI()
    return OpenAIProvider(client, model)


def _build_openai_compat(name: str, base_url: str, api_key: str, model: str) -> Provider:
    from openai import AsyncOpenAI

    from .openai_compat import OpenAICompatProvider

    client = AsyncOpenAI(base_url=base_url, api_key=api_key or "ignored")
    return OpenAICompatProvider(client, model, name)


def _build_gemini(model: str) -> Provider:
    import os

    from google import genai

    from .gemini import GeminiProvider

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or None)
    return GeminiProvider(client, model)
