"""OpenAI-compatible provider adapter.

Same wire format as :mod:`core.providers.openai`, but pointed at a custom
``base_url`` and ``api_key`` for self-hosted / OSS gateways (vLLM, LM Studio,
LocalAI, Ollama OpenAI mode, OpenRouter...). One :class:`OpenAICompatProvider`
instance backs each named ``[providers.<name>]`` entry from core's config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .openai import OpenAIProvider

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class OpenAICompatProvider(OpenAIProvider):
    """OpenAI Chat Completions against a custom base_url.

    Reuses every helper from :class:`OpenAIProvider`; only the client
    (configured with base_url/api_key upstream) and the public ``name`` differ.
    """

    def __init__(self, client: "AsyncOpenAI", model: str, name: str) -> None:
        super().__init__(client, model)
        self.name = name


__all__ = ["OpenAICompatProvider"]
