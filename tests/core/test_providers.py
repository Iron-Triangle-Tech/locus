"""Tests for the provider adapters (:mod:`core.providers`).

The adapters take their vendor SDK client as a constructor argument, so we
inject **fake clients** that speak the same attribute/method shape and assert
the adapters correctly:

* marshal neutral ``ToolDef`` / history / tool results into each vendor's
  native request shape, and
* parse vendor response/stream objects back into neutral
  :class:`ProviderResponse` / :class:`ProviderStreamChunk` (text + tool calls
  + finish reason).

No network, no real SDK. Each fake records what the adapter sent so we can also
assert the marshalled request, not just the parsed result.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from core.providers.anthropic import AnthropicProvider as _Anth
from core.providers.base import (
    AssistantTurn,
    ToolCall,
    ToolDef,
    ToolResultMessage,
    UserTurn,
)
from core.providers.gemini import GeminiProvider
from core.providers.openai import OpenAIProvider

# --------------------------------------------------------------------------- #
# Tiny attribute-object helpers (so we can build SDK response shapes inline)
# --------------------------------------------------------------------------- #


class Obj:
    """A bag of attributes, like a stripped-down SDK response object."""

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)

    def __repr__(self) -> str:
        return f"Obj({self.__dict__})"


def _tool_def(name: str = "get_weather", *, desc: str = "get weather") -> ToolDef:
    return ToolDef(
        name=name,
        description=desc,
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #


class FakeAnthropic:
    def __init__(
        self,
        resp: Any,
        stream_events: list[Any] | None = None,
        *,
        final_stop_reason: str = "end_turn",
    ) -> None:
        self.messages = self._Messages(self, resp, stream_events, final_stop_reason)
        self.last_kwargs: dict[str, Any] = {}

    class _Messages:
        def __init__(
            self,
            outer: FakeAnthropic,
            resp: Any,
            events: list[Any] | None,
            final_stop_reason: str,
        ) -> None:
            self._outer = outer
            self._resp = resp
            self._events = events
            self._final_stop_reason = final_stop_reason

        async def create(self, **kwargs: Any) -> Any:
            self._outer.last_kwargs = kwargs
            return self._resp

        def stream(self, **kwargs: Any) -> Any:
            self._outer.last_kwargs = kwargs
            return _AnthStream(self._events, self._final_stop_reason)


class _AnthStream:
    def __init__(self, events: list[Any], final_stop_reason: str = "end_turn") -> None:
        self._events = events
        self._final_stop_reason = final_stop_reason

    async def __aenter__(self) -> _AnthStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def __aiter__(self) -> _AnthStream:
        self._it = iter(self._events)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration

    async def get_final_message(self) -> Any:
        return Obj(stop_reason=self._final_stop_reason)


class TestAnthropicAdapter:
    async def test_complete_parses_text_and_tool_use(self) -> None:
        resp = Obj(
            content=[
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "c1", "name": "get_weather", "input": {"city": "SF"}},
            ],
            stop_reason="tool_use",
        )
        client = FakeAnthropic(resp)
        prov = _Anth(client, "claude-x")
        out = await prov.complete(
            history=[],
            user=UserTurn(content="weather?"),
            tools=[_tool_def()],
            prior_tool_results=[],
        )
        assert out.text == "Let me check."
        assert out.tool_calls == [ToolCall(id="c1", name="get_weather", arguments={"city": "SF"})]
        assert out.finish_reason == "tool_calls"
        # Marshalling: tools -> anthropic input_schema shape; user appended.
        sent = client.last_kwargs
        assert sent["model"] == "claude-x"
        assert sent["tools"] == [
            {
                "name": "get_weather",
                "description": "get weather",
                "input_schema": _tool_def().parameters,
            }
        ]
        assert sent["messages"] == [{"role": "user", "content": "weather?"}]

    async def test_complete_history_and_tool_results_interleaved(self) -> None:
        resp = Obj(content=[{"type": "text", "text": "ok"}], stop_reason="end_turn")
        client = FakeAnthropic(resp)
        prov = _Anth(client, "claude-x")
        history = [
            AssistantTurn(
                text="I will use a tool",
                tool_calls=[ToolCall(id="c1", name="get_weather", arguments={"city": "SF"})],
            )
        ]
        tr = ToolResultMessage(call_id="c1", name="get_weather", content="sunny")
        await prov.complete(
            history, UserTurn(content="thanks"), [_tool_def()], prior_tool_results=[tr]
        )
        sent = client.last_kwargs["messages"]
        # Anthropic emits *all* tool results as user messages first, then the
        # assistant turn(s), then the new user turn (see _history_to_anthropic).
        assert sent[0] == {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "c1",
                    "content": "sunny",
                    "is_error": False,
                }
            ],
        }
        assert sent[1]["role"] == "assistant"
        # the assistant turn carries its text + the tool_use block
        assert sent[1]["content"] == [
            {"type": "text", "text": "I will use a tool"},
            {"type": "tool_use", "id": "c1", "name": "get_weather", "input": {"city": "SF"}},
        ]
        assert sent[-1] == {"role": "user", "content": "thanks"}

    async def test_stream_emits_tokens_then_tool_call_then_finish(self) -> None:
        events = [
            Obj(type="content_block_start", index=0, content_block=Obj(type="text")),
            Obj(type="content_block_delta", index=0, delta=Obj(type="text_delta", text="Hel")),
            Obj(type="content_block_delta", index=0, delta=Obj(type="text_delta", text="lo")),
            Obj(type="content_block_stop", index=0),
            Obj(
                type="content_block_start",
                index=1,
                content_block=Obj(type="tool_use", id="c1", name="get_weather"),
            ),
            Obj(
                type="content_block_delta",
                index=1,
                delta=Obj(type="input_json_delta", partial_json='{"city":'),
            ),
            Obj(
                type="content_block_delta",
                index=1,
                delta=Obj(type="input_json_delta", partial_json=' "SF"}'),
            ),
            Obj(type="content_block_stop", index=1),
        ]
        client = FakeAnthropic(Obj(), stream_events=events, final_stop_reason="tool_use")
        prov = _Anth(client, "claude-x")
        chunks = []
        async for c in prov.stream([], UserTurn(content="go"), [_tool_def()], []):
            chunks.append(c)
        tokens = "".join(c.token for c in chunks if c.token)
        assert tokens == "Hello"
        tcs = [c.tool_call for c in chunks if c.tool_call]
        assert tcs == [ToolCall(id="c1", name="get_weather", arguments={"city": "SF"})]
        reasons = [c.finish_reason for c in chunks if c.finish_reason]
        assert reasons == ["tool_calls"]


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #


class FakeOpenAI:
    def __init__(self, resp: Any, stream_chunks: list[Any] | None = None) -> None:
        self.chat = self._Chat(self, resp, stream_chunks)
        self.last_kwargs: dict[str, Any] = {}

    class _Chat:
        def __init__(self, outer: FakeOpenAI, resp: Any, chunks: list[Any] | None) -> None:
            self._outer = outer
            self._resp = resp
            self._chunks = chunks
            self.completions = self

        async def create(self, **kwargs: Any) -> Any:
            self._outer.last_kwargs = kwargs
            if kwargs.get("stream"):
                return _oai_stream(self._chunks)
            return self._resp


async def _oai_stream(chunks: list[Any]) -> Any:
    for c in chunks:
        yield c


class TestOpenAIAdapter:
    async def test_complete_parses_text_and_tool_calls(self) -> None:
        resp = Obj(
            choices=[
                Obj(
                    finish_reason="tool_calls",
                    message=Obj(
                        content=None,
                        tool_calls=[
                            Obj(
                                id="c1",
                                function=Obj(
                                    name="get_weather", arguments=json.dumps({"city": "SF"})
                                ),
                            )
                        ],
                    ),
                )
            ]
        )
        client = FakeOpenAI(resp)
        prov = OpenAIProvider(client, "gpt-x")
        out = await prov.complete([], UserTurn(content="weather?"), [_tool_def()], [])
        assert out.text == ""
        assert out.tool_calls == [ToolCall(id="c1", name="get_weather", arguments={"city": "SF"})]
        assert out.finish_reason == "tool_calls"
        sent = client.last_kwargs
        assert sent["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "get weather",
                    "parameters": _tool_def().parameters,
                },
            }
        ]
        assert sent["messages"] == [{"role": "user", "content": "weather?"}]

    async def test_complete_interleaves_tool_results_after_assistant_turn(self) -> None:
        resp = Obj(choices=[Obj(finish_reason="stop", message=Obj(content="ok"))])
        client = FakeOpenAI(resp)
        prov = OpenAIProvider(client, "gpt-x")
        history = [
            AssistantTurn(
                text="tool-running",
                tool_calls=[ToolCall(id="c1", name="get_weather", arguments={"city": "SF"})],
            )
        ]
        tr = ToolResultMessage(call_id="c1", name="get_weather", content="sunny")
        await prov.complete(history, UserTurn(content="thx"), [_tool_def()], [tr])
        sent = client.last_kwargs["messages"]
        # assistant turn first, then role=tool result, then final user msg.
        assert sent[0]["role"] == "assistant"
        assert sent[0]["tool_calls"][0]["id"] == "c1"
        assert sent[1] == {"role": "tool", "tool_call_id": "c1", "content": "sunny"}
        assert sent[-1] == {"role": "user", "content": "thx"}

    async def test_stream_tokens_and_tool_call_then_finish(self) -> None:
        chunks = [
            Obj(choices=[Obj(finish_reason=None, delta=Obj(content="Hi", tool_calls=None))]),
            Obj(
                choices=[
                    Obj(
                        finish_reason=None,
                        delta=Obj(
                            content=None,
                            tool_calls=[
                                Obj(
                                    index=0,
                                    id="c1",
                                    function=Obj(name="get_weather", arguments='{"city":'),
                                )
                            ],
                        ),
                    )
                ]
            ),
            Obj(
                choices=[
                    Obj(
                        finish_reason=None,
                        delta=Obj(
                            content=None,
                            tool_calls=[
                                Obj(index=0, id=None, function=Obj(name=None, arguments=' "SF"}'))
                            ],
                        ),
                    )
                ]
            ),
            Obj(
                choices=[Obj(finish_reason="tool_calls", delta=Obj(content=None, tool_calls=None))]
            ),
        ]
        client = FakeOpenAI(None, stream_chunks=chunks)
        prov = OpenAIProvider(client, "gpt-x")
        out = []
        async for c in prov.stream([], UserTurn(content="x"), [_tool_def()], []):
            out.append(c)
        tokens = "".join(c.token for c in out if c.token)
        assert tokens == "Hi"
        tcs = [c.tool_call for c in out if c.tool_call]
        assert tcs == [ToolCall(id="c1", name="get_weather", arguments={"city": "SF"})]
        reasons = [c.finish_reason for c in out if c.finish_reason]
        assert reasons == ["tool_calls"]


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #


class FakeGeminiAio:
    """Mimics ``google.genai.Client``: async surface lives under ``.aio.models``."""

    def __init__(self, resp: Any, stream_chunks: list[Any] | None = None) -> None:
        self.aio = Obj(models=self._Models(self, resp, stream_chunks))
        self.last_kwargs: dict[str, Any] = {}

    class _Models:
        def __init__(self, outer: FakeGeminiAio, resp: Any, chunks: list[Any] | None) -> None:
            self._outer = outer
            self._resp = resp
            self._chunks = chunks

        async def generate_content(self, **kwargs: Any) -> Any:
            self._outer.last_kwargs = kwargs
            return self._resp

        async def generate_content_stream(self, **kwargs: Any) -> Any:
            self._outer.last_kwargs = kwargs
            return _gemini_stream(self._chunks)


async def _gemini_stream(chunks: list[Any]) -> Any:
    for c in chunks:
        yield c


def _gemini_fc_part(name: str, args: Any) -> Any:
    return Obj(function_call=Obj(name=name, args=args, id=name))


def _gemini_text_part(text: str) -> Any:
    return Obj(text=text)


def _gemini_resp(parts: list[Any]) -> Any:
    return Obj(candidates=[Obj(content=Obj(parts=parts))])


class TestGeminiAdapter:
    async def test_complete_parses_text_and_function_call(self) -> None:
        resp = _gemini_resp(
            [
                _gemini_text_part("Checking"),
                _gemini_fc_part("get_weather", {"city": "SF"}),
            ]
        )
        client = FakeGeminiAio(resp)
        prov = GeminiProvider(client, "gemini-x")
        out = await prov.complete([], UserTurn(content="weather?"), [_tool_def()], [])
        assert out.text == "Checking"
        assert out.tool_calls == [
            ToolCall(id="get_weather", name="get_weather", arguments={"city": "SF"})
        ]
        assert out.finish_reason == "tool_calls"
        sent = client.last_kwargs
        assert sent["model"] == "gemini-x"
        assert sent["tools"] == [
            {
                "function_declarations": [
                    {
                        "name": "get_weather",
                        "description": "get weather",
                        "parameters": _tool_def().parameters,
                    }
                ]
            }
        ]
        assert sent["contents"][-1] == {"role": "user", "parts": [{"text": "weather?"}]}

    async def test_complete_history_pairs_tool_results_by_name(self) -> None:
        resp = _gemini_resp([_gemini_text_part("ok")])
        client = FakeGeminiAio(resp)
        prov = GeminiProvider(client, "gemini-x")
        history = [
            AssistantTurn(
                text="run",
                tool_calls=[ToolCall(id="c1", name="get_weather", arguments={"city": "SF"})],
            )
        ]
        tr = ToolResultMessage(call_id="c1", name="get_weather", content="sunny")
        await prov.complete(history, UserTurn(content="thx"), [_tool_def()], [tr])
        contents = client.last_kwargs["contents"]
        # model turn then function_response user turn then final user turn.
        assert contents[0]["role"] == "model"
        assert contents[1]["role"] == "user"
        fr = contents[1]["parts"][0]["function_response"]
        assert fr["name"] == "get_weather"
        assert fr["response"]["content"] == "sunny"
        assert contents[-1] == {"role": "user", "parts": [{"text": "thx"}]}

    async def test_stream_text_and_function_call_then_finish(self) -> None:
        chunks = [
            _gemini_resp([_gemini_text_part("Hel")]),
            _gemini_resp([_gemini_text_part("lo")]),
            _gemini_resp([_gemini_fc_part("get_weather", {"city": "SF"})]),
        ]
        client = FakeGeminiAio(None, stream_chunks=chunks)
        prov = GeminiProvider(client, "gemini-x")
        out = []
        async for c in prov.stream([], UserTurn(content="x"), [_tool_def()], []):
            out.append(c)
        tokens = "".join(c.token for c in out if c.token)
        assert tokens == "Hello"
        tcs = [c.tool_call for c in out if c.tool_call]
        assert tcs == [ToolCall(id="get_weather", name="get_weather", arguments={"city": "SF"})]
        reasons = [c.finish_reason for c in out if c.finish_reason]
        assert reasons == ["tool_calls"]


# --------------------------------------------------------------------------- #
# openai_compat: same logic, different name/base_url
# --------------------------------------------------------------------------- #


class TestOpenAICompatAdapter:
    async def test_name_override_and_logic_reused(self) -> None:
        from core.providers.openai_compat import OpenAICompatProvider

        resp = Obj(choices=[Obj(finish_reason="stop", message=Obj(content="hi"))])
        client = FakeOpenAI(resp)
        prov = OpenAICompatProvider(client, "llama", "my_local")
        out = await prov.complete([], UserTurn(content="hi"), [], [])
        assert prov.name == "my_local"
        assert out.text == "hi"
        assert out.finish_reason == "stop"


# --------------------------------------------------------------------------- #
# Registry: get_provider / resolve_provider_name
# --------------------------------------------------------------------------- #


class _Sentinel:
    """A stand-in provider returned by monkeypatched builders."""

    def __init__(self, tag: str = "") -> None:
        self.tag = tag


class TestRegistry:
    """The registry resolves provider names + dispatches to the builders.

    We monkeypatch the ``_build_*`` functions to sentinels so no real SDK client
    is constructed (which would read env / require installed packages).
    """

    def _settings(self, **over: Any) -> Any:
        from core.settings import CoreSettings, ProviderSettings

        prov = over.pop("provider", None)
        providers = over.pop("providers", None)
        kw: dict[str, Any] = {}
        if prov is not None:
            # Accept either a default-name string or a full ProviderSettings.
            kw["provider"] = (
                prov if isinstance(prov, ProviderSettings) else ProviderSettings(default=prov)
            )
        if providers is not None:
            kw["providers"] = providers
        return CoreSettings(**kw)

    def test_resolve_auto_uses_default(self) -> None:
        from core.providers import resolve_provider_name

        s = self._settings(provider="openai")
        assert resolve_provider_name("auto", s) == "openai"
        s2 = self._settings(provider="anthropic")
        assert resolve_provider_name("auto", s2) == "anthropic"

    def test_resolve_known_builtin_passes_through(self) -> None:
        from core.providers import resolve_provider_name

        s = self._settings()
        for name in ("anthropic", "openai", "gemini"):
            assert resolve_provider_name(name, s) == name

    def test_resolve_known_named_provider_passes(self) -> None:
        from core.providers import resolve_provider_name
        from core.settings import NamedProviderSettings

        s = self._settings(
            providers={
                "local": NamedProviderSettings(base_url="http://x", api_key="k"),
            }
        )
        assert resolve_provider_name("local", s) == "local"

    def test_resolve_unknown_raises_keyerror(self) -> None:
        from core.providers import resolve_provider_name

        s = self._settings()
        with pytest.raises(KeyError):
            resolve_provider_name("nope", s)

    def test_get_provider_dispatches_builtins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import core.providers as reg
        from core.providers import get_provider

        s = self._settings(provider="anthropic")  # model resolved from defaults
        calls: list[str] = []
        monkeypatch.setattr(reg, "_build_anthropic", lambda m: (calls.append(m), _Sentinel())[1])
        monkeypatch.setattr(reg, "_build_openai", lambda m: (calls.append(m), _Sentinel("oai"))[1])
        monkeypatch.setattr(reg, "_build_gemini", lambda m: (calls.append(m), _Sentinel("gem"))[1])
        # auto -> default anthropic
        p = get_provider("auto", s)
        assert isinstance(p, _Sentinel)
        assert calls == ["claude-3-5-sonnet-20241022"]
        calls.clear()
        get_provider("openai", s)
        assert calls == ["gpt-4o"]
        calls.clear()
        get_provider("gemini", s)
        assert calls == ["gemini-1.5-flash"]

    def test_get_provider_dispatches_named_compat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import core.providers as reg
        from core.providers import get_provider
        from core.settings import NamedProviderSettings

        s = self._settings(
            providers={
                "local": NamedProviderSettings(base_url="http://x", api_key="k", model="llama-3"),
            }
        )
        seen: dict[str, Any] = {}
        monkeypatch.setattr(
            reg,
            "_build_openai_compat",
            lambda name, base_url, api_key, model: (
                seen.update(name=name, base_url=base_url, api_key=api_key, model=model)
                or _Sentinel("compat")
            ),
        )
        p = get_provider("local", s)
        assert isinstance(p, _Sentinel)
        assert seen == {
            "name": "local",
            "base_url": "http://x",
            "api_key": "k",
            "model": "llama-3",
        }

    def test_get_provider_unknown_raises(self) -> None:
        from core.providers import get_provider

        s = self._settings()
        with pytest.raises(KeyError):
            get_provider("nope", s)

    def test_get_provider_no_model_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from core.providers import get_provider
        from core.settings import ProviderSettings

        # anthropic with an empty models map -> ValueError (no configured model)
        s = self._settings(provider=ProviderSettings(models={}))
        with pytest.raises(ValueError):
            get_provider("anthropic", s)
