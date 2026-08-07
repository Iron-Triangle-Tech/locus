"""Agent loop tests with a fake streaming provider.

Exercises the loop end-to-end against an in-memory store + bus + registry,
using a controllable fake :class:`Provider` whose ``stream`` emits canned
chunks. Covers:

* one-shot (no tools) turn -> FinalEvent + persisted assistant text
* tool-call turn where the tool is BUILT-IN -> local dispatch + result event
* tool-call turn where the tool is AD-HOC -> NoAdhocDispatcher returns is_error
* multi-iteration turn (tool call then a real answer) -> two provider calls
* provider resolver error -> ErrorEvent(fatal=True), no crash
* bus receives the expected event sequence per turn
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from core.agent.loop import AgentLoop, LoopConfig, NoAdhocDispatcher
from core.bus import EventBus
from core.providers.base import ProviderStreamChunk, ToolCall
from core.settings import CoreSettings, StorageSettings, ToolsSettings
from core.storage import session as session_mod
from core.storage.database_io import Base, MemoryStore
from core.tools.file import default_file_tools
from core.tools.loader import load_tool_defs, seed_missing
from core.tools.registry import ToolRegistry


class FakeProvider:
    """A controllable streaming provider.

    ``scripts`` is a list of "iterations" (outer list). Each iteration is the
    list of chunks to yield before the final ``finish_reason`` chunk. The loop
    makes one ``stream()`` call per iteration, so each script entry is consumed
    in order.
    """

    name = "fake"

    def __init__(self, model: str, scripts: list[list[ProviderStreamChunk]]) -> None:
        self.model = model
        self._scripts = list(scripts)
        self.calls = 0

    async def complete(self, history, user, tools, prior_tool_results):
        raise NotImplementedError

    def stream(
        self, history, user, tools, prior_tool_results
    ) -> AsyncIterator[ProviderStreamChunk]:
        self.calls += 1
        return self._iter(self._scripts.pop(0))

    async def _iter(self, chunks) -> AsyncIterator[ProviderStreamChunk]:
        for c in chunks:
            yield c


@pytest.fixture
async def env(tmp_path: Path):
    settings = CoreSettings(
        storage=StorageSettings(sqlite_path=":memory:"),
        tools=ToolsSettings(agent_root=str(tmp_path / "ws")),
    )
    engine = session_mod.create_engine(settings)
    await session_mod.init_db(engine, Base.metadata)
    sf = session_mod.make_session_factory(engine)
    store = MemoryStore(sf)
    # Seed ROM defs so the registry has defs for the built-in tools.
    await seed_missing(store, load_tool_defs())
    reg = ToolRegistry()
    reg.register_compiled_tools(*default_file_tools(tmp_path / "ws"))
    reg.load_defs({d.name: d for d in await store.list_tool_defs()})
    bus = EventBus()
    yield store, reg, bus, settings
    await engine.dispose()
    bus.close()


async def _run_and_collect(
    loop: AgentLoop, thread_id: str, content: str, *, provider_name: str = "auto"
) -> list:
    """Run a turn and collect ALL frames the loop published to the bus."""
    frames: list = []
    orig = loop._bus.publish  # type: ignore[attr-defined]

    def spy(frame) -> None:
        frames.append(frame)
        orig(frame)

    loop._bus.publish = spy  # type: ignore[attr-defined, method-assign]
    await loop.run_agent_turn(thread_id, content, provider_name=provider_name)
    loop._bus.publish = orig  # type: ignore[attr-defined, method-assign]
    return frames


class TestOneShotTurn:
    async def test_final_event_emitted(self, env) -> None:
        store, reg, bus, settings = env
        thread = await store.create_thread(provider="fake", model="fake-1")
        provider = FakeProvider(
            "fake-1",
            scripts=[
                [
                    ProviderStreamChunk(token="hel"),
                    ProviderStreamChunk(token="lo"),
                    ProviderStreamChunk(finish_reason="stop"),
                ]
            ],
        )
        # Bypass get_provider by injecting the provider via a small shim.
        loop = AgentLoop(
            settings=settings,
            store=store,
            registry=reg,
            bus=bus,
            config=LoopConfig(max_iters=4),
        )

        # Monkeypatch get_provider to return our fake (avoids settings dance).
        import core.agent.loop as loop_mod

        orig_gp = loop_mod.get_provider
        loop_mod.get_provider = lambda name, settings: provider
        try:
            frames = await _run_and_collect(loop, thread.id, "hi")
        finally:
            loop_mod.get_provider = orig_gp

        kinds = [f.type for f in frames]
        assert "token" in kinds
        assert "final" in kinds
        final = next(f for f in frames if f.type == "final")
        assert final.text == "hello"

        # Persisted.
        turns, results = await store.load_history(thread.id)
        assert any(t.text == "hello" for t in turns)
        assert results == []


class TestBuiltInToolTurn:
    async def test_local_tool_dispatched_and_result(self, env, tmp_path: Path) -> None:
        store, reg, bus, settings = env
        thread = await store.create_thread(provider="fake", model="fake-1")
        # Write a file via the loop: model calls file_write, then finalizes.
        provider = FakeProvider(
            "fake-1",
            scripts=[
                [
                    ProviderStreamChunk(token="writing"),
                    ProviderStreamChunk(
                        tool_call=ToolCall(
                            id="p1",
                            name="file_write",
                            arguments={"path": "a.txt", "content": "data"},
                        )
                    ),
                    ProviderStreamChunk(finish_reason="tool_calls"),
                ],
                [
                    ProviderStreamChunk(token="done"),
                    ProviderStreamChunk(finish_reason="stop"),
                ],
            ],
        )
        loop = AgentLoop(
            settings=settings, store=store, registry=reg, bus=bus, config=LoopConfig(max_iters=4)
        )
        import core.agent.loop as loop_mod

        orig = loop_mod.get_provider
        loop_mod.get_provider = lambda name, settings: provider
        try:
            frames = await _run_and_collect(loop, thread.id, "go")
        finally:
            loop_mod.get_provider = orig

        kinds = [f.type for f in frames]
        assert "tool_call" in kinds
        assert "tool_result_event" in kinds
        assert "final" in kinds
        tc = next(f for f in frames if f.type == "tool_call")
        assert tc.local is True
        assert tc.name == "file_write"
        tr = next(f for f in frames if f.type == "tool_result_event")
        assert tr.ok is True
        assert (tmp_path / "ws" / "a.txt").read_text() == "data"
        # Persisted assistant turn has the (core-uuid-id) call + a result row.
        turns, results = await store.load_history(thread.id)
        assert any(t.tool_calls for t in turns)
        assert len(results) == 1


class TestAdhocToolTurn:
    async def test_adhoc_without_dispatcher_errors(self, env) -> None:
        store, reg, bus, settings = env
        thread = await store.create_thread(provider="fake", model="fake-1")
        provider = FakeProvider(
            "fake-1",
            scripts=[
                [
                    ProviderStreamChunk(
                        tool_call=ToolCall(id="p1", name="endpoint_thing", arguments={"x": 1})
                    ),
                    ProviderStreamChunk(finish_reason="tool_calls"),
                ],
                [
                    ProviderStreamChunk(token="ok"),
                    ProviderStreamChunk(finish_reason="stop"),
                ],
            ],
        )
        loop = AgentLoop(
            settings=settings,
            store=store,
            registry=reg,
            bus=bus,
            dispatcher=NoAdhocDispatcher(),
            config=LoopConfig(max_iters=4),
        )
        import core.agent.loop as loop_mod

        orig = loop_mod.get_provider
        loop_mod.get_provider = lambda name, settings: provider
        try:
            frames = await _run_and_collect(loop, thread.id, "do it")
        finally:
            loop_mod.get_provider = orig

        tc = next(f for f in frames if f.type == "tool_call")
        assert tc.local is False
        assert tc.name == "endpoint_thing"
        tr = next(f for f in frames if f.type == "tool_result_event")
        assert tr.ok is False
        assert tr.error and "no ad-hoc dispatcher" in tr.error
        # Kept going to a final.
        assert any(f.type == "final" for f in frames)


class TestProviderError:
    async def test_unknown_provider_emits_fatal_error(self, env) -> None:
        store, reg, bus, settings = env
        thread = await store.create_thread()
        loop = AgentLoop(settings=settings, store=store, registry=reg, bus=bus)
        # Genuinely unknown provider => get_provider raises KeyError => fatal.
        frames = await _run_and_collect(loop, thread.id, "hi", provider_name="nope")
        err = next(f for f in frames if f.type == "error")
        assert err.fatal is True
        assert "provider error" in err.message


class TestMaxIters:
    async def test_exhausted_max_iters_emits_error(self, env) -> None:
        store, reg, bus, settings = env
        thread = await store.create_thread(provider="fake", model="fake-1")
        # Each iteration the model emits a tool call -> never terminates.
        provider = FakeProvider(
            "fake-1",
            scripts=[
                [
                    ProviderStreamChunk(
                        tool_call=ToolCall(id=f"p{i}", name="file_list", arguments={})
                    ),
                    ProviderStreamChunk(finish_reason="tool_calls"),
                ]
                for i in range(10)
            ],
        )
        loop = AgentLoop(
            settings=settings, store=store, registry=reg, bus=bus, config=LoopConfig(max_iters=2)
        )
        import core.agent.loop as loop_mod

        orig = loop_mod.get_provider
        loop_mod.get_provider = lambda name, settings: provider
        try:
            frames = await _run_and_collect(loop, thread.id, "loop")
        finally:
            loop_mod.get_provider = orig

        errs = [f for f in frames if f.type == "error"]
        assert errs and "max_iters" in errs[-1].message
        assert provider.calls == 2
