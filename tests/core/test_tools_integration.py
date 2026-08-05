"""Integration test for Step 6: ROM -> store -> registry -> dispatch.

Covers the whole tool layer chain end-to-end against an in-memory store:

1. ``load_tool_defs`` parses ``core/tools.toml`` into ``ToolDef`` dict.
2. ``seed_missing`` inserts missing rows (INSERT-only; re-seed is a no-op).
3. ``MemoryStore.list_tool_defs`` reads them back as ``ToolDef``.
4. ``ToolRegistry`` joins runnables (file/http tools) with defs and
   ``export_defs`` returns the intersection (sorted, all 4 here).
5. ``dispatch`` runs built-in tools (file_read/file_write/file_list) and
   truncation + sandbox escape is exercised.

No network is hit (``http_fetch`` is dispatched against a non-routable URL to
confirm its error path only); HTTP behavior is left to dedicated tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.providers.base import ToolDef
from core.settings import CoreSettings, StorageSettings, ToolsSettings
from core.storage import session as session_mod
from core.storage.database_io import Base, MemoryStore
from core.tools.file import default_file_tools
from core.tools.http import HttpFetch
from core.tools.loader import load_tool_defs, seed_missing
from core.tools.registry import ToolRegistry


@pytest.fixture
async def store(tmp_path: Path) -> MemoryStore:
    """An in-memory MemoryStore with tables created."""
    settings = CoreSettings(
        storage=StorageSettings(sqlite_path=":memory:"),
        tools=ToolsSettings(agent_root=str(tmp_path / "ws")),
    )
    engine = session_mod.create_engine(settings)
    await session_mod.init_db(engine, Base.metadata)
    sf = session_mod.make_session_factory(engine)
    yield MemoryStore(sf)
    await engine.dispose()


@pytest.fixture
async def registry(store: MemoryStore, tmp_path: Path) -> ToolRegistry:
    """A ToolRegistry whose defs come FROM THE STORE (after ROM seeding).

    This is the real startup path: ROM -> seed_missing -> list_tool_defs ->
    registry.load_defs, joined with the compiled runnables.
    """
    await seed_missing(store, load_tool_defs())
    reg = ToolRegistry()
    agent_root = tmp_path / "ws"
    reg.register_compiled_tools(*default_file_tools(agent_root))
    reg.register_compiled_tools(
        HttpFetch(max_bytes=1024, default_timeout=2.0)
    )
    defs = {d.name: d for d in await store.list_tool_defs()}
    reg.load_defs(defs)
    return reg


class TestRomLoad:
    def test_parses_all_four_tools(self) -> None:
        defs = load_tool_defs()
        assert set(defs) == {"file_read", "file_write", "file_list", "http_fetch"}

    def test_defs_have_schema_and_description(self) -> None:
        defs = load_tool_defs()
        for name, d in defs.items():
            assert isinstance(d, ToolDef)
            assert d.name == name
            assert d.description, f"{name} missing description"
            assert d.parameters.get("type") == "object", f"{name} not object schema"

    def test_file_read_required_path(self) -> None:
        defs = load_tool_defs()
        params = defs["file_read"].parameters
        assert "path" in params.get("required", [])
        assert "path" in params["properties"]

    def test_http_fetch_required_url(self) -> None:
        defs = load_tool_defs()
        params = defs["http_fetch"].parameters
        assert "url" in params.get("required", [])


class TestSeedAndStore:
    async def test_seed_inserts_missing(self, store: MemoryStore) -> None:
        defs = load_tool_defs()
        inserted = await seed_missing(store, defs)
        assert sorted(inserted) == ["file_list", "file_read", "file_write", "http_fetch"]

        rows = await store.list_tool_defs()
        assert {r.name for r in rows} == set(defs)

    async def test_seed_is_insert_only(self, store: MemoryStore) -> None:
        defs = load_tool_defs()
        await seed_missing(store, defs)

        # Mutate a def's description in the store directly.
        edited = ToolDef(
            name="file_read",
            description="EDITED BY USER",
            parameters={"type": "object"},
        )
        await store.upsert_tool_def(edited)

        # Re-seed: existing rows untouched.
        inserted = await seed_missing(store, defs)
        assert inserted == []

        rows = {r.name: r for r in await store.list_tool_defs()}
        assert rows["file_read"].description == "EDITED BY USER"

    async def test_list_returns_tooldef_objects(self, store: MemoryStore) -> None:
        await seed_missing(store, load_tool_defs())
        rows = await store.list_tool_defs()
        assert all(isinstance(r, ToolDef) for r in rows)
        # Ordered by name.
        assert [r.name for r in rows] == sorted(r.name for r in rows)


class TestRegistryIntersection:
    def test_export_defs_returns_all_four(self, registry: ToolRegistry) -> None:
        exported = registry.export_defs()
        assert [d.name for d in exported] == [
            "file_list", "file_read", "file_write", "http_fetch"
        ]
        assert all(isinstance(d, ToolDef) for d in exported)

    async def test_def_without_runnable_is_hidden(
        self, store: MemoryStore, tmp_path: Path
    ) -> None:
        # Seed the ROM first so the built-in tools have defs in the store.
        await seed_missing(store, load_tool_defs())
        # Add a def row for a tool that has no runnable.
        await store.upsert_tool_def(
            ToolDef(name="ghost", description="no impl", parameters={"type": "object"})
        )
        reg = ToolRegistry()
        reg.register_compiled_tools(*default_file_tools(tmp_path / "ws"))
        reg.register_compiled_tools(HttpFetch(max_bytes=1024, default_timeout=2.0))

        defs = {d.name: d for d in await store.list_tool_defs()}
        reg.load_defs(defs)

        exported = {d.name for d in reg.export_defs()}
        assert "ghost" not in exported
        assert "file_read" in exported

    def test_runnable_without_def_is_hidden(self, tmp_path: Path) -> None:
        reg = ToolRegistry()
        reg.register_compiled_tools(*default_file_tools(tmp_path / "ws"))
        # No defs loaded at all -> intersection is empty.
        reg.load_defs({})
        assert reg.export_defs() == []


class TestDispatch:
    async def test_file_write_then_read(self, registry: ToolRegistry) -> None:
        w = await registry.dispatch(
            "file_write", {"path": "sub/hello.txt", "content": "hi"}
        )
        assert not w.is_error, w.content
        r = await registry.dispatch("file_read", {"path": "sub/hello.txt"})
        assert not r.is_error
        assert r.content == "hi"

    async def test_file_list(self, registry: ToolRegistry) -> None:
        await registry.dispatch("file_write", {"path": "a.txt", "content": "1"})
        await registry.dispatch("file_write", {"path": "b.txt", "content": "2"})
        r = await registry.dispatch("file_list", {"path": ""})
        assert not r.is_error
        assert "file\ta.txt" in r.content
        assert "file\tb.txt" in r.content

    async def test_file_read_truncates(self, registry: ToolRegistry) -> None:
        big = "x" * 200
        await registry.dispatch("file_write", {"path": "big.txt", "content": big})
        r = await registry.dispatch("file_read", {"path": "big.txt", "max_bytes": 50})
        assert not r.is_error
        assert r.content.endswith("…[truncated at 50 bytes]")
        assert len(r.content) < 200

    async def test_sandbox_escape_rejected(self, registry: ToolRegistry) -> None:
        r = await registry.dispatch("file_read", {"path": "../etc/passwd"})
        assert r.is_error
        assert "not inside agent root" in r.content

    async def test_absolute_path_rejected(self, registry: ToolRegistry) -> None:
        r = await registry.dispatch("file_read", {"path": "/etc/passwd"})
        assert r.is_error

    async def test_dispatch_missing_tool(self, registry: ToolRegistry) -> None:
        r = await registry.dispatch("no_such_tool", {})
        assert r.is_error
        assert "no such tool" in r.content

    async def test_http_fetch_network_error(
        self, registry: ToolRegistry
    ) -> None:
        # A non-routable address -> network error, returned as is_error.
        r = await registry.dispatch(
            "http_fetch", {"url": "http://127.0.0.1:1/nope", "timeout": 0.1}
        )
        assert r.is_error
        assert "http request failed" in r.content
