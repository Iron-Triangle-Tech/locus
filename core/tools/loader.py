"""ROM loader for the core's built-in tool kit.

Parses ``core/tools.toml`` into ``ToolDef`` objects (the agent's knowledge of
which tools exist and how to call them) and seeds them into the ``tool_defs``
store table. Seeding is INSERT-only: rows whose ``name`` already exists are
left untouched, so any runtime edits a user made persist across restarts.

This module is the ONLY place that knows the ROM file exists. The agent loop
and registry talk to the store, not to this file, after startup.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from core.providers.base import ToolDef

if TYPE_CHECKING:
    from core.storage.database_io import MemoryStore

__all__ = ["DEFAULT_ROM_PATH", "load_tool_defs", "seed_missing"]

_log = logging.getLogger(__name__)

# Next to this module's package (core/tools/loader.py -> core/tools.toml).
DEFAULT_ROM_PATH = Path(__file__).resolve().parent.parent / "tools.toml"


def load_tool_defs(path: Path | str = DEFAULT_ROM_PATH) -> dict[str, ToolDef]:
    """Parse the ROM TOML into a ``{name: ToolDef}`` dict.

    Each ``[[tools]]`` entry must have ``name``, ``description`` and a
    ``parameters`` inline JSON Schema (a TOML table). Entries with a missing
    ``name`` are skipped with a warning; duplicate names raise ``ValueError``
    (the ROM is author-controlled, duplicates are a bug).
    """
    p = Path(path)
    with p.open("rb") as f:
        data = tomllib.load(f)

    raw_list = data.get("tools", [])
    if not isinstance(raw_list, list):
        raise ValueError(f"ROM {p}: top-level `tools` must be an array of tables")

    out: dict[str, ToolDef] = {}
    for i, entry in enumerate(raw_list):
        if not isinstance(entry, dict):
            _log.warning("ROM %s: tools[%d] is not a table, skipping", p, i)
            continue
        name = entry.get("name")
        if not name or not isinstance(name, str):
            _log.warning("ROM %s: tools[%d] has no string `name`, skipping", p, i)
            continue
        if name in out:
            raise ValueError(f"ROM {p}: duplicate tool name {name!r}")
        description = entry.get("description", "")
        parameters = entry.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError(f"ROM {p}: tool {name!r} `parameters` must be a table")
        out[name] = ToolDef(
            name=name,
            description=str(description),
            parameters=parameters,
        )
    return out


async def seed_missing(
    store: MemoryStore,
    defs: dict[str, ToolDef],
    *,
    log: logging.Logger | None = None,
) -> list[str]:
    """Insert a def row for each ROM name missing from the store.

    INSERT-only: existing rows are never overwritten. Returns the list of
    names actually inserted (the caller logs or asserts on it).
    """
    lg = log or _log
    inserted = await store.insert_missing_tool_defs(list(defs.values()))
    if inserted:
        lg.info("seeded %d tool def(s) from ROM: %s", len(inserted), sorted(inserted))
    else:
        lg.debug("ROM seed: no new tool defs to insert")
    return inserted
