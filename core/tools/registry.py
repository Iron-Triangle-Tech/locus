"""Neutral tool registry for the core's built-in tool kit.

Behavior-only :class:`Tool` protocol + :class:`ToolRegistry` joining runnables
(code) with metadata defs (memory). See module history for full design.
"""

from __future__ import annotations

import logging
import traceback
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from core.providers.base import ToolDef

__all__ = ["DuplicateToolError", "Tool", "ToolRegistry", "ToolResult"]

_log = logging.getLogger(__name__)


class ToolResult(BaseModel):
    """The result of running a tool, fed back to the model as a tool message."""

    model_config = ConfigDict(extra="forbid")

    content: str
    is_error: bool = False


@runtime_checkable
class Tool(Protocol):
    """A built-in tool runnable by the core -- BEHAVIOR ONLY."""

    name: str

    async def run(self, arguments: dict) -> ToolResult: ...


class DuplicateToolError(ValueError):
    """Raised by :meth:`ToolRegistry.register_compiled_tools` on a dup name."""


class ToolRegistry:
    """In-process registry joining runnables (code) with defs (memory).

    Construct empty, then :meth:`register_compiled_tools` with the Python
    runnables, then :meth:`load_defs` with the metadata dict read from the
    store. :meth:`export_defs` advertises the **intersection**.
    """

    def __init__(self) -> None:
        self._runnables: dict[str, Tool] = {}
        self._defs: dict[str, ToolDef] = {}

    def register_compiled_tools(self, *tools: Tool) -> None:
        """Add runnable tools. Duplicate ``name`` raises :class:`DuplicateToolError`."""
        for tool in tools:
            if tool.name in self._runnables:
                raise DuplicateToolError(f"tool already registered: {tool.name!r}")
            self._runnables[tool.name] = tool

    def load_defs(self, defs: dict[str, ToolDef]) -> None:
        """Attach the per-name metadata read from the ``tool_defs`` store."""
        self._defs = dict(defs)

    def get(self, name: str) -> Tool:
        """Fetch a runnable by name. Raises ``KeyError`` if not registered."""
        try:
            return self._runnables[name]
        except KeyError as e:
            raise KeyError(f"tool runnable not registered: {name!r}") from e

    def names(self) -> list[str]:
        """Sorted names of the EXPORTABLE tools (runnable ∩ def)."""
        return sorted(self._runnables.keys() & self._defs.keys())

    def export_defs(self) -> list[ToolDef]:
        """Snapshot of defs for the EXPORTABLE tools (intersection), sorted by name.

        Warns once at debug level for mismatches so the operator notices:
        a def whose name has no runnable, or a runnable whose name has no def.
        """
        runnable_names = set(self._runnables)
        def_names = set(self._defs)
        only_defs = def_names - runnable_names
        only_runnables = runnable_names - def_names
        if only_defs:
            _log.warning(
                "tool defs without runnables (hidden from provider): %s",
                sorted(only_defs),
            )
        if only_runnables:
            _log.warning(
                "tool runnables without defs (not advertised): %s",
                sorted(only_runnables),
            )
        exported = sorted(runnable_names & def_names)
        return [self._defs[name] for name in exported]

    async def dispatch(self, name: str, arguments: dict) -> ToolResult:
        """Look up ``name`` and run it (intersection membership NOT required --
        dispatch works for any registered runnable, including ones not yet
        advertised via a def).

        Any exception raised by the tool is caught and returned as
        ``ToolResult(is_error=True, content=<formatted exception>)`` so a bad
        single call does not crash the whole agent turn. A missing tool is also
        reported this way (not raised) -- the model can react to "no such tool".
        """
        try:
            tool = self.get(name)
        except KeyError as e:
            return ToolResult(content=f"no such tool: {name!r}: {e}", is_error=True)
        try:
            return await tool.run(arguments)
        except Exception:
            tb = traceback.format_exc()
            return ToolResult(
                content=f"tool {name!r} raised:\n{tb}",
                is_error=True,
            )

    # -- container ergonomics so the loop can iterate the registry --

    def __len__(self) -> int:
        return len(self._runnables)

    def __iter__(self):
        return iter(self._runnables.values())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._runnables
