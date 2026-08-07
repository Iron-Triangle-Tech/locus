"""Built-in file tools, sandboxed to ``settings.tools.agent_root``.

Three tools implement :class:`core.tools.registry.Tool` -- BEHAVIOR ONLY:

* :class:`FileRead` (``file_read``) -- read a text file, truncated to
  ``max_bytes`` with a trailing marker if larger.
* :class:`FileWrite` (``file_write``) -- create/overwrite a file (creating
  missing parent directories).
* :class:`FileList` (``file_list``) -- list a directory's entries (name+type).

The metadata advertised to the provider (``description`` + ``parameters`` JSON
Schema) for each of these lives in ``core/tools.toml`` (the "ROM") and is
loaded into the ``tool_defs`` memory table at startup -- NOT in this file.

Sandbox rules (enforced on every call):

* Only **relative** ``path`` arguments are accepted; they are joined against
  ``agent_root``. Absolute paths and any resolved path that escapes
  ``agent_root`` (e.g. via ``..`` or symlinks) are rejected with ``is_error``.
* Reads/writes are **text only**. The caller may pass an ``encoding``
  (default ``utf-8``); common encodings are accepted. A decode/encode error
  is returned as ``is_error`` rather than raised.
* All real I/O runs off the event loop via :func:`anyio.to_thread.run_sync`
  so a slow disk does not stall the agent loop.
"""

from __future__ import annotations

from pathlib import Path

import anyio

from core.tools.registry import Tool, ToolResult

__all__ = ["FileList", "FileRead", "FileWrite", "default_file_tools"]


def _resolve(root: Path, raw: str) -> Path | None:
    """Resolve ``raw`` against ``root`` and return it iff it stays inside.

    Returns ``None`` (so the caller can emit an ``is_error`` ToolResult) on any
    rejection: absolute path, parent escape, or a symlink target outside root.
    """
    if not raw:
        candidate = root
    else:
        candidate = root / raw
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError):
        return None
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


class _FileBase:
    """Shared base: pins the resolved ``agent_root`` from settings at build."""

    def __init__(self, agent_root: Path) -> None:
        if not agent_root.exists():
            # Create the sandbox root lazily so the agent can write into it
            # without the operator precreating the dir. Best-effort; if it
            # fails we let the first tool call surface the OSError as is_error.
            try:
                agent_root.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
        self._root: Path = agent_root


class FileRead(_FileBase, Tool):
    """``file_read`` -- read a text file under the agent root, truncated."""

    name = "file_read"

    async def run(self, arguments: dict) -> ToolResult:
        raw = arguments.get("path", "")
        enc = arguments.get("encoding") or "utf-8"
        max_bytes = int(arguments.get("max_bytes") or 65536)
        path = _resolve(self._root, raw)
        if path is None:
            return ToolResult(content=f"path not inside agent root: {raw!r}", is_error=True)
        if not path.exists():
            return ToolResult(content=f"not found: {raw!r}", is_error=True)
        if path.is_dir():
            return ToolResult(content=f"is a directory: {raw!r}", is_error=True)

        try:
            data = await anyio.to_thread.run_sync(self._read_bytes, path, max_bytes)
        except OSError as e:
            return ToolResult(content=f"read failed: {e}", is_error=True)
        try:
            text, truncated = await anyio.to_thread.run_sync(self._decode, data, enc, max_bytes)
        except (LookupError, UnicodeDecodeError) as e:
            return ToolResult(content=f"decode failed ({enc}): {e}", is_error=True)
        return ToolResult(content=text, is_error=False)

    @staticmethod
    def _read_bytes(path: Path, max_bytes: int) -> bytes:
        # Read one byte past the cap; if we got that many, the file was larger
        # and we marker-truncate to ``max_bytes`` in :meth:`_decode`.
        with path.open("rb") as fh:
            return fh.read(max_bytes + 1)

    @staticmethod
    def _decode(data: bytes, enc: str, max_bytes: int) -> tuple[str, bool]:
        truncated = len(data) > max_bytes
        if truncated:
            data = data[:max_bytes]
        text = data.decode(enc)
        if truncated:
            text += f"\n…[truncated at {max_bytes} bytes]"
        return text, truncated


class FileWrite(_FileBase, Tool):
    """``file_write`` -- write text to a file under the agent root, overwriting."""

    name = "file_write"

    async def run(self, arguments: dict) -> ToolResult:
        raw = arguments.get("path", "")
        content = arguments.get("content", "")
        enc = arguments.get("encoding") or "utf-8"
        path = _resolve(self._root, raw)
        if path is None:
            return ToolResult(content=f"path not inside agent root: {raw!r}", is_error=True)
        if path.exists() and path.is_dir():
            return ToolResult(content=f"is a directory: {raw!r}", is_error=True)

        try:
            payload = content.encode(enc)
        except (LookupError, UnicodeEncodeError) as e:
            return ToolResult(content=f"encode failed ({enc}): {e}", is_error=True)
        try:
            await anyio.to_thread.run_sync(self._write, path, payload)
        except OSError as e:
            return ToolResult(content=f"write failed: {e}", is_error=True)
        return ToolResult(content=f"wrote {len(payload)} bytes to {raw!r}", is_error=False)

    @staticmethod
    def _write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            fh.write(payload)


class FileList(_FileBase, Tool):
    """``file_list`` -- list entries (name + type) under the agent root."""

    name = "file_list"

    async def run(self, arguments: dict) -> ToolResult:
        raw = arguments.get("path", "")
        path = _resolve(self._root, raw)
        if path is None:
            return ToolResult(content=f"path not inside agent root: {raw!r}", is_error=True)
        if not path.exists():
            return ToolResult(content=f"not found: {raw!r}", is_error=True)
        if not path.is_dir():
            return ToolResult(content=f"not a directory: {raw!r}", is_error=True)

        try:
            rows = await anyio.to_thread.run_sync(self._list, path)
        except OSError as e:
            return ToolResult(content=f"list failed: {e}", is_error=True)
        return ToolResult(content="\n".join(rows) or "(empty)", is_error=False)

    @staticmethod
    def _list(path: Path) -> list[str]:
        out: list[str] = []
        for entry in sorted(path.iterdir(), key=lambda p: p.name):
            if entry.is_dir():
                kind = "dir"
            elif entry.is_file():
                kind = "file"
            else:
                kind = "other"
            out.append(f"{kind}\t{entry.name}")
        return out


def default_file_tools(agent_root: Path) -> list[Tool]:
    """Convenience builder: ``[FileRead, FileWrite, FileList]`` for an app."""
    return [FileRead(agent_root), FileWrite(agent_root), FileList(agent_root)]
