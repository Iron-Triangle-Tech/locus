"""Resolve and bootstrap the per-user Locus data directory.

Locus keeps runtime state -- the SQLite database, the agent file workspace, and
a user-overridable ``config.toml`` -- in a single per-user data directory,
rather than relative to the process's current working directory. ``pip``/wheels
cannot create directories outside the installed package, so this directory is
bootstrapped lazily on first use (first call to ``core.settings.get_settings()``),
not at install time.

Layout (created on first run, mode 0o700):

* macOS / Linux / other POSIX    ~/Locus
* Windows                        %APPDATA%\\Locus

Override the base with the ``LOCUS_DATA_DIR`` environment variable (handy for
tests, self-contained deployments, and containers). The directory and its
``workspace/`` child are created by :func:`ensure_data_dir`; the DB file and the
user ``config.toml`` are created lazily by their owners (SQLAlchemy / the user),
not here -- this module only guarantees the parent exists.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

__all__ = [
    "data_dir",
    "db_path",
    "default_config_path",
    "ensure_data_dir",
    "workspace_path",
]


def _base_dir() -> Path:
    """Return the base Locus data directory path (without creating it)."""
    env = os.environ.get("LOCUS_DATA_DIR")
    if env:
        return Path(env).expanduser()
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return base / "Locus"
    # macOS, Linux, and any other POSIX: a flat ~/Locus dir in the home folder.
    return Path.home() / "Locus"


def ensure_data_dir(root: Path | None = None) -> Path:
    """Create ``root`` (default: platform default) + its ``workspace/`` child.

    Mode 0o700. Idempotent: safe to call every startup. Best-effort tightens
    permissions if the dir pre-existed with looser bits (silently ignores
    platforms where chmod can't enforce it). Returns the resolved root.
    """
    target = root or _base_dir()
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    (target / "workspace").mkdir(parents=True, exist_ok=True, mode=0o700)
    with contextlib.suppress(OSError):
        os.chmod(target, 0o700)
        os.chmod(target / "workspace", 0o700)
    return target


def data_dir() -> Path:
    """Return the Locus data directory for the current platform (creating it)."""
    return ensure_data_dir()


def default_config_path() -> Path:
    """Path to the user ``config.toml`` (may not exist yet)."""
    return data_dir() / "config.toml"


def db_path() -> Path:
    """Path to the SQLite database file. Created on first DB connect by SQLAlchemy."""
    return data_dir() / "locus.db.sqlite"


def workspace_path() -> Path:
    """Path to the agent's on-disk file workspace (the file-tools sandbox root)."""
    return data_dir() / "workspace"
