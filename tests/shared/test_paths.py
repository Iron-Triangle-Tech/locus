"""Tests for the per-user data directory bootstrap (shared.paths).

Uses ``LOCUS_DATA_DIR`` to point the resolution at a temp dir so tests never
touch the real ``~/Locus``. ``key`` ids keep nested children out of ``root``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.paths import (
    data_dir,
    db_path,
    default_config_path,
    ensure_data_dir,
    workspace_path,
)


def test_ensure_data_dir_creates_root_and_workspace(tmp_path: Path) -> None:
    root = tmp_path / "Locus"
    out = ensure_data_dir(root)
    assert out == root
    assert root.is_dir()
    assert (root / "workspace").is_dir()


def test_ensure_data_dir_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "Locus"
    ensure_data_dir(root)
    # Second call must not raise and must keep both dirs present.
    ensure_data_dir(root)
    assert root.is_dir()
    assert (root / "workspace").is_dir()


def test_data_dir_honors_locus_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LOCUS_DATA_DIR", str(tmp_path / "custom"))
    root = data_dir()
    assert root == tmp_path / "custom"
    assert root.is_dir()
    assert (root / "workspace").is_dir()
    # Derived paths all live under the resolved root.
    assert default_config_path() == root / "config.toml"
    assert db_path() == root / "locus.db.sqlite"
    assert workspace_path() == root / "workspace"


def test_data_dir_no_env_uses_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # No LOCUS_DATA_DIR: resolution falls back to ~/Locus. Point HOME at the
    # tmp dir so we don't touch the real home during the test.
    monkeypatch.delenv("LOCUS_DATA_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    root = data_dir()
    assert root == tmp_path / "Locus"
    assert root.is_dir()
