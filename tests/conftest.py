"""Shared pytest fixtures for the Locus test suite.

The settings layer (``core.settings.get_settings`` / ``shared.paths``) resolves a
per-user data directory at ``~/Locus`` by default. To keep the test suite from
touching the real user home (creating ``~/Locus`` / writing a real DB), we force
``LOCUS_DATA_DIR`` at a temp path for the whole session. Individual tests that
need a fresh per-test dir are free to monkeypatch ``LOCUS_DATA_DIR`` again --
last-set wins on the next ``ensure_data_dir()`` / ``get_settings()`` call.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect ``~/Locus`` resolution to a session-scoped temp dir for every test."""
    root = tmp_path_factory.mktemp("locus_data")
    monkeypatch.setenv("LOCUS_DATA_DIR", str(root))
