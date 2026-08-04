"""Generic async persistence infrastructure for the ``core`` service.

Concept-agnostic: the ORM ``MetaData`` is passed into ``init_db`` by the
caller, so no import path back to domain modules. Caller owns engine +
session factory lifetimes; no module-level singleton.

SQLite specifics:
- URL: ``sqlite+aiosqlite://`` (``:memory:`` or file path)
- ``check_same_thread=False`` for shared connection pool
- WAL mode + ``foreign_keys=ON`` on every connection
"""

from __future__ import annotations

from sqlalchemy import MetaData, event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from core.settings import CoreSettings

__all__ = [
    "AsyncSession",
    "create_engine",
    "db_url",
    "init_db",
    "make_session_factory",
]


def db_url(settings: CoreSettings) -> str:
    """Build the async SQLAlchemy URL from :class:`CoreSettings`.

    ``storage.sqlite_path`` may be ``:memory:`` or a (relative or absolute)
    filesystem path; the ``sqlite+aiosqlite://`` prefix is added here.
    """
    path = settings.storage.sqlite_path
    if path == ":memory:":
        return "sqlite+aiosqlite:///:memory:"
    return f"sqlite+aiosqlite:///{path}"


def _install_pragmas(sync_engine) -> None:  # type: ignore[no-untyped-def]
    """Set WAL + foreign_keys=ON on every raw DBAPI connection this engine opens."""

    @event.listens_for(sync_engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _record):  # type: ignore[no-untyped-def]
        cur = dbapi_conn.cursor()
        try:
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
        finally:
            cur.close()


def create_engine(settings: CoreSettings) -> AsyncEngine:
    """Create a standalone ``AsyncEngine`` for the given settings.

    Callers (app bootstrap, tests) own the returned engine's lifetime.
    Tests wanting isolation should pass a settings whose
    ``storage.sqlite_path`` is ``":memory:"`` rather than touching any
    shared state.
    """
    engine = create_async_engine(
        db_url(settings),
        echo=False,
        connect_args={"check_same_thread": False},
    )
    _install_pragmas(engine.sync_engine)
    return engine


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Build an ``async_sessionmaker`` bound to ``engine``.

    ``expire_on_commit=False`` so ORM objects remain readable after a commit --
    store code yields loaded rows to callers (and reads attributes) outside
    the session that produced them.
    """
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


async def init_db(engine: AsyncEngine, metadata: MetaData) -> None:
    """Create all tables declared on ``metadata`` if missing. Idempotent.

    The caller passes the domain ``MetaData`` so this module never needs to
    import any domain module. Call once at startup.
    """
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
