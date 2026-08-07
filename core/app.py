"""Core FastAPI app: REST control surface + WebSocket agent-event link.

Owns process-lifetime singletons:

* the async SQLAlchemy engine + ``MemoryStore`` (memory concept) backed by SQLite,
* the :class:`ToolRegistry` with the built-in file/http tool runnables + defs
  loaded from the persistent ``tool_defs`` table (seeded from the ROM at
  startup),
* the in-process :class:`EventBus` that fans agent events out to WS links,
* the :class:`LinkRegistry` + :class:`WSLinkAdhocDispatcher` that turn endpoint
  ``ToolResult`` frames into the futures the agent loop waits on.

REST surface (thin, per the plan -- agent activity happens over WS, not REST):

* ``GET  /health``            -- liveness; no deps touched.
* ``POST /threads``          -- create a thread, pinning provider/model.
* ``GET  /threads/{id}``     -- fetch a thread's metadata.

WS surface:

* ``WS /link``               -- the core<->endpoint link. ``app.state`` carries
  everything :func:`handle_link` needs; auth rejects the upgrade *before* the
  socket is accepted via Starlette's ``HTTPException(status_code=403)``.

``main()`` is the ``locus-core`` console entry point: build settings, build the
app, hand it to uvicorn. Imports of heavy SDKs (anthropic/openai/google-genai)
are lazy in :mod:`core.providers`, so importing this module does not require
any provider SDK to be installed.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, WebSocket, status
from pydantic import BaseModel

from core.agent.loop import AgentLoop, LoopConfig
from core.bus import EventBus
from core.endpoint_conn.server import (
    LinkRegistry,
    WSLinkAdhocDispatcher,
    handle_link,
)
from core.providers import resolve_provider_name
from core.settings import CoreSettings, get_settings
from core.storage.database_io import Base, MemoryStore
from core.storage.session import create_engine, init_db, make_session_factory
from core.tools.file import default_file_tools
from core.tools.http import HttpFetch
from core.tools.loader import load_tool_defs, seed_missing
from core.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from shared.protocol import UserMessage

__all__ = ["AppState", "app_factory", "create_app", "main"]

_log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# REST request/response models
# --------------------------------------------------------------------------- #


class CreateThreadRequest(BaseModel):
    """Body for ``POST /threads``. All fields optional; sensible defaults."""

    model_config = {"extra": "forbid"}

    provider: str = "auto"
    model: str | None = None
    title: str | None = None


class ThreadResponse(BaseModel):
    """Body for ``POST /threads`` and ``GET /threads/{id}`` responses."""

    model_config = {"extra": "forbid"}

    id: str
    provider: str | None
    model: str | None
    title: str | None


# --------------------------------------------------------------------------- #
# App state container
# --------------------------------------------------------------------------- #


class AppState:
    """Process singletons owned by one app instance.

    Kept as a plain class (not a Pydantic model) because it holds SQLAlchemy
    engine/async objects that are not Pydantic-friendly. Attached to
    ``app.state`` so route handlers and the WS endpoint can reach it.
    """

    def __init__(self, settings: CoreSettings) -> None:
        self.settings = settings
        self.engine = create_engine(settings)
        self.session_factory = make_session_factory(self.engine)
        self.store = MemoryStore(self.session_factory)
        self.registry = ToolRegistry()
        self.bus = EventBus()
        self.link_registry = LinkRegistry()
        self.dispatcher = WSLinkAdhocDispatcher(self.link_registry)
        self.loop_cfg = LoopConfig()

    async def start(self) -> None:
        """Create tables, seed the ROM, register built-in tools + load defs."""
        await init_db(self.engine, Base.metadata)

        # ROM -> tool_defs table (INSERT-only for missing names).
        rom_defs = load_tool_defs()
        await seed_missing(self.store, rom_defs)

        # Built-in runnables: file (read/write/list) + http_fetch, sandboxed
        # per settings.tools.
        agent_root = Path(self.settings.tools.agent_root).resolve()
        for tool in default_file_tools(agent_root):
            self.registry.register_compiled_tools(tool)
        self.registry.register_compiled_tools(
            HttpFetch(
                max_bytes=self.settings.tools.http_max_bytes,
                default_timeout=self.settings.tools.http_timeout,
            )
        )

        # Defs = what's in the tool_defs table; registry advertises the
        # intersection (runnable AND def). Tools present in ROM but not yet
        # registered as runnables are warned + hidden by export_defs.
        defs = await self.store.list_tool_defs()
        self.registry.load_defs({d.name: d for d in defs})

    async def stop(self) -> None:
        """Close the bus + dispose the engine. Idempotent-ish (safe to call
        twice; the second call is a no-op)."""
        self.bus.close()
        await self.engine.dispose()


# --------------------------------------------------------------------------- #
# App factory + lifespan
# --------------------------------------------------------------------------- #


def app_factory(state: AppState) -> FastAPI:
    """Build the FastAPI app wire-rest onto ``state``. No side effects."""

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        await state.start()
        try:
            yield
        finally:
            await state.stop()

    app = FastAPI(
        title="Locus core",
        version="0.1.0",
        lifespan=_lifespan,
        # The WS endpoint does its own auth; nothing else needs docs/CORS here.
    )
    app.state.locus = state
    _wire_routes(app, state)
    return app


def create_app(settings: CoreSettings | None = None) -> FastAPI:
    """Build a runnable app from ``settings`` (default: the on-disk config).

    Entry point for tests: ``create_app(SettingsForTest())`` gives an app whose
    lifespan boots an isolated in-memory store + seeded ROM.
    """
    s = settings or get_settings()
    state = AppState(s)
    return app_factory(state)


# --------------------------------------------------------------------------- #
# Route wiring
# --------------------------------------------------------------------------- #


def _wire_routes(app: FastAPI, state: AppState) -> None:
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/threads", response_model=ThreadResponse)
    async def create_thread(req: CreateThreadRequest) -> ThreadResponse:
        try:
            provider = resolve_provider_name(req.provider, state.settings)
        except KeyError as e:
            # Unknown provider name is a client error, not a 500.
            raise HTTPException(status_code=400, detail=str(e)) from e
        model = req.model or state.settings.resolved_model(provider) or None
        thread = await state.store.create_thread(
            provider=provider, model=model, title=req.title
        )
        return ThreadResponse(
            id=thread.id, provider=thread.provider, model=thread.model, title=thread.title
        )

    @app.get("/threads/{thread_id}", response_model=ThreadResponse)
    async def get_thread(thread_id: str) -> ThreadResponse:
        thread = await state.store.get_thread(thread_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="thread not found")
        return ThreadResponse(
            id=thread.id, provider=thread.provider, model=thread.model, title=thread.title
        )

    @app.websocket("/link")
    async def link(websocket: WebSocket) -> None:
        # Auth the upgrade BEFORE accepting so a bad token produces a clean
        # 403 instead of an accepted-then-closed socket. Starlette raises
        # HTTPException out of a websocket route before the handshake when the
        # socket has not been accepted yet; here we accept-then-close only as
        # a fallback.
        headers = {k: v for k, v in websocket.headers.items()}
        from shared.auth import check_token

        expected = state.settings.link_token
        if not expected or not check_token(headers, expected):
            # Reject the upgrade with 403. Starlette maps this to an HTTP 403
            # response and does NOT upgrade the connection.
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="unauthorized"
            )

        await websocket.accept()
        await handle_link(
            websocket,
            bus=state.bus,
            registry=state.link_registry,
            expected_token=expected,
            on_user_message=_make_on_user_message(state),
            idle_timeout=state.settings.server.ws_idle_timeout,
        )


def _make_on_user_message(state: AppState):  # type: ignore[no-untyped-def]
    """Build the callback the link reader calls for each inbound UserMessage.

    The callback resolves the provider + (re)uses the shared dispatcher and
    spawns the agent turn as a background task so the reader keeps draining.
    If the ``UserMessage`` carries no ``thread_id``, a fresh thread is created
    first (provider pinned from the message / settings default) so the loop
    has something to write into. The created/fetched thread id is what the
    loop publishes under; the endpoint sees it on every event's ``thread_id``.
    """

    async def _spawn(frame: "UserMessage") -> None:
        provider_name = frame.provider or "auto"
        try:
            provider = resolve_provider_name(provider_name, state.settings)
        except KeyError as e:
            state.bus.publish(
                _error_event(frame.thread_id, f"unknown provider: {e}")
            )
            return

        thread_id = frame.thread_id
        if thread_id is None:
            model = state.settings.resolved_model(provider) or None
            thread = await state.store.create_thread(provider=provider, model=model)
            thread_id = thread.id
        else:
            existing = await state.store.get_thread(thread_id)
            if existing is None:
                # Create with the requested id so the loop has a row to write
                # into; pin provider/model from the request / settings.
                model = state.settings.resolved_model(provider) or None
                await state.store.create_thread(
                    provider=provider, model=model, thread_id=thread_id
                )

        loop = AgentLoop(
            settings=state.settings,
            store=state.store,
            registry=state.registry,
            bus=state.bus,
            dispatcher=state.dispatcher,
            config=state.loop_cfg,
        )
        # Fire-and-forget: events flow back over the bus -> WS pump. Errors
        # are emitted on the bus by the loop itself, not raised here.
        asyncio.create_task(
            loop.run_agent_turn(thread_id, frame.content, provider_name=provider),
            name=f"agent-turn-{thread_id}",
        )

    return _spawn


def _error_event(thread_id: str | None, message: str):  # type: ignore[no-untyped-def]
    from shared.protocol import ErrorEvent

    return ErrorEvent(thread_id=thread_id, message=message, fatal=True)


# --------------------------------------------------------------------------- #
# Console entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    """``locus-core`` entry point: build the app and run it under uvicorn."""
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    app = create_app(settings)

    # uvicorn owns the event loop and drives ``app.router.lifespan``. The host
    # and port come from the same settings everything else does, so a single
    # ``LOCUS_CORE_SERVER_HOST``/``LOCUS_CORE_SERVER_PORT`` env override moves
    # the whole process.
    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        log_level="info",
    )
