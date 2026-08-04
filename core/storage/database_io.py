"""ORM models + async CRUD store for the agent's persistent memory.

Schema (4 tables):
- Thread: one conversation; pins provider + model at creation
- Message: a turn ordered by per-thread seq (user/assistant only)
- ToolCallRow: assistant tool call with Core-generated UUID id
- ToolResultRow: call result, unique per call_id (retries get new UUID)

Tool results live only in ToolResultRow; history reconstruction JOINs
messages -> tool_calls -> tool_results by call_id. Timestamps are tz-aware UTC.
JSON stored as TEXT (SQLite has no native JSON type).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from core.providers.base import AssistantTurn, ToolCall, ToolResultMessage

__all__ = [
    "Base",
    "MemoryStore",
    "Message",
    "Thread",
    "ToolCallRow",
    "ToolResultRow",
    "uuid4_str",
]


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


class Base(DeclarativeBase):
    """Declarative base for the memory database."""


def _utcnow() -> datetime:
    """Timezone-aware UTC now (stored in every ``created_at``)."""
    return datetime.now(timezone.utc)


class Thread(Base):
    """A single conversation thread.

    ``provider`` / ``model`` are *pinned* at creation so the conversation's
    history is replayed against the same model on every turn. ``None`` /
    ``"auto"`` values resolve to defaults at read time, but a normal flow
    pins concrete values up front.
    """

    __tablename__ = "threads"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    title: Mapped[str | None] = mapped_column(String(256), nullable=True)

    # ``passive_deletes=True`` lets the DB-level ``ON DELETE CASCADE`` (FK)
    # remove the child rows on a thread delete, instead of SQLAlchemy's
    # unit-of-work issuing its own child DELETEs (which race the FK cascade
    # and raise a spurious "expected to delete N rows; 0 were matched"
    # SAWarning). ``delete-orphan`` is kept so explicitly removing a child
    # from this collection still cascades to a delete in the ORM.
    messages: Mapped[list[Message]] = relationship(
        back_populates="thread",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.seq",
    )
    tool_calls: Mapped[list[ToolCallRow]] = relationship(
        back_populates="thread",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    tool_results: Mapped[list[ToolResultRow]] = relationship(
        back_populates="thread",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Message(Base):
    """A single conversational turn belonging to a :class:`Thread`.

    ``role`` is one of ``user`` or ``assistant``. Tool results are NOT stored
    as message rows; they live in :class:`ToolResultRow` and are JOINed when
    reconstructing history. ``tool_calls_json`` is non-NULL only for assistant
    turns that requested tool calls; it holds the JSON-encoded neutral
    ``ToolCall`` list.

    ``seq`` is a dense, monotonically increasing integer scoped to the
    thread, giving stable ordering with no reliance on wall-clock timestamps.
    Enforced unique per thread.
    """

    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("thread_id", "seq", name="uq_messages_thread_seq"),
        Index("ix_messages_thread_seq", "thread_id", "seq"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("threads.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(
        Enum("user", "assistant", name="message_role"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    tool_calls_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    thread: Mapped[Thread] = relationship(back_populates="messages")


class ToolCallRow(Base):
    """A tool call requested by an assistant turn.

    ``id`` is the Core-generated UUID tool-call id (independent of the
    provider's native id). ``message_id`` points at the assistant
    :class:`Message` that produced it.
    """

    __tablename__ = "tool_calls"
    __table_args__ = (Index("ix_tool_calls_thread_id", "thread_id"),)

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    thread_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("threads.id", ondelete="CASCADE"), nullable=False
    )
    message_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    arguments_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    thread: Mapped[Thread] = relationship(back_populates="tool_calls")


class ToolResultRow(Base):
    """The result of a :class:`ToolCallRow`, fed back to the model.

    ``call_id`` references :attr:`ToolCallRow.id` and is unique: one result
    per call id. A *retried* tool call gets a NEW uuid (a new ToolCallRow),
    not a second result row for the same id.
    """

    __tablename__ = "tool_results"
    __table_args__ = (
        UniqueConstraint("call_id", name="uq_tool_results_call_id"),
        Index("ix_tool_results_call_id", "call_id"),
        Index("ix_tool_results_thread_id", "thread_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(String(128), nullable=False)
    thread_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("threads.id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    is_error: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    thread: Mapped[Thread] = relationship(back_populates="tool_results")


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


def uuid4_str() -> str:
    """Default id factory: a fresh UUID4 as a lowercase string."""
    return str(uuid.uuid4())


class MemoryStore:
    """Async CRUD facade over the memory tables.

    The only thing the agent loop talks to for persistence. It owns the
    *memory concept*: how rows become the neutral :class:`AssistantTurn` /
    :class:`ToolResultMessage` history the provider consumes, and how the
    loop's produced turns/tool records get persisted. It does NOT own the
    storage infra (engine, session factory) -- that's injected as
    ``session_factory`` from :mod:`core.storage.session`.

    Rules (confirmed against the loop contract):

    * Call ids are **Core-generated UUIDs** minted by the loop *before* calling
      :meth:`append_assistant_message`. The store trusts ``ToolCall.id`` and
      uses it verbatim as the ``ToolCallRow.id`` primary key; it does NOT mint
      call ids. The store *does* mint thread ids (via ``id_factory``) when
      :meth:`create_thread` isn't given one.
    * :meth:`load_history` returns every persisted assistant turn (including
      ones whose tool results are still in flight); the loop tracks in-flight
      calls in memory and avoids re-sending them. The store reflects
      persistence faithfully.
    * Each append is a single atomic transaction (commit on success, rollback
      on error). :meth:`append_assistant_message` writes the :class:`Message`
      and ALL its :class:`ToolCallRow` rows together.
    * ``prior_tool_results`` is returned as a flat list separate from the
      assistant turns, matching :meth:`Provider.complete`'s signature.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        id_factory: Callable[[], str] = uuid4_str,
    ) -> None:
        self._sf = session_factory
        self._id = id_factory

    # ---- Thread lifecycle -------------------------------------------------

    async def create_thread(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        title: str | None = None,
        thread_id: str | None = None,
    ) -> Thread:
        """Create + persist a new :class:`Thread`, pinning provider/model.

        If ``thread_id`` is omitted a fresh UUID (from ``id_factory``) is used.
        """
        thread = Thread(
            id=thread_id or self._id(),
            provider=provider,
            model=model,
            title=title,
        )
        async with self._sf() as session:
            session.add(thread)
            await session.commit()
            # Returned object is detached (expire_on_commit=False) so its
            # attributes remain readable after the session closes.
            return thread

    async def get_thread(self, thread_id: str) -> Thread | None:
        """Fetch a thread by id, or ``None`` if it doesn't exist."""
        async with self._sf() as session:
            return await session.get(Thread, thread_id)

    async def set_thread_title(self, thread_id: str, title: str) -> None:
        """Set a thread's title (overwriting any existing value)."""
        async with self._sf() as session:
            thread = await session.get(Thread, thread_id)
            if thread is None:
                raise KeyError(f"thread not found: {thread_id!r}")
            thread.title = title
            await session.commit()

    # ---- Appends (each a single transaction) -----------------------------

    async def append_user_message(self, thread_id: str, content: str) -> Message:
        """Append a ``role=user`` message to ``thread_id`` at the next seq."""
        async with self._sf() as session:
            seq = await self._next_seq(session, thread_id)
            msg = Message(
                thread_id=thread_id,
                role="user",
                seq=seq,
                content=content,
            )
            session.add(msg)
            await session.commit()
            return msg

    async def append_assistant_message(
        self,
        thread_id: str,
        *,
        text: str,
        tool_calls: list[ToolCall] | None = None,
    ) -> tuple[Message, list[ToolCallRow]]:
        """Atomically append an assistant turn and all its :class:`ToolCallRow`s.

        ``tool_calls`` are neutral :class:`ToolCall` objects whose ``id`` is the
        Core-generated call id (minted by the loop BEFORE this call). The store
        uses ``id`` verbatim as the ``ToolCallRow.id`` primary key.

        Returns the persisted ``Message`` and the persisted ``ToolCallRow``
        list (empty if ``tool_calls`` was empty/None).
        """
        calls = list(tool_calls or [])
        async with self._sf() as session:
            seq = await self._next_seq(session, thread_id)
            msg = Message(
                thread_id=thread_id,
                role="assistant",
                seq=seq,
                content=text,
                tool_calls_json=json.dumps([c.model_dump() for c in calls]),
            )
            session.add(msg)
            await session.flush()  # populate msg.id for the FK
            call_rows: list[ToolCallRow] = []
            for c in calls:
                row = ToolCallRow(
                    id=c.id,
                    thread_id=thread_id,
                    message_id=msg.id,
                    name=c.name,
                    arguments_json=json.dumps(c.arguments),
                )
                session.add(row)
                call_rows.append(row)
            await session.commit()
            return msg, call_rows

    async def append_tool_result(
        self,
        thread_id: str,
        *,
        call_id: str,
        content: str,
        is_error: bool = False,
    ) -> ToolResultRow:
        """Persist the result of a tool call (one row per call id)."""
        async with self._sf() as session:
            row = ToolResultRow(
                call_id=call_id,
                thread_id=thread_id,
                content=content,
                is_error=is_error,
            )
            session.add(row)
            await session.commit()
            return row

    # ---- History reconstruction for the provider --------------------------

    async def load_history(
        self, thread_id: str
    ) -> tuple[list[AssistantTurn], list[ToolResultMessage]]:
        """Reconstruct provider-consumable history for ``thread_id``.

        Returns ``(assistant_turns, prior_tool_results)``:

        * ``assistant_turns`` -- one :class:`AssistantTurn` per assistant
          :class:`Message` ordered by ``seq``. Includes turns whose tool
          results are still in flight; the loop must avoid re-sending those.
        * ``prior_tool_results`` -- one :class:`ToolResultMessage` per
          :class:`ToolResultRow`, in call arrival order. ``name`` is fetched
          by left-JOIN to :class:`ToolCallRow` on ``call_id`` and falls back
          to ``""`` if the call row is missing.
        """
        async with self._sf() as session:
            turns = await self._load_assistant_turns(session, thread_id)
            results = await self._load_tool_results(session, thread_id)
            return turns, results

    async def pending_tool_calls(self, thread_id: str) -> list[ToolCallRow]:
        """Return tool calls for ``thread_id`` with no persisted result yet.

        The loop uses this to keep outstanding calls out of the next provider
        request (e.g. on resume after an endpoint disconnect).
        """
        async with self._sf() as session:
            # LEFT JOIN tool_results; keep rows where there is no matching
            # result row. ``ToolResultRow.id`` is NULL only when the join
            # found nothing (one result per call id by unique constraint).
            stmt = (
                select(ToolCallRow)
                .outerjoin(
                    ToolResultRow,
                    ToolResultRow.call_id == ToolCallRow.id,
                )
                .where(
                    ToolCallRow.thread_id == thread_id,
                    ToolResultRow.id.is_(None),
                )
            )
            rows = (await session.execute(stmt)).scalars().all()
            return list(rows)

    # ---- internals --------------------------------------------------------

    async def _next_seq(self, session: AsyncSession, thread_id: str) -> int:
        """Next ``seq`` for the thread: 0 if empty, else max(seq)+1."""
        stmt = select(Message.seq).where(Message.thread_id == thread_id)
        existing = (await session.execute(stmt)).scalars().all()
        if not existing:
            return 0
        return max(existing) + 1

    async def _load_assistant_turns(
        self, session: AsyncSession, thread_id: str
    ) -> list[AssistantTurn]:
        stmt = (
            select(Message)
            .where(
                Message.thread_id == thread_id,
                Message.role == "assistant",
            )
            .order_by(Message.seq)
        )
        rows = (await session.execute(stmt)).scalars().all()
        turns: list[AssistantTurn] = []
        for r in rows:
            calls: list[ToolCall] = []
            if r.tool_calls_json:
                raw = json.loads(r.tool_calls_json)
                calls = [ToolCall.model_validate(item) for item in raw]
            turns.append(AssistantTurn(text=r.content, tool_calls=calls))
        return turns

    async def _load_tool_results(
        self, session: AsyncSession, thread_id: str
    ) -> list[ToolResultMessage]:
        # One ToolResultMessage per ToolResultRow. ``name`` comes from the
        # matching ToolCallRow (left-JOIN on call_id); missing -> "".
        stmt = (
            select(ToolResultRow, ToolCallRow.name)
            .outerjoin(ToolCallRow, ToolCallRow.id == ToolResultRow.call_id)
            .where(ToolResultRow.thread_id == thread_id)
            .order_by(ToolResultRow.id)
        )
        out: list[ToolResultMessage] = []
        for row, name in (await session.execute(stmt)).all():
            out.append(
                ToolResultMessage(
                    call_id=row.call_id,
                    name=name or "",
                    content=row.content,
                    is_error=row.is_error,
                )
            )
        return out
