from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from shared.protocol import CoreFrame, ErrorEvent

__all__ = ["EventBus", "Subscription"]

_log = logging.getLogger(__name__)


class Subscription:
    """A single consumer's view of the bus (one bounded queue)."""

    def __init__(
        self,
        queue: asyncio.Queue[CoreFrame | None],
        thread_id: str | None,
        drop_counter: list[int],
    ) -> None:
        self._queue = queue
        self._thread_id = thread_id
        self._drop_counter = drop_counter

    async def events(self) -> AsyncIterator[CoreFrame]:
        """Yield frames as they arrive; exits when the bus is closed."""
        while True:
            item = await self._queue.get()
            if item is None:  # sentinel: bus closed
                return
            yield item

    @property
    def dropped(self) -> int:
        """Total frames dropped from this subscription's queue so far."""
        return sum(self._drop_counter)


class EventBus:
    """Async multi-subscriber pub/sub of :class:`CoreFrame` agent events."""

    def __init__(self, queue_size: int = 1024) -> None:
        self._queue_size = queue_size
        # (queue, thread_id_filter_or_None, drop_counter_holder)
        self._subs: list[tuple[asyncio.Queue[CoreFrame | None], str | None, list[int]]] = []
        self._closed = False

    def subscribe(self, thread_id: str | None = None) -> Subscription:
        """Subscribe, optionally filtering to one ``thread_id``'s frames.

        The returned :class:`Subscription` is the only way to read this
        subscriber's stream. Dropping the reference stops receiving; the
        underlying queue stays until :meth:`unsubscribe` or :meth:`close`.
        """
        if self._closed:
            # Subscribe-then-immediately-end so callers don't special-case.
            q: asyncio.Queue[CoreFrame | None] = asyncio.Queue(maxsize=self._queue_size)
            q.put_nowait(None)
            return Subscription(q, thread_id, [0])
        q = asyncio.Queue(maxsize=self._queue_size)
        self._subs.append((q, thread_id, [0]))
        return Subscription(q, thread_id, self._subs[-1][2])

    def unsubscribe(self, sub: Subscription) -> None:
        """Remove a subscription's queue; no-op if not found."""
        target = sub._queue  # type: ignore[attr-defined]
        before = len(self._subs)
        self._subs = [s for s in self._subs if s[0] is not target]
        if len(self._subs) != before:
            # Signal the consumer's iterator to exit if it is awaiting.
            try:
                target.put_nowait(None)
            except asyncio.QueueFull:
                pass

    def publish(self, frame: CoreFrame) -> None:
        """Fan-out ``frame`` to every matching subscriber, drop-oldest on full.

        Never blocks. Synchronous because all queue ops are ``put_nowait``.
        """
        if self._closed:
            return
        for q, tid, drops in self._subs:
            if tid is not None:
                f_tid = getattr(frame, "thread_id", None)
                if f_tid != tid:
                    continue
            if q.full():
                try:
                    q.get_nowait()
                    drops[0] += 1
                    # Tell this subscriber (and only this one) that frames
                    # were dropped so the user is not silently starved by a
                    # slow consumer.
                    q.put_nowait(
                        ErrorEvent(
                            thread_id=f_tid,
                            message=f"event bus: dropped {drops[0]} frame(s) due to slow consumer",
                            fatal=False,
                        )
                    )
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                # Edge case: a drop-notification we just inserted filled it.
                drops[0] += 1
                _log.warning(
                    "event bus: frame dropped (queue full after drop-notify) thread_id=%s",
                    getattr(frame, "thread_id", None),
                )

    def close(self) -> None:
        """Signal all subscribers to exit; idempotent."""
        if self._closed:
            return
        self._closed = True
        for q, _tid, _drops in self._subs:
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                # Force-drain one to slot the sentinel.
                try:
                    q.get_nowait()
                    q.put_nowait(None)
                except asyncio.QueueEmpty:
                    pass
        self._subs.clear()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)
