"""
core/events.py
--------------
Async event bus using asyncio.Queue for decoupled communication
between the trading engine components (market data → strategy → execution → risk).

Usage:
    bus = EventBus()
    await bus.publish(BarEvent(symbol="BTC/USD", bar=my_bar))

    async for event in bus.subscribe(BarEvent):
        await handle_bar(event)
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator, Type, TypeVar

from core.types import Bar, Fill, Order, Signal


# ── Event Base & Concrete Events ─────────────────────────────────────────────────

@dataclass
class Event:
    """Base class for all engine events."""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BarEvent(Event):
    """Emitted when a new OHLCV bar is complete."""
    bar: Bar | None = None


@dataclass
class TickEvent(Event):
    """Emitted on every raw price tick."""
    symbol: str = ""
    price:  float = 0.0
    volume: float = 0.0


@dataclass
class SignalEvent(Event):
    """Emitted when the strategy generates a signal."""
    signal: Signal | None = None


@dataclass
class OrderEvent(Event):
    """Emitted when an order is created or updated."""
    order: Order | None = None


@dataclass
class FillEvent(Event):
    """Emitted when a broker fill confirmation arrives."""
    fill: Fill | None = None


@dataclass
class ErrorEvent(Event):
    """Emitted when a non-fatal error occurs in any component."""
    source:  str = ""
    message: str = ""
    exc:     Exception | None = None


@dataclass
class ShutdownEvent(Event):
    """Emitted to signal a graceful shutdown of all components."""
    reason: str = "manual"


# ── Type variable for generic subscriptions ────────────────────────────────────────

E = TypeVar("E", bound=Event)

# ── Event Bus ─────────────────────────────────────────────────────────────────────

class EventBus:
    """
    Async publish/subscribe event bus.

    - Publishers call `await bus.publish(event)` — O(1), non-blocking.
    - Subscribers get a dedicated `asyncio.Queue` per event type.
    - No shared state between subscribers (fan-out pattern).
    - Backpressure: queues have a configurable max size; full queues drop
      oldest events (LIFO overflow) to prevent memory unbounded growth.
    """

    def __init__(self, max_queue_size: int = 1_000) -> None:
        self._queues: dict[type, list[asyncio.Queue]] = defaultdict(list)
        self._max_queue_size = max_queue_size
        self._lock = asyncio.Lock()

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers of its type."""
        event_type = type(event)
        queues = self._queues.get(event_type, [])
        for q in queues:
            if q.full():
                # Drain oldest event to make room (ring-buffer behaviour)
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await q.put(event)

    async def subscribe(self, event_type: Type[E]) -> AsyncIterator[E]:
        """
        Async generator that yields events of *event_type* as they arrive.
        Creates a dedicated queue for this subscriber.

        Usage:
            async for event in bus.subscribe(BarEvent):
                process(event)
        """
        q: asyncio.Queue[E] = asyncio.Queue(maxsize=self._max_queue_size)
        async with self._lock:
            self._queues[event_type].append(q)
        try:
            while True:
                event = await q.get()
                if isinstance(event, ShutdownEvent):
                    return
                yield event
        finally:
            async with self._lock:
                queues = self._queues.get(event_type, [])
                if q in queues:
                    queues.remove(q)

    async def broadcast_shutdown(self) -> None:
        """Signal all subscribers to stop — sends ShutdownEvent to every queue."""
        shutdown = ShutdownEvent()
        async with self._lock:
            for queues in self._queues.values():
                for q in queues:
                    await q.put(shutdown)

    @property
    def subscriber_count(self) -> int:
        return sum(len(qs) for qs in self._queues.values())


# ── Singleton instance (shared across the process) ────────────────────────────────

_default_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Return the process-wide singleton EventBus."""
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
    return _default_bus
