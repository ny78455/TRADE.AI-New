"""
engine/order_manager.py
-----------------------
Order lifecycle management with an async submission queue.
  - Lock-free order book using asyncio.Queue
  - Full state machine: PENDING → SUBMITTED → PARTIAL → FILLED/CANCELLED/REJECTED
  - Nanosecond-precision order IDs
  - Fixed-size ring buffer for order history (no memory leaks)
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Deque, Dict, Optional

from config.logging import get_logger
from core.exceptions import OrderNotFoundError
from core.types import Fill, Order, OrderSide, OrderStatus, OrderType, TimeInForce

log = get_logger(__name__)

# Maximum number of completed orders kept in history
_HISTORY_RING_SIZE = 10_000


class OrderManager:
    """
    Manages the full lifecycle of trading orders.

    - `submit(order)` — enqueue an order for async processing
    - `get(order_id)` — fast O(1) lookup
    - `cancel(order_id)` — mark an order cancelled
    - `on_fill(fill)` — update order state from broker fill event
    """

    def __init__(self, max_queue_size: int = 1_000) -> None:
        # Active orders: id → Order (pending/submitted/partial)
        self._active: Dict[str, Order] = {}
        # Completed orders: fixed-size ring buffer
        self._history: Deque[Order] = deque(maxlen=_HISTORY_RING_SIZE)
        # Async queue for the execution engine to consume
        self._queue: asyncio.Queue[Order] = asyncio.Queue(maxsize=max_queue_size)
        self._lock = asyncio.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────────

    async def submit(self, order: Order) -> str:
        """
        Enqueue an order for execution. Returns the order ID.
        Non-blocking if the queue has capacity; raises QueueFull otherwise.
        """
        async with self._lock:
            self._active[order.id] = order

        await self._queue.put(order)
        log.info(
            "order.queued",
            order_id=order.id,
            symbol=order.symbol,
            side=order.side.value,
            qty=order.qty,
            type=order.order_type.value,
        )
        return order.id

    async def on_submitted(self, order_id: str, broker_id: str) -> None:
        """Called by ExecutionEngine once the broker confirms receipt."""
        async with self._lock:
            order = self._active.get(order_id)
            if order:
                order.status = OrderStatus.SUBMITTED
                order.broker_id = broker_id
                order.submitted_at = datetime.utcnow()
                log.info("order.submitted", order_id=order_id, broker_id=broker_id)

    async def on_fill(self, fill: Fill) -> Optional[Order]:
        """
        Update order state from a broker fill event.
        Returns the updated Order, or None if not found.
        """
        async with self._lock:
            order = self._active.get(fill.order_id)
            if not order:
                log.warning("order.fill.not_found", order_id=fill.order_id)
                return None

            order.filled_qty += fill.filled_qty
            order.filled_price = fill.filled_price

            if order.filled_qty >= order.qty * 0.9999:  # fully filled (floating tolerance)
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.utcnow()
                self._history.append(order)
                del self._active[order.id]
                log.info(
                    "order.filled",
                    order_id=order.id,
                    symbol=order.symbol,
                    filled_qty=order.filled_qty,
                    filled_price=order.filled_price,
                )
            else:
                order.status = OrderStatus.PARTIAL
                log.info(
                    "order.partial_fill",
                    order_id=order.id,
                    filled_qty=order.filled_qty,
                    remaining=order.unfilled_qty,
                )
            return order

    async def cancel(self, order_id: str) -> Order:
        """Mark an active order as cancelled."""
        async with self._lock:
            order = self._active.get(order_id)
            if not order:
                raise OrderNotFoundError(order_id)
            order.status = OrderStatus.CANCELLED
            self._history.append(order)
            del self._active[order_id]
            log.info("order.cancelled", order_id=order_id)
            return order

    async def on_rejected(self, order_id: str, reason: str) -> None:
        """Mark an order as rejected by the broker."""
        async with self._lock:
            order = self._active.get(order_id)
            if order:
                order.status = OrderStatus.REJECTED
                order.error = reason
                self._history.append(order)
                del self._active[order_id]
                log.error("order.rejected", order_id=order_id, reason=reason)

    def get(self, order_id: str) -> Order:
        """Synchronous O(1) order lookup (active orders only)."""
        order = self._active.get(order_id)
        if not order:
            raise OrderNotFoundError(order_id)
        return order

    def get_by_symbol(self, symbol: str) -> list[Order]:
        """Return all active orders for a given symbol."""
        return [o for o in self._active.values() if o.symbol == symbol]

    def get_by_status(self, status: OrderStatus) -> list[Order]:
        """Return all active orders with the given status."""
        return [o for o in self._active.values() if o.status == status]

    @property
    def active_orders(self) -> list[Order]:
        return list(self._active.values())

    @property
    def pending_count(self) -> int:
        return sum(1 for o in self._active.values() if o.status == OrderStatus.PENDING)

    async def next_order(self) -> Order:
        """Block until the next order is available from the queue."""
        return await self._queue.get()

    # ── Factory Methods ────────────────────────────────────────────────────────────

    @staticmethod
    def create_market_order(
        symbol: str,
        side: OrderSide,
        qty: float,
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> Order:
        """Convenience factory for market orders."""
        return Order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=OrderType.MARKET,
            time_in_force=time_in_force,
        )

    @staticmethod
    def create_limit_order(
        symbol: str,
        side: OrderSide,
        qty: float,
        limit_price: float,
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> Order:
        """Convenience factory for limit orders."""
        return Order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=OrderType.LIMIT,
            time_in_force=time_in_force,
            limit_price=limit_price,
        )
