"""
engine/position_manager.py
--------------------------
Real-time position tracking with mark-to-market P&L.
Thread-safe (via asyncio.Lock) for concurrent price updates and fill events.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Dict, Iterator, Optional

from config.logging import get_logger
from core.types import AssetClass, Fill, OrderSide, Position

log = get_logger(__name__)


class PositionManager:
    """
    Tracks all open positions and calculates real-time P&L.

    - `on_fill(fill)` — update/open/close position from broker fill
    - `update_price(symbol, price)` — mark-to-market a position
    - `get(symbol)` — retrieve a position
    - `net_exposure` — total market value of all open positions
    """

    def __init__(self) -> None:
        self._positions: Dict[str, Position] = {}
        self._realized_pnl: float = 0.0
        self._lock = asyncio.Lock()

    # ── Updates ────────────────────────────────────────────────────────────────────

    async def on_fill(self, fill: Fill, cost_basis: Optional[float] = None) -> Position | None:
        """
        Process a fill event and update the corresponding position.
        - Buy fill → open or increase position
        - Sell fill → reduce or close position
        Returns the updated (or None if position closed).
        """
        async with self._lock:
            sym = fill.symbol

            if fill.side == OrderSide.BUY:
                return await self._process_buy(fill)
            else:
                return await self._process_sell(fill)

    async def _process_buy(self, fill: Fill) -> Position:
        sym = fill.symbol
        existing = self._positions.get(sym)

        if existing is None:
            # New position
            pos = Position(
                symbol=sym,
                qty=fill.filled_qty,
                avg_entry_price=fill.filled_price,
                current_price=fill.filled_price,
            )
            self._positions[sym] = pos
            log.info(
                "position.opened",
                symbol=sym,
                qty=fill.filled_qty,
                price=fill.filled_price,
            )
        else:
            # Average in
            total_qty = existing.qty + fill.filled_qty
            existing.avg_entry_price = (
                (existing.avg_entry_price * existing.qty + fill.filled_price * fill.filled_qty)
                / total_qty
            )
            existing.qty = total_qty
            existing.update_price(fill.filled_price)
            log.info(
                "position.increased",
                symbol=sym,
                total_qty=existing.qty,
                avg_price=existing.avg_entry_price,
            )
            pos = existing

        return pos

    async def _process_sell(self, fill: Fill) -> Position | None:
        sym = fill.symbol
        existing = self._positions.get(sym)

        if not existing:
            log.warning("position.sell.no_position", symbol=sym)
            return None

        # Calculate realized P&L for the sold portion
        realized = (fill.filled_price - existing.avg_entry_price) * fill.filled_qty
        existing.realized_pnl += realized
        self._realized_pnl += realized
        existing.qty -= fill.filled_qty

        if existing.qty <= 1e-8:
            # Position fully closed
            log.info(
                "position.closed",
                symbol=sym,
                realized_pnl=realized,
                total_realized=self._realized_pnl,
            )
            del self._positions[sym]
            return None
        else:
            existing.update_price(fill.filled_price)
            log.info(
                "position.reduced",
                symbol=sym,
                remaining_qty=existing.qty,
                realized_pnl=realized,
            )
            return existing

    async def update_price(self, symbol: str, price: float) -> None:
        """Mark-to-market: update unrealized P&L for a position."""
        async with self._lock:
            pos = self._positions.get(symbol)
            if pos:
                pos.update_price(price)

    # ── Queries ────────────────────────────────────────────────────────────────────

    def get(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        pos = self._positions.get(symbol)
        return pos is not None and pos.qty > 1e-8

    def __iter__(self) -> Iterator[Position]:
        return iter(self._positions.values())

    @property
    def positions(self) -> list[Position]:
        return list(self._positions.values())

    @property
    def net_exposure(self) -> float:
        """Total market value of all open positions."""
        return sum(p.market_value for p in self._positions.values())

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self._positions.values())

    @property
    def total_realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def total_pnl(self) -> float:
        return self.total_realized_pnl + self.total_unrealized_pnl

    @property
    def open_symbol_count(self) -> int:
        return len(self._positions)

    def summary(self) -> dict:
        return {
            "open_positions": self.open_symbol_count,
            "net_exposure": round(self.net_exposure, 2),
            "unrealized_pnl": round(self.total_unrealized_pnl, 2),
            "realized_pnl": round(self.total_realized_pnl, 2),
            "total_pnl": round(self.total_pnl, 2),
            "positions": [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "avg_entry": p.avg_entry_price,
                    "current_price": p.current_price,
                    "market_value": round(p.market_value, 2),
                    "unrealized_pnl": round(p.unrealized_pnl, 2),
                }
                for p in self._positions.values()
            ],
        }
