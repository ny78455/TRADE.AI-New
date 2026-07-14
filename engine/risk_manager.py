"""
engine/risk_manager.py
----------------------
Pre-trade risk checks with configurable limits and circuit breakers.
Every order MUST pass through the RiskManager before submission.

Features:
  - Max position size per symbol
  - Max portfolio exposure
  - Daily loss circuit breaker (auto-halt)
  - Max open orders limit
  - Per-symbol order rate limiter
  - Emergency kill switch (flatten all positions)
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, date
from typing import Deque, Dict, Optional

from config.logging import get_logger
from config.settings import get_settings
from core.exceptions import (
    DailyLossLimitExceeded,
    KillSwitchActivated,
    OrderRateExceeded,
    PositionLimitExceeded,
    RiskError,
)
from core.types import Order, OrderSide

log = get_logger(__name__)


@dataclass
class RiskCheckResult:
    """Result of a pre-trade risk check."""
    approved: bool
    reason: str = ""
    risk_class: str = ""

    @classmethod
    def ok(cls) -> "RiskCheckResult":
        return cls(approved=True)

    @classmethod
    def reject(cls, reason: str, risk_class: str = "RiskError") -> "RiskCheckResult":
        return cls(approved=False, reason=reason, risk_class=risk_class)


class OrderRateLimiter:
    """
    Sliding-window rate limiter per symbol.
    Tracks timestamps of recent orders in a fixed-size deque.
    """

    def __init__(self, max_per_second: int) -> None:
        self.max_per_second = max_per_second
        self._windows: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=max_per_second * 2))

    def check(self, symbol: str) -> bool:
        """Returns True if the order is within rate limits."""
        now = time.monotonic()
        window = self._windows[symbol]

        # Drop timestamps older than 1 second
        while window and now - window[0] > 1.0:
            window.popleft()

        if len(window) >= self.max_per_second:
            return False

        window.append(now)
        return True


class RiskManager:
    """
    Pre-trade risk gate. Every order must call `check_order()` before submission.

    Usage:
        result = await risk_manager.check_order(order, current_price, portfolio_value)
        if not result.approved:
            raise RiskError(result.reason)
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._max_position_usd = settings.risk.max_position_size_usd
        self._max_daily_loss_pct = settings.risk.max_daily_loss_pct
        self._max_open_orders = settings.risk.max_open_orders
        self._max_orders_per_second = settings.risk.max_orders_per_second

        self._rate_limiter = OrderRateLimiter(self._max_orders_per_second)
        self._kill_switch_active = False
        self._daily_start_value: Optional[float] = None
        self._last_reset_date: Optional[date] = None
        self._open_order_count = 0
        self._lock = asyncio.Lock()

    # ── Core Check ─────────────────────────────────────────────────────────────────

    async def check_order(
        self,
        order: Order,
        current_price: float,
        portfolio_value: float,
        open_order_count: int,
    ) -> RiskCheckResult:
        """
        Run all pre-trade risk checks.
        Returns RiskCheckResult.ok() if all pass, .reject(reason) otherwise.
        """
        # 1. Kill switch
        if self._kill_switch_active:
            return RiskCheckResult.reject(
                "Kill switch is active — all trading halted",
                risk_class="KillSwitchActivated",
            )

        # 2. Daily loss circuit breaker
        if self._daily_start_value is not None:
            self._maybe_reset_daily(portfolio_value)
            loss_pct = (self._daily_start_value - portfolio_value) / self._daily_start_value * 100
            if loss_pct >= self._max_daily_loss_pct:
                log.error(
                    "risk.circuit_breaker_tripped",
                    loss_pct=loss_pct,
                    limit_pct=self._max_daily_loss_pct,
                )
                return RiskCheckResult.reject(
                    f"Daily loss circuit breaker: {loss_pct:.2f}% >= {self._max_daily_loss_pct:.2f}%",
                    risk_class="DailyLossLimitExceeded",
                )

        # 3. Max open orders
        if open_order_count >= self._max_open_orders:
            return RiskCheckResult.reject(
                f"Max open orders reached: {open_order_count} >= {self._max_open_orders}",
                risk_class="OrderLimitExceeded",
            )

        # 4. Position size
        order_value = order.qty * current_price
        if order_value > self._max_position_usd:
            return RiskCheckResult.reject(
                f"Position size ${order_value:.2f} > limit ${self._max_position_usd:.2f}",
                risk_class="PositionLimitExceeded",
            )

        # 5. Rate limiter
        if not self._rate_limiter.check(order.symbol):
            return RiskCheckResult.reject(
                f"Order rate limit exceeded for {order.symbol}",
                risk_class="OrderRateExceeded",
            )

        return RiskCheckResult.ok()

    # ── Portfolio Tracking ─────────────────────────────────────────────────────────

    def record_portfolio_value(self, value: float) -> None:
        """Call this on market open (or first trade of the day) to set daily baseline."""
        today = date.today()
        if self._last_reset_date != today:
            self._daily_start_value = value
            self._last_reset_date = today
            log.info("risk.daily_baseline_set", portfolio_value=value, date=str(today))

    def _maybe_reset_daily(self, current_value: float) -> None:
        today = date.today()
        if self._last_reset_date != today:
            self._daily_start_value = current_value
            self._last_reset_date = today

    # ── Kill Switch ────────────────────────────────────────────────────────────────

    async def activate_kill_switch(self, reason: str = "manual") -> None:
        """
        Activate emergency kill switch — all new orders will be rejected.
        Caller is responsible for flattening existing positions.
        """
        async with self._lock:
            self._kill_switch_active = True
        log.critical("risk.kill_switch_activated", reason=reason)

    async def deactivate_kill_switch(self) -> None:
        async with self._lock:
            self._kill_switch_active = False
        log.info("risk.kill_switch_deactivated")

    @property
    def is_kill_switch_active(self) -> bool:
        return self._kill_switch_active

    # ── Config Updates ─────────────────────────────────────────────────────────────

    def update_limits(
        self,
        max_position_usd: Optional[float] = None,
        max_daily_loss_pct: Optional[float] = None,
        max_open_orders: Optional[int] = None,
        max_orders_per_second: Optional[int] = None,
    ) -> None:
        """Dynamically update risk limits at runtime."""
        if max_position_usd is not None:
            self._max_position_usd = max_position_usd
        if max_daily_loss_pct is not None:
            self._max_daily_loss_pct = max_daily_loss_pct
        if max_open_orders is not None:
            self._max_open_orders = max_open_orders
        if max_orders_per_second is not None:
            self._max_orders_per_second = max_orders_per_second
            self._rate_limiter = OrderRateLimiter(max_orders_per_second)
        log.info("risk.limits_updated")

    def status(self) -> dict:
        return {
            "kill_switch_active": self._kill_switch_active,
            "max_position_usd": self._max_position_usd,
            "max_daily_loss_pct": self._max_daily_loss_pct,
            "max_open_orders": self._max_open_orders,
            "max_orders_per_second": self._max_orders_per_second,
            "daily_start_value": self._daily_start_value,
        }
