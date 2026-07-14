"""
engine/strategy_runner.py
--------------------------
Event-driven strategy runner.
Replaces the `while True: time.sleep(8)` trading loop with a proper
async event loop that responds to market data events the moment they arrive.

Architecture:
  MarketDataFeed → BarEvent → StrategyRunner → on_bar() → Signal
  Signal → RiskManager.check_order() → ExecutionEngine.submit_order()
  ExecutionEngine → FillEvent → PositionManager.on_fill()
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from config.logging import get_logger
from config.settings import get_settings
from core.events import (
    BarEvent, ErrorEvent, FillEvent, OrderEvent,
    ShutdownEvent, SignalEvent, get_event_bus,
)
from core.types import (
    Bar, Fill, Order, OrderSide, Signal, SignalType,
    StrategyState, TimeInForce, Timeframe,
)
from engine.execution import ExecutionEngine
from engine.order_manager import OrderManager
from engine.position_manager import PositionManager
from engine.risk_manager import RiskManager

log = get_logger(__name__)


# ── Abstract Strategy Interface ────────────────────────────────────────────────────

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Subclass this and implement on_bar() to create a new strategy.
    """

    def __init__(self, symbol: str, timeframe: Timeframe = Timeframe.M1) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.state = StrategyState.INITIALIZING
        self._bar_count = 0
        self._warm_up_bars: int = 200  # Override in subclass if needed

    async def initialize(self) -> None:
        """Called once before the strategy begins receiving bars."""
        self.state = StrategyState.WARMING_UP

    @abstractmethod
    async def on_bar(self, bar: Bar) -> Optional[Signal]:
        """
        Called for every new completed bar.
        Return a Signal to trigger order submission, or None to do nothing.
        """
        ...

    async def on_fill(self, fill: Fill) -> None:
        """Called when an order placed by this strategy is filled."""
        pass

    async def shutdown(self) -> None:
        """Called on graceful shutdown — clean up any open state."""
        self.state = StrategyState.SHUTDOWN

    def is_warmed_up(self) -> bool:
        return self._bar_count >= self._warm_up_bars

    def _tick_bar(self) -> None:
        self._bar_count += 1
        if self._bar_count == self._warm_up_bars:
            self.state = StrategyState.ACTIVE
            log.info("strategy.warmed_up", symbol=self.symbol, bars=self._bar_count)


# ── Strategy Runner ────────────────────────────────────────────────────────────────

class StrategyRunner:
    """
    Orchestrates the full trading lifecycle:
      1. Subscribes to BarEvents from the event bus
      2. Runs the strategy's on_bar() for each new candle
      3. Validates signals through RiskManager
      4. Routes approved orders through ExecutionEngine
      5. Updates PositionManager on fills
      6. Publishes all state changes back to the event bus

    This replaces:
      - `while True: time.sleep(8)` in app.py
      - Global mutable `ticker`, `period`, `interval` variables
      - Self-calling HTTP buy_stock()/sell_stock() functions
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        order_manager: Optional[OrderManager] = None,
        position_manager: Optional[PositionManager] = None,
        risk_manager: Optional[RiskManager] = None,
        execution_engine: Optional[ExecutionEngine] = None,
    ) -> None:
        self._strategy = strategy
        self._bus = get_event_bus()
        self._order_mgr = order_manager or OrderManager()
        self._position_mgr = position_manager or PositionManager()
        self._risk_mgr = risk_manager or RiskManager()
        self._execution = execution_engine or ExecutionEngine()
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Initialize the strategy and begin the event processing loop."""
        log.info("runner.starting", strategy=type(self._strategy).__name__)
        await self._strategy.initialize()
        self._running = True

        # Launch concurrent tasks
        self._tasks = [
            asyncio.create_task(self._bar_loop(), name="bar_loop"),
            asyncio.create_task(self._fill_loop(), name="fill_loop"),
        ]

        log.info("runner.started")
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Gracefully stop the runner and the strategy."""
        log.info("runner.stopping")
        self._running = False
        await self._strategy.shutdown()
        for task in self._tasks:
            task.cancel()
        await self._bus.broadcast_shutdown()
        log.info("runner.stopped")

    # ── Event Loops ────────────────────────────────────────────────────────────────

    async def _bar_loop(self) -> None:
        """Consume BarEvents and run the strategy."""
        async for event in self._bus.subscribe(BarEvent):
            if not self._running:
                break
            if event.bar is None or event.bar.symbol != self._strategy.symbol:
                continue

            bar = event.bar
            self._strategy._tick_bar()

            try:
                signal = await self._strategy.on_bar(bar)
            except Exception as e:
                log.error("runner.strategy_error", error=str(e), exc_info=True)
                await self._bus.publish(ErrorEvent(source="strategy", message=str(e), exc=e))
                continue

            if signal is not None and self._strategy.is_warmed_up():
                await self._handle_signal(signal, bar)

    async def _fill_loop(self) -> None:
        """Consume FillEvents and update position manager."""
        async for event in self._bus.subscribe(FillEvent):
            if not self._running:
                break
            if event.fill is None:
                continue
            fill = event.fill
            await self._position_mgr.on_fill(fill)
            await self._strategy.on_fill(fill)
            log.info(
                "runner.fill_processed",
                symbol=fill.symbol,
                side=fill.side.value,
                qty=fill.filled_qty,
                price=fill.filled_price,
            )

    # ── Signal → Order Flow ────────────────────────────────────────────────────────

    async def _handle_signal(self, signal: Signal, bar: Bar) -> None:
        """Convert a signal into an order after risk checks."""
        await self._bus.publish(SignalEvent(signal=signal))

        if signal.signal == SignalType.HOLD:
            return

        # Determine order direction and quantity
        side = OrderSide.BUY if signal.signal.is_buy else OrderSide.SELL
        account_value = await self._execution.get_account_value()
        position = self._position_mgr.get(signal.symbol)

        # Simple position sizing: use 10% of account per trade
        qty = (account_value * 0.10) / bar.close
        qty = round(qty, 6)

        if qty <= 0:
            log.warning("runner.zero_qty", symbol=signal.symbol)
            return

        order = OrderManager.create_market_order(signal.symbol, side, qty)

        # Risk gate
        result = await self._risk_mgr.check_order(
            order=order,
            current_price=bar.close,
            portfolio_value=account_value,
            open_order_count=self._order_mgr.pending_count,
        )

        if not result.approved:
            log.warning(
                "runner.order_rejected_by_risk",
                signal=signal.signal.name,
                reason=result.reason,
                risk_class=result.risk_class,
            )
            return

        # Enqueue and execute
        await self._order_mgr.submit(order)
        await self._bus.publish(OrderEvent(order=order))

        try:
            fill = await self._execution.submit_order(order)
            await self._order_mgr.on_fill(fill)
            await self._bus.publish(FillEvent(fill=fill))
        except Exception as e:
            log.error("runner.order_failed", order_id=order.id, error=str(e))
            await self._order_mgr.on_rejected(order.id, str(e))
            await self._bus.publish(ErrorEvent(source="execution", message=str(e), exc=e))
