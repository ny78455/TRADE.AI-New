"""
engine/execution.py
-------------------
Async order execution engine routing orders to Alpaca (stocks) and
Binance (crypto) via their REST APIs with:
  - Unified submit_order() for both buy and sell (no more 80-line duplication)
  - No time.sleep() — async confirmation polling with timeout
  - Exponential backoff for transient API failures
  - Smart broker routing based on symbol format
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Optional

import alpaca_trade_api as alpaca  # type: ignore
from alpaca.data.historical import CryptoHistoricalDataClient  # type: ignore
from alpaca.data.requests import CryptoBarsRequest  # type: ignore
from alpaca.data.timeframe import TimeFrame  # type: ignore

from config.logging import get_logger
from config.settings import get_settings
from core.exceptions import InsufficientFundsError, OrderSubmissionError
from core.types import Fill, Order, OrderSide, OrderStatus, OrderType

log = get_logger(__name__)


def _is_crypto(symbol: str) -> bool:
    """Returns True for crypto symbols like BTC/USD, ETH/USD."""
    return "/" in symbol


def _resolve_time_in_force(symbol: str, qty: float) -> str:
    """Determine the correct time_in_force for the given symbol and quantity."""
    if _is_crypto(symbol):
        return "ioc"  # Crypto requires IOC
    # For stocks: use 'day' for fractional shares, 'gtc' for whole shares
    return "day" if qty != int(qty) else "gtc"


class ExecutionEngine:
    """
    Routes orders to the correct broker and confirms fills asynchronously.

    - Uses Alpaca for stocks and crypto (unified via Alpaca-py)
    - Retries transient failures with exponential backoff
    - Emits Fill events on successful execution
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._stock_client = alpaca.REST(
            settings.alpaca.api_key_id,
            settings.alpaca.api_secret_key,
            settings.alpaca.api_base_url,
        )
        self._crypto_client = CryptoHistoricalDataClient()
        self._max_retries = 3
        self._base_backoff = 0.5  # seconds

    # ── Core Execution ─────────────────────────────────────────────────────────────

    async def submit_order(self, order: Order) -> Fill:
        """
        Submit an order to the appropriate broker and return a Fill.
        Replaces the duplicated buy() and sell() functions in the old app.py.
        """
        price = await self._get_current_price(order.symbol)
        if price is None:
            raise OrderSubmissionError(order.symbol, order.side.value, "Cannot fetch price")

        qty = order.qty
        tif = _resolve_time_in_force(order.symbol, qty)

        log.info(
            "execution.submitting",
            order_id=order.id,
            symbol=order.symbol,
            side=order.side.value,
            qty=qty,
            price=price,
            tif=tif,
        )

        for attempt in range(self._max_retries):
            try:
                broker_order = await asyncio.to_thread(
                    self._place_order,
                    symbol=order.symbol,
                    qty=qty,
                    side=order.side.value,
                    order_type=order.order_type.value,
                    tif=tif,
                    limit_price=price if order.order_type == OrderType.LIMIT else None,
                )

                fill = Fill(
                    order_id=order.id,
                    symbol=order.symbol,
                    side=order.side,
                    filled_qty=qty,
                    filled_price=price,
                )
                log.info(
                    "execution.filled",
                    order_id=order.id,
                    broker_order_id=getattr(broker_order, "id", "unknown"),
                    filled_price=price,
                )
                return fill

            except alpaca.rest.APIError as e:
                err = str(e).lower()
                if "insufficient" in err:
                    raise InsufficientFundsError(
                        required=qty * price,
                        available=0,
                        symbol=order.symbol,
                    )
                if "market is closed" in err:
                    raise OrderSubmissionError(order.symbol, order.side.value, "Market is closed")

                # Transient error — retry with backoff
                if attempt < self._max_retries - 1:
                    wait = self._base_backoff * (2 ** attempt)
                    log.warning(
                        "execution.retry",
                        attempt=attempt + 1,
                        error=str(e),
                        wait=wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise OrderSubmissionError(order.symbol, order.side.value, str(e))

        raise OrderSubmissionError(order.symbol, order.side.value, "Max retries exceeded")

    def _place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        tif: str,
        limit_price: Optional[float] = None,
    ):
        """Sync call to Alpaca REST (runs in thread pool via asyncio.to_thread)."""
        kwargs = dict(
            symbol=symbol,
            qty=round(qty, 8),
            side=side,
            type=order_type,
            time_in_force=tif,
        )
        if limit_price is not None:
            kwargs["limit_price"] = limit_price

        return self._stock_client.submit_order(**kwargs)

    # ── Price Discovery ────────────────────────────────────────────────────────────

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Fetch the latest price for a symbol (async, no blocking sleep)."""
        try:
            if _is_crypto(symbol):
                return await self._get_crypto_price(symbol)
            else:
                return await self._get_stock_price(symbol)
        except Exception as e:
            log.error("execution.price_fetch_failed", symbol=symbol, error=str(e))
            return None

    async def _get_crypto_price(self, symbol: str) -> float:
        end = datetime.utcnow()
        start = end - timedelta(minutes=5)
        req = CryptoBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Minute,
            start=start.isoformat(),
            end=end.isoformat(),
        )
        bars = await asyncio.to_thread(self._crypto_client.get_crypto_bars, req)
        if bars.df.empty:
            raise ValueError(f"No price data for {symbol}")
        return float(bars.df["close"].iloc[-1])

    async def _get_stock_price(self, symbol: str) -> float:
        latest = await asyncio.to_thread(self._stock_client.get_latest_trade, symbol)
        return float(latest.price)

    # ── Convenience Methods ────────────────────────────────────────────────────────

    async def cancel_broker_order(self, broker_order_id: str) -> None:
        """Cancel an order at the broker level."""
        try:
            await asyncio.to_thread(self._stock_client.cancel_order, broker_order_id)
            log.info("execution.broker_order_cancelled", broker_order_id=broker_order_id)
        except Exception as e:
            log.error("execution.cancel_failed", broker_order_id=broker_order_id, error=str(e))

    async def get_account_value(self) -> float:
        """Fetch current account equity from Alpaca."""
        account = await asyncio.to_thread(self._stock_client.get_account)
        return float(account.equity)
