"""
core/types.py
-------------
Shared domain types: enumerations, dataclasses, and TypedDicts.
Using Python dataclasses (not Pydantic) for zero-overhead hot-path objects.
Pydantic models live in api/schemas/ for HTTP boundary validation.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional


# ── Enumerations ──────────────────────────────────────────────────────────────────

class OrderSide(str, Enum):
    BUY  = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT  = "limit"
    STOP   = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    PENDING    = "pending"
    SUBMITTED  = "submitted"
    PARTIAL    = "partial_fill"
    FILLED     = "filled"
    CANCELLED  = "cancelled"
    REJECTED   = "rejected"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class Timeframe(str, Enum):
    M1  = "1MINUTE"
    M3  = "3MINUTE"
    M5  = "5MINUTE"
    M15 = "15MINUTE"
    M30 = "30MINUTE"
    H1  = "1HOUR"
    H2  = "2HOUR"
    H4  = "4HOUR"
    H6  = "6HOUR"
    H8  = "8HOUR"
    H12 = "12HOUR"
    D1  = "1DAY"
    D3  = "3DAY"
    W1  = "1WEEK"
    MN  = "1MONTH"


class SignalType(int, Enum):
    """
    Prediction signal values produced by the ensemble model.
    0 = Hold / No signal
    1 = Sell / Short
    2 = Buy (hold 2 candles)
    3 = Buy (hold 3 candles)
    4 = Buy (hold 4 candles)
    5 = Buy (hold 5 candles)
    """
    HOLD  = 0
    SELL  = 1
    BUY_2 = 2
    BUY_3 = 3
    BUY_4 = 4
    BUY_5 = 5

    @property
    def is_buy(self) -> bool:
        return self.value >= 2

    @property
    def hold_periods(self) -> int:
        """Number of candles to hold this position."""
        return self.value if self.is_buy else 0


class AssetClass(str, Enum):
    CRYPTO  = "crypto"
    STOCK   = "stock"
    ETF     = "etf"
    FOREX   = "forex"
    FUTURES = "futures"
    BOND    = "bond"


class StrategyState(str, Enum):
    INITIALIZING = "initializing"
    WARMING_UP   = "warming_up"
    ACTIVE       = "active"
    PAUSED       = "paused"
    SHUTDOWN     = "shutdown"
    ERROR        = "error"


# ── Core Data Structures ───────────────────────────────────────────────────────────

@dataclass(slots=True)
class Bar:
    """A single OHLCV bar (candle)."""
    symbol:    str
    timestamp: datetime
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float
    timeframe: Timeframe = Timeframe.M1


@dataclass(slots=True)
class Order:
    """Represents a single trading order with full lifecycle tracking."""
    symbol:       str
    side:         OrderSide
    qty:          float
    order_type:   OrderType
    time_in_force: TimeInForce

    # Auto-generated
    id:         str         = field(default_factory=lambda: str(uuid.uuid4()))
    status:     OrderStatus = field(default=OrderStatus.PENDING)

    # Optional fields
    limit_price:  Optional[float]    = None
    stop_price:   Optional[float]    = None
    filled_qty:   float              = 0.0
    filled_price: Optional[float]    = None
    broker_id:    Optional[str]      = None
    error:        Optional[str]      = None

    # Timestamps
    created_at:   datetime           = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    filled_at:    Optional[datetime] = None

    @property
    def is_terminal(self) -> bool:
        """True if the order has reached a final state."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        )

    @property
    def unfilled_qty(self) -> float:
        return max(0.0, self.qty - self.filled_qty)


@dataclass(slots=True)
class Position:
    """Represents an open position in a single symbol."""
    symbol:           str
    qty:              float
    avg_entry_price:  float
    asset_class:      AssetClass      = AssetClass.CRYPTO

    # Marked-to-market
    current_price:    float           = 0.0
    unrealized_pnl:   float           = 0.0
    realized_pnl:     float           = 0.0

    opened_at:        datetime        = field(default_factory=datetime.utcnow)

    @property
    def market_value(self) -> float:
        return self.qty * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.qty * self.avg_entry_price

    def update_price(self, price: float) -> None:
        self.current_price = price
        self.unrealized_pnl = (price - self.avg_entry_price) * self.qty


@dataclass(slots=True)
class Signal:
    """A trading signal produced by a strategy."""
    symbol:     str
    signal:     SignalType
    timestamp:  datetime
    confidence: float       = 1.0
    source:     str         = "ensemble"
    metadata:   dict        = field(default_factory=dict)


@dataclass(slots=True)
class Fill:
    """Represents a broker fill event confirming an order execution."""
    order_id:    str
    symbol:      str
    side:        OrderSide
    filled_qty:  float
    filled_price: float
    timestamp:   datetime   = field(default_factory=datetime.utcnow)
    commission:  float      = 0.0
