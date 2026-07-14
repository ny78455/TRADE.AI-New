"""
core/exceptions.py
------------------
Domain exception hierarchy for TRADE.AI.
Replaces the generic src/exception.py with typed, structured exceptions
that carry enough context for proper logging and error handling.
"""
from __future__ import annotations


# ── Base ─────────────────────────────────────────────────────────────────────────

class TradeAIError(Exception):
    """Root exception for all TRADE.AI domain errors."""

    def __init__(self, message: str, context: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict = context or {}

    def __repr__(self) -> str:
        parts = [f"{self.__class__.__name__}({self.message!r}"]
        if self.context:
            parts.append(f", context={self.context!r}")
        return "".join(parts) + ")"


# ── Data Layer ────────────────────────────────────────────────────────────────────

class DataError(TradeAIError):
    """Raised when data fetching, parsing, or validation fails."""


class DataFetchError(DataError):
    """Raised when a market data API call fails."""

    def __init__(self, symbol: str, source: str, reason: str) -> None:
        super().__init__(
            f"Failed to fetch data for {symbol!r} from {source!r}: {reason}",
            context={"symbol": symbol, "source": source, "reason": reason},
        )


class DataValidationError(DataError):
    """Raised when incoming data fails schema or sanity validation."""


class InsufficientDataError(DataError):
    """Raised when there are not enough bars to compute an indicator."""

    def __init__(self, required: int, available: int, indicator: str = "") -> None:
        label = f" for {indicator!r}" if indicator else ""
        super().__init__(
            f"Insufficient data{label}: need {required} bars, got {available}",
            context={"required": required, "available": available, "indicator": indicator},
        )


# ── Order / Execution ─────────────────────────────────────────────────────────────

class OrderError(TradeAIError):
    """Base for all order-related errors."""


class OrderSubmissionError(OrderError):
    """Raised when order submission to the broker fails."""

    def __init__(self, symbol: str, side: str, reason: str) -> None:
        super().__init__(
            f"Order submission failed: {side} {symbol!r} — {reason}",
            context={"symbol": symbol, "side": side, "reason": reason},
        )


class OrderNotFoundError(OrderError):
    """Raised when an order ID cannot be found in the order book."""

    def __init__(self, order_id: str) -> None:
        super().__init__(
            f"Order not found: {order_id!r}",
            context={"order_id": order_id},
        )


class OrderCancellationError(OrderError):
    """Raised when order cancellation fails."""


class InsufficientFundsError(OrderError):
    """Raised when there is not enough capital to submit an order."""

    def __init__(self, required: float, available: float, symbol: str = "") -> None:
        super().__init__(
            f"Insufficient funds for {symbol!r}: need ${required:.2f}, have ${available:.2f}",
            context={"required": required, "available": available, "symbol": symbol},
        )


# ── Risk ──────────────────────────────────────────────────────────────────────────

class RiskError(TradeAIError):
    """Base for all risk management violations."""


class PositionLimitExceeded(RiskError):
    """Raised when an order would exceed the max position size."""

    def __init__(self, symbol: str, requested: float, limit: float) -> None:
        super().__init__(
            f"Position limit exceeded for {symbol!r}: "
            f"requested ${requested:.2f} > limit ${limit:.2f}",
            context={"symbol": symbol, "requested": requested, "limit": limit},
        )


class DailyLossLimitExceeded(RiskError):
    """Raised when the circuit breaker trips on daily P&L."""

    def __init__(self, current_loss_pct: float, limit_pct: float) -> None:
        super().__init__(
            f"Daily loss limit exceeded: {current_loss_pct:.2f}% > {limit_pct:.2f}% limit",
            context={"current_loss_pct": current_loss_pct, "limit_pct": limit_pct},
        )


class OrderRateExceeded(RiskError):
    """Raised when too many orders are submitted in a short window."""

    def __init__(self, symbol: str, rate: int, limit: int) -> None:
        super().__init__(
            f"Order rate exceeded for {symbol!r}: {rate}/s > {limit}/s",
            context={"symbol": symbol, "rate": rate, "limit": limit},
        )


class KillSwitchActivated(RiskError):
    """Raised when the emergency kill switch is triggered."""


# ── ML / Model ────────────────────────────────────────────────────────────────────

class ModelError(TradeAIError):
    """Base for ML model errors."""


class ModelNotLoadedError(ModelError):
    """Raised when a prediction is attempted before the model is loaded."""


class ModelInferenceError(ModelError):
    """Raised when model inference fails."""


class ModelVersionError(ModelError):
    """Raised when there is a model version/shape mismatch."""


# ── Configuration ─────────────────────────────────────────────────────────────────

class ConfigurationError(TradeAIError):
    """Raised when required configuration is missing or invalid."""
