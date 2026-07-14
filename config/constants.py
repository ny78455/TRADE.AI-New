"""
config/constants.py
-------------------
All trading constants, valid values, and Binance interval mappings.
Replaces magic strings scattered across pre_pipeline.py and app.py.
"""
from __future__ import annotations

from typing import Final

from binance.client import Client  # type: ignore[import]

# ── Binance Interval Mapping ───────────────────────────────────────────────────
BINANCE_INTERVAL_MAP: Final[dict[str, str]] = {
    "1MINUTE":  Client.KLINE_INTERVAL_1MINUTE,
    "3MINUTE":  Client.KLINE_INTERVAL_3MINUTE,
    "5MINUTE":  Client.KLINE_INTERVAL_5MINUTE,
    "15MINUTE": Client.KLINE_INTERVAL_15MINUTE,
    "30MINUTE": Client.KLINE_INTERVAL_30MINUTE,
    "1HOUR":    Client.KLINE_INTERVAL_1HOUR,
    "2HOUR":    Client.KLINE_INTERVAL_2HOUR,
    "4HOUR":    Client.KLINE_INTERVAL_4HOUR,
    "6HOUR":    Client.KLINE_INTERVAL_6HOUR,
    "8HOUR":    Client.KLINE_INTERVAL_8HOUR,
    "12HOUR":   Client.KLINE_INTERVAL_12HOUR,
    "1DAY":     Client.KLINE_INTERVAL_1DAY,
    "3DAY":     Client.KLINE_INTERVAL_3DAY,
    "1WEEK":    Client.KLINE_INTERVAL_1WEEK,
    "1MONTH":   Client.KLINE_INTERVAL_1MONTH,
}

VALID_INTERVALS: Final[frozenset[str]] = frozenset(BINANCE_INTERVAL_MAP.keys())

# ── Alpaca Time-In-Force ────────────────────────────────────────────────────────
TIF_IOC: Final[str] = "ioc"     # Immediate-or-cancel (crypto)
TIF_DAY: Final[str] = "day"     # Day order (fractional stocks)
TIF_GTC: Final[str] = "gtc"     # Good-till-cancelled (whole-qty stocks)

# ── Order Types ─────────────────────────────────────────────────────────────────
ORDER_TYPE_MARKET: Final[str] = "market"
ORDER_TYPE_LIMIT:  Final[str] = "limit"
VALID_ORDER_TYPES: Final[frozenset[str]] = frozenset({ORDER_TYPE_MARKET, ORDER_TYPE_LIMIT})

# ── Valid Market Types (for routing) ────────────────────────────────────────────
VALID_MARKET_TYPES: Final[frozenset[str]] = frozenset({
    "us_stocks", "world_stocks", "etfs", "crypto", "forex", "futures", "bonds"
})

# ── Technical Indicator Defaults ─────────────────────────────────────────────────
DEFAULT_EMA_PERIOD:         Final[int] = 4
DEFAULT_RSI_PERIOD:         Final[int] = 14
DEFAULT_BOLLINGER_WINDOW:   Final[int] = 20
DEFAULT_BOLLINGER_STD:      Final[int] = 2
DEFAULT_BACKCANDLES:        Final[int] = 15
DEFAULT_PIVOT_WINDOW:       Final[int] = 5

# ── Backtesting ──────────────────────────────────────────────────────────────────
TRADING_DAYS_PER_YEAR:  Final[int] = 252
DEFAULT_INITIAL_CASH:   Final[float] = 10_000.0
DEFAULT_SLIPPAGE_BPS:   Final[float] = 2.0    # 2 basis points
DEFAULT_COMMISSION_BPS: Final[float] = 5.0    # 5 basis points

# ── Artifact Path Keys ───────────────────────────────────────────────────────────
ARTIFACT_DF_CLEANED:          Final[str] = "df_cleaned.csv"
ARTIFACT_DF_NEW:              Final[str] = "df_new.csv"
ARTIFACT_PREDICTIONS:         Final[str] = "predictions_new.csv"
ARTIFACT_TRAIN:               Final[str] = "train.csv"
ARTIFACT_TEST:                Final[str] = "test.csv"
ARTIFACT_DATA:                Final[str] = "data.csv"
ARTIFACT_PREDICTED_PROB:      Final[str] = "predicted_probablity.csv"
ARTIFACT_MODEL:               Final[str] = "model.h5"

# ── Timezone ─────────────────────────────────────────────────────────────────────
TIMEZONE_UTC:      Final[str] = "UTC"
TIMEZONE_NEW_YORK: Final[str] = "America/New_York"
