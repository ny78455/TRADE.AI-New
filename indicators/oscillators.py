"""
indicators/oscillators.py
--------------------------
Vectorized RSI and ADX implementations via pandas_ta.
"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta  # type: ignore

from indicators.base import Indicator


class RSI(Indicator):
    """Relative Strength Index — fully vectorized via pandas_ta."""

    def __init__(self, period: int = 14) -> None:
        self.period = period

    @property
    def name(self) -> str:
        return f"RSI_{self.period}"

    @property
    def required_columns(self) -> list[str]:
        return ["Close"]

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return ta.rsi(df["Close"], length=self.period).bfill()


class ADX(Indicator):
    """Average Directional Index — via pandas_ta."""

    def __init__(self, period: int = 14) -> None:
        self.period = period

    @property
    def name(self) -> str:
        return f"ADX_{self.period}"

    @property
    def required_columns(self) -> list[str]:
        return ["High", "Low", "Close"]

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=self.period)
        col = f"ADX_{self.period}"
        if col in adx_df.columns:
            return adx_df[col].bfill()
        # Fallback to first column if name varies
        return adx_df.iloc[:, 0].bfill()
