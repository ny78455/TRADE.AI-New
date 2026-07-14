"""
indicators/bands.py
-------------------
Bollinger Bands — vectorized via the `ta` library.
Replaces the duplicate BollingerIndicators and SignalGenerator1 classes in pre_pipeline.py.
"""
from __future__ import annotations

import pandas as pd
from ta.volatility import BollingerBands as _BB  # type: ignore

from indicators.base import Indicator


class BollingerBandsIndicator(Indicator):
    """
    Computes Bollinger Bands and attaches upper/middle/lower band columns.
    Returns the %B position (0=at lower band, 1=at upper band) as the primary series.
    """

    def __init__(self, window: int = 20, window_dev: float = 2.0) -> None:
        self.window = window
        self.window_dev = window_dev

    @property
    def name(self) -> str:
        return "BB_pct_b"

    @property
    def required_columns(self) -> list[str]:
        return ["Close"]

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        bb = _BB(close=df["Close"], window=self.window, window_dev=self.window_dev)
        pct_b = (df["Close"] - bb.bollinger_lband()) / (
            bb.bollinger_hband() - bb.bollinger_lband() + 1e-10
        )
        return pct_b.bfill().fillna(0.5)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extends apply() to add all 4 band columns + %B."""
        self.validate(df)
        df = df.copy()
        bb = _BB(close=df["Close"], window=self.window, window_dev=self.window_dev)
        df["BB_upper"]  = bb.bollinger_hband().bfill()
        df["BB_middle"] = bb.bollinger_mavg().bfill()
        df["BB_lower"]  = bb.bollinger_lband().bfill()
        df["BB_pct_b"]  = self.calculate(df)
        return df
