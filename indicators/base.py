"""
indicators/base.py
------------------
Abstract base class for all technical indicators.
Every indicator must implement calculate() which returns a pd.Series.
This enforces a consistent interface across all 25+ indicator/signal classes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Indicator(ABC):
    """
    Abstract base for all technical indicators.

    Subclasses implement `calculate(df)` to return a normalized pd.Series
    of the same length as the input DataFrame.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this indicator (used as column name)."""
        ...

    @property
    def required_columns(self) -> list[str]:
        """List of DataFrame columns required by this indicator."""
        return ["Open", "High", "Low", "Close", "Volume"]

    def validate(self, df: pd.DataFrame) -> None:
        """Raise ValueError if required columns are missing."""
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"{self.name}: missing required columns: {missing}")

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute the indicator on the given OHLCV DataFrame.

        Args:
            df: DataFrame with at minimum the columns listed in `required_columns`.

        Returns:
            pd.Series aligned with df.index, same length as df.
        """
        ...

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the indicator and attach it as a column to df.
        Returns df with the new column added in-place copy.
        """
        self.validate(df)
        df = df.copy()
        df[self.name] = self.calculate(df)
        return df
