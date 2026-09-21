"""
Chronological Time-Series Data Splitting
Polar Navigator AI - MoES / NCPOR

Enforces chronological time-based splitting for sea-ice time-series data:
Training:   Earlier dates
Validation: Intermediate dates
Testing:    Latest historical period

Strictly prevents future lookahead leakage.
"""

from datetime import datetime
import pandas as pd
import numpy as np


class ChronologicalTimeSplitter:
    """Splits time-series cryospheric observations chronologically."""

    @staticmethod
    def split_by_date(
        df: pd.DataFrame,
        date_column: str = "date",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits DataFrame chronologically into (train_df, val_df, test_df).
        """
        if date_column in df.columns:
            df_sorted = df.sort_values(by=date_column).reset_index(drop=True)
        else:
            # If no explicit date column, preserve row sequence
            df_sorted = df.copy().reset_index(drop=True)

        n = len(df_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df_sorted.iloc[:train_end]
        val_df = df_sorted.iloc[train_end:val_end]
        test_df = df_sorted.iloc[val_end:]

        return train_df, val_df, test_df
