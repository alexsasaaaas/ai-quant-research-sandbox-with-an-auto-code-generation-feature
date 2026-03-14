"""
preprocess.py
Data cleaning and preprocessing for OHLCV data.
All operations are strictly backward-looking (no leakage).
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Forward-fill small gaps (up to 3 consecutive missing days)
    - Drop remaining NaN rows
    - Remove zero or negative prices
    - Ensure sorted index
    """
    df = df.copy()
    df = df.sort_index()

    # Forward fill up to 3 day gaps (holidays, minor data issues)
    df = df.ffill(limit=3)

    # Drop rows where close is NaN, zero, or negative
    df = df[df["close"].notna() & (df["close"] > 0)]

    # Ensure OHLC consistency: high >= close >= low
    df = df[df["high"] >= df["close"]]
    df = df[df["close"] >= df["low"]]

    logger.info(f"Cleaned OHLCV: {len(df)} rows remaining")
    return df


def compute_basic_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute basic return columns.
    All returns use only past prices (shift-based, no leakage).
    """
    df = df.copy()

    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)
    df["log_return_1d"] = np.log(df["close"] / df["close"].shift(1))

    return df


def build_target(
    df: pd.DataFrame,
    task_type: str,
    horizon: int,
) -> pd.DataFrame:
    """
    Build prediction target column without leakage.

    Args:
        task_type: 'regression' or 'classification'
        horizon:   1 or 5 (days ahead)

    Target:
        regression:    future return over horizon days
        classification: 1 if positive future return, 0 otherwise
    """
    df = df.copy()

    # Future return: price N days from now vs today
    # Using shift(-horizon) introduces future info — handled by dropping NaN rows AFTER building all features
    future_return = df["close"].pct_change(horizon).shift(-horizon)

    if task_type == "regression":
        df["target"] = future_return
    elif task_type == "classification":
        df["target"] = (future_return > 0).astype(int)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    # NOTE: Last `horizon` rows will have NaN target — they must be dropped before training
    return df


def split_time_series(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strict chronological split. No shuffling.
    Returns (train, test).
    """
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    logger.info(f"Split: train={len(train)}, test={len(test)}")
    return train, test
