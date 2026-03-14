"""
loader.py
Unified data loading interface used by all upper layers.
"""

import logging
from typing import Optional
import pandas as pd

from src.data.tw_stock_api import fetch_stock_data
from src.data.preprocess import clean_ohlcv, compute_basic_returns

logger = logging.getLogger(__name__)


def load_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    cache_dir: Optional[str] = "data/cache",
) -> pd.DataFrame:
    """
    Full pipeline: download → clean → returns.
    Returns a clean DataFrame ready for feature engineering.
    """
    df = fetch_stock_data(ticker, start_date, end_date, cache_dir=cache_dir)
    df = clean_ohlcv(df)
    df = compute_basic_returns(df)
    logger.info(f"Loaded {len(df)} rows for {ticker} [{start_date} ~ {end_date}]")
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return a dict summarizing the loaded dataset."""
    fundamentals = df.attrs.get("fundamentals", {})
    
    return {
        "n_rows": len(df),
        "start_date": str(df.index.min().date()),
        "end_date": str(df.index.max().date()),
        "columns": list(df.columns),
        "missing_pct": float(df.isnull().mean().mean() * 100),
        "close_start": float(df["close"].iloc[0]),
        "close_end": float(df["close"].iloc[-1]),
        "total_return_pct": float(
            (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
        ),
        "fundamentals": fundamentals  # Include fetched info
    }
