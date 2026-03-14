"""
tw_stock_api.py
Taiwan stock data fetcher using yfinance with graceful fallback.
Appends '.TW' suffix for TWSE and '.TWO' for OTC stocks.
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def normalize_ticker(ticker: str) -> str:
    """Normalize ticker: add .TW suffix if missing."""
    ticker = ticker.strip().upper()
    if "." not in ticker:
        return f"{ticker}.TW"
    return ticker


def fetch_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch Taiwan stock OHLCV data from yfinance.
    Returns DataFrame with columns: Open, High, Low, Close, Adj Close, Volume.
    Falls back to synthetic data if download fails.

    Args:
        ticker: Stock ticker (e.g., '2330' or '2330.TW')
        start_date: Start date string 'YYYY-MM-DD'
        end_date:   End date string 'YYYY-MM-DD'
        cache_dir:  Directory for CSV cache. None = no cache.
    """
    ticker = normalize_ticker(ticker)
    cache_path = None

    # --- Cache check ---
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{ticker}_{start_date}_{end_date}.csv"
        if cache_path.exists():
            logger.info(f"Loading from cache: {cache_path}")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return df

    # --- yfinance download ---
    try:
        import yfinance as yf
        logger.info(f"Downloading {ticker} from {start_date} to {end_date}")
        
        # Download OHLCV
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)

        if df.empty:
            raise ValueError(f"No data returned for {ticker}")

        # Fetch Fundamental info (PE, PB, Market Cap)
        # Note: .info is slower, so we do it as an enhancement
        fundamentals = {}
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            fundamentals = {
                "pe_ratio": info.get("trailingPE"),
                "pb_ratio": info.get("priceToBook"),
                "market_cap": info.get("marketCap"),
                "dividend_yield": info.get("dividendYield"),
                "long_name": info.get("longName")
            }
        except Exception as e:
            logger.warning(f"Could not fetch fundamental info for {ticker}: {e}")

        # Flatten multi-level columns if needed (yfinance >= 0.2.x)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Standardize column names
        df.columns = [c.replace(" ", "_").lower() for c in df.columns]
        df.index.name = "date"
        df = df.sort_index()

        # Drop rows where all OHLCV is NaN
        df = df.dropna(subset=["close"])
        
        # Attach fundamental attributes
        df.attrs["fundamentals"] = fundamentals
        df.attrs["is_synthetic"] = False

        # Cache result
        if cache_path is not None:
            df.to_csv(cache_path)
            # Store fundamentals in a separate hidden file if needed, 
            # but for MVP we just keep them in attrs for the current session.
            logger.info(f"Cached to {cache_path}")

        return df

    except Exception as e:
        logger.warning(f"yfinance download failed for {ticker}: {e}. Using synthetic fallback.")
        return _generate_synthetic_data(ticker, start_date, end_date)


def _generate_synthetic_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generate realistic-looking synthetic OHLCV data for fallback.
    This is ONLY for demo/testing; label it clearly.
    """
    logger.warning("⚠️  Using SYNTHETIC (simulated) data — for testing only!")
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    n = len(dates)

    np.random.seed(42)
    returns = np.random.normal(0.0003, 0.015, n)
    price = 500.0 * np.exp(np.cumsum(returns))  # geometric random walk

    high = price * (1 + np.abs(np.random.normal(0, 0.008, n)))
    low = price * (1 - np.abs(np.random.normal(0, 0.008, n)))
    open_ = price * (1 + np.random.normal(0, 0.006, n))
    volume = np.random.randint(5_000_000, 50_000_000, n).astype(float)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": price,
            "adj_close": price,
            "volume": volume,
        },
        index=dates,
    )
    df.index.name = "date"
    df.attrs["is_synthetic"] = True
    df.attrs["fundamentals"] = {
        "pe_ratio": 15.0,
        "pb_ratio": 2.5,
        "market_cap": 10000000000,
        "dividend_yield": 0.03,
        "long_name": f"Synthetic {ticker}"
    }
    return df


def get_available_tickers() -> list[str]:
    """Return a curated list of popular Taiwan stocks for the UI dropdown."""
    return [
        "2330.TW",  # TSMC
        "2317.TW",  # Hon Hai
        "2454.TW",  # MediaTek
        "2382.TW",  # Quanta
        "3008.TW",  # Largan
        "2308.TW",  # Delta Electronics
        "2881.TW",  # Fubon Financial
        "2882.TW",  # Cathay Financial
        "6505.TW",  # Formosa Petrochemical
        "1301.TW",  # Formosa Plastics
        "2412.TW",  # Chunghwa Telecom
        "2886.TW",  # Mega Financial
        "2891.TW",  # CTBC Financial
        "2357.TW",  # ASUS
        "2303.TW",  # United Micro
    ]
