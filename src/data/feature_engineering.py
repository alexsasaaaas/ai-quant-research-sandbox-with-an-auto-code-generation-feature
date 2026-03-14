"""
feature_engineering.py
Technical indicator computation using the `ta` library.
All indicators are strictly backward-looking (rolling on past data only).
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ── Price Features ────────────────────────────────────────────────────────────

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add moving averages and price-based features."""
    df = df.copy()

    for w in [5, 20, 60]:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
        df[f"close_over_sma{w}"] = df["close"] / df[f"sma_{w}"] - 1

    # EMA
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

    # High-Low range
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]

    # Gap (open vs prior close)
    df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    return df


# ── Volume Features ────────────────────────────────────────────────────────────

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume moving averages and relative volume."""
    df = df.copy()

    df["volume_ma5"] = df["volume"].rolling(5).mean()
    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma20"]

    # On-Balance Volume (accumulative)
    sign = np.sign(df["close"].diff())
    df["obv"] = (sign * df["volume"]).fillna(0).cumsum()
    df["obv_ma5"] = df["obv"].rolling(5).mean()

    return df


# ── Momentum / Oscillator Features ────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add MACD, RSI, Bollinger Bands, ATR, Stoch, CCI.
    All rolling windows use only past data.
    """
    df = df.copy()

    # MACD
    macd_line = df["ema_12"] - df["ema_26"]
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = macd_line - signal_line

    # RSI (14)
    df["rsi_14"] = _compute_rsi(df["close"], period=14)

    # Bollinger Bands (20, 2σ)
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # ATR (14)
    df["atr_14"] = _compute_atr(df, period=14)

    # Stochastic Oscillator (14)
    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    df["stoch_k"] = 100 * (df["close"] - low14) / (high14 - low14)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # CCI (20)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    ma_tp = typical.rolling(20).mean()
    mad = typical.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci_20"] = (typical - ma_tp) / (0.015 * mad)

    return df


# ── Helper Functions ───────────────────────────────────────────────────────────

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


# ── Fundamental Features ──────────────────────────────────────────────────────

def add_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fundamental ratios (PE, PB, etc.) if available in df.attrs.
    Since these are often static for daily data, we repeat them for all rows.
    """
    df = df.copy()
    fundamentals = df.attrs.get("fundamentals", {})
    
    # Fill with available data or NaN
    df["pe_ratio"] = fundamentals.get("pe_ratio", np.nan)
    df["pb_ratio"] = fundamentals.get("pb_ratio", np.nan)
    df["market_cap_log"] = np.log10(fundamentals.get("market_cap")) if fundamentals.get("market_cap") else np.nan
    df["dividend_yield"] = fundamentals.get("dividend_yield", np.nan)
    
    return df


# ── Master Feature Builder ─────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    use_price: bool = True,
    use_volume: bool = True,
    use_technical: bool = True,
    use_fundamental: bool = False,
) -> pd.DataFrame:
    """
    Build all requested features on the cleaned OHLCV dataframe.
    Returns DataFrame with new columns appended.
    """
    if use_price:
        df = add_price_features(df)
    if use_volume:
        df = add_volume_features(df)
    if use_technical:
        df = add_technical_indicators(df)
    if use_fundamental:
        df = add_fundamental_features(df)

    logger.info(f"Feature engineering complete. Total columns: {len(df.columns)}")
    return df


def get_feature_columns(
    df: pd.DataFrame,
    use_price: bool = True,
    use_volume: bool = True,
    use_technical: bool = True,
    use_fundamental: bool = False,
) -> list[str]:
    """
    Return the list of feature columns to use for modeling.
    Excludes raw OHLCV, target, and date columns.
    """
    exclude = {
        "open", "high", "low", "close", "adj_close", "volume",
        "target", "date",
    }
    feature_cols = []

    if use_price:
        price_cols = [
            "return_1d", "return_5d", "return_20d", "log_return_1d",
            "sma_5", "sma_20", "sma_60",
            "close_over_sma5", "close_over_sma20", "close_over_sma60",
            "ema_12", "ema_26", "hl_range", "gap",
        ]
        feature_cols += [c for c in price_cols if c in df.columns]

    if use_volume:
        vol_cols = [
            "volume_ma5", "volume_ma20", "volume_ratio", "obv", "obv_ma5",
        ]
        feature_cols += [c for c in vol_cols if c in df.columns]

    if use_technical:
        tech_cols = [
            "macd", "macd_signal", "macd_hist",
            "rsi_14",
            "bb_upper", "bb_lower", "bb_width", "bb_pct",
            "atr_14",
            "stoch_k", "stoch_d",
            "cci_20",
        ]
        feature_cols += [c for c in tech_cols if c in df.columns]

    if use_fundamental:
        fund_cols = ["pe_ratio", "pb_ratio", "market_cap_log", "dividend_yield"]
        feature_cols += [c for c in fund_cols if c in df.columns]

    # Final filter: only columns that exist in df and are not in exclude
    feature_cols = [c for c in feature_cols if c in df.columns and c not in exclude]
    return feature_cols
