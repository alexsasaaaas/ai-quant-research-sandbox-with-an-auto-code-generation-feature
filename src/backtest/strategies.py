"""
strategies.py
Template trading strategies for the Strategy Sandbox.
All strategies are long-only, single-asset, daily frequency.
Signal generation is strictly separated from execution (no look-ahead).
"""

import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 1: Moving Average Cross
# ══════════════════════════════════════════════════════════════════════════════

def ma_cross_signals(
    df: pd.DataFrame,
    fast_window: int = 5,
    slow_window: int = 20,
) -> pd.Series:
    """
    Generate buy/sell signals based on moving average crossover.
    Signal = 1 (long) when fast MA crosses above slow MA.
    Signal = 0 (flat) when fast MA crosses below slow MA.

    Signals are generated on day T and traded on day T+1 (next open).
    """
    fast_ma = df["close"].rolling(fast_window).mean()
    slow_ma = df["close"].rolling(slow_window).mean()

    signal = (fast_ma > slow_ma).astype(int)
    # Shift by 1 so we trade on next-day open (avoids look-ahead)
    return signal.shift(1).fillna(0)


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 2: RSI Mean Reversion
# ══════════════════════════════════════════════════════════════════════════════

def rsi_mean_reversion_signals(
    df: pd.DataFrame,
    rsi_period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> pd.Series:
    """
    Long when RSI crosses above oversold threshold.
    Flat when RSI crosses above overbought threshold.
    """
    # Compute RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    position = pd.Series(0, index=df.index)
    in_position = False

    for i in range(1, len(rsi)):
        r = rsi.iloc[i]
        if pd.isna(r):
            continue
        if not in_position and r < oversold:
            in_position = True
        elif in_position and r > overbought:
            in_position = False
        position.iloc[i] = 1 if in_position else 0

    # Shift by 1: signal generated today, traded tomorrow
    return position.shift(1).fillna(0)


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 3: Prediction-based Strategy
# ══════════════════════════════════════════════════════════════════════════════

def prediction_based_signals(
    df: pd.DataFrame,
    predictions: pd.Series,
    task_type: str = "regression",
    threshold: float = 0.0,
) -> pd.Series:
    """
    Use model predictions as trading signals.
    - Regression: long if predicted return > threshold
    - Classification: long if predicted class == 1

    Args:
        predictions: pd.Series indexed like df, containing model outputs
        threshold: for regression, minimum predicted return to go long
    """
    if task_type == "classification":
        signal = (predictions > 0.5).astype(int)
    else:
        signal = (predictions > threshold).astype(int)

    # Align to df index, fill unknowns as 0
    signal = signal.reindex(df.index).fillna(0)
    # Shift by 1: trade next day
    return signal.shift(1).fillna(0)


# ══════════════════════════════════════════════════════════════════════════════
# Strategy Registry
# ══════════════════════════════════════════════════════════════════════════════

STRATEGY_REGISTRY = {
    "MA Cross":                 ma_cross_signals,
    "RSI Mean Reversion":       rsi_mean_reversion_signals,
    "Prediction-based":         prediction_based_signals,
}
