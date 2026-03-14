"""
helpers.py
General helper functions for mathematical operations and data manipulation.
"""

import numpy as np
import pandas as pd


def safe_divide(a, b, default=0):
    """Safely divide two numbers, returning default if denominator is zero."""
    try:
        if b == 0:
            return default
        return a / b
    except Exception:
        return default


def calculate_return_metrics(prices: pd.Series):
    """Calculate basic return metrics for a price series."""
    returns = prices.pct_change().dropna()
    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
    annualized_return = (1 + total_return) ** (252 / len(prices)) - 1
    volatility = returns.std() * np.sqrt(252)
    
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe": annualized_return / volatility if volatility > 0 else 0
    }
