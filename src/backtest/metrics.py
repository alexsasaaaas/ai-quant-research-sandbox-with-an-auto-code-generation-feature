"""
metrics.py
Backtest performance metrics calculation.
"""

import numpy as np
import pandas as pd


def compute_backtest_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    initial_capital: float,
    n_trades: int,
    n_wins: int,
) -> dict:
    """
    Compute standard backtest performance statistics.

    Args:
        equity_curve: portfolio value over time
        returns:      daily portfolio returns (decimal)
        initial_capital: starting portfolio value
        n_trades:     number of completed trades
        n_wins:       trades that ended in profit
    """
    total_return = (equity_curve.iloc[-1] / initial_capital - 1)
    n_days = len(returns)
    trading_days_per_year = 252

    # Annualized return (CAGR approximation)
    years = n_days / trading_days_per_year
    ann_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1 if total_return > -1 else -1.0

    # Sharpe ratio (annualized, risk-free rate = 0 for simplicity)
    daily_std = returns.std()
    sharpe = (returns.mean() / daily_std * np.sqrt(trading_days_per_year)
              if daily_std > 1e-10 else 0.0)

    # Maximum drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Calmar ratio
    calmar = ann_return / abs(max_drawdown) if abs(max_drawdown) > 1e-10 else 0.0

    # Win rate
    win_rate = n_wins / n_trades if n_trades > 0 else 0.0

    return {
        "total_return": float(total_return),
        "total_return_pct": float(total_return * 100),
        "annualized_return": float(ann_return),
        "annualized_return_pct": float(ann_return * 100),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "max_drawdown_pct": float(max_drawdown * 100),
        "calmar_ratio": float(calmar),
        "n_trades": int(n_trades),
        "win_rate": float(win_rate),
        "win_rate_pct": float(win_rate * 100),
        "final_equity": float(equity_curve.iloc[-1]),
    }


def compute_buy_and_hold(
    df: pd.DataFrame,
    initial_capital: float,
) -> dict:
    """Compute buy-and-hold baseline metrics for comparison."""
    bh_return = df["close"].pct_change().fillna(0)
    bh_equity = initial_capital * (1 + bh_return).cumprod()
    total_ret = bh_equity.iloc[-1] / initial_capital - 1
    n_days = len(bh_return)
    years = n_days / 252
    ann_ret = (1 + total_ret) ** (1 / max(years, 0.01)) - 1 if total_ret > -1 else -1.0
    d = bh_equity - bh_equity.cummax()
    max_dd = (d / bh_equity.cummax()).min()

    return {
        "strategy": "Buy & Hold",
        "total_return_pct": float(total_ret * 100),
        "annualized_return_pct": float(ann_ret * 100),
        "max_drawdown_pct": float(max_dd * 100),
        "equity_curve": bh_equity,
    }
