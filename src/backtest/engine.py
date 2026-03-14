"""
engine.py
Main backtest orchestrator. Ties together strategy → execution → metrics.
"""

import logging
import pandas as pd
import numpy as np

from src.backtest.strategies import (
    ma_cross_signals,
    rsi_mean_reversion_signals,
    prediction_based_signals,
)
from src.backtest.execution import simulate_execution
from src.backtest.metrics import compute_backtest_metrics, compute_buy_and_hold

logger = logging.getLogger(__name__)


def run_backtest(
    df: pd.DataFrame,
    strategy_name: str,
    strategy_params: dict,
    initial_capital: float = 1_000_000,
    commission_rate: float = 0.001425,
    tax_rate: float = 0.003,
    slippage: float = 0.001,
    predictions: pd.Series = None,
    task_type: str = "regression",
) -> dict:
    """
    Run a full backtest for the given strategy on df.

    Returns: dict with signals, execution_df, metrics, bh_metrics
    """
    df = df.copy().dropna(subset=["open", "high", "low", "close"])

    # --- Signal Generation ---
    if strategy_name == "MA Cross":
        signals = ma_cross_signals(
            df,
            fast_window=strategy_params.get("fast_window", 5),
            slow_window=strategy_params.get("slow_window", 20),
        )
    elif strategy_name == "RSI Mean Reversion":
        signals = rsi_mean_reversion_signals(
            df,
            rsi_period=strategy_params.get("rsi_period", 14),
            oversold=strategy_params.get("oversold", 30.0),
            overbought=strategy_params.get("overbought", 70.0),
        )
    elif strategy_name == "Prediction-based":
        if predictions is None:
            raise ValueError("Prediction-based strategy requires 'predictions' argument.")
        signals = prediction_based_signals(
            df,
            predictions=predictions,
            task_type=task_type,
            threshold=strategy_params.get("threshold", 0.0),
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # --- Execution ---
    exec_df, n_trades, n_wins = simulate_execution(
        df,
        signals,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        tax_rate=tax_rate,
        slippage=slippage,
    )

    equity_curve = pd.Series(exec_df["equity"].values, index=exec_df.index)
    daily_returns = pd.Series(exec_df["daily_return"].values, index=exec_df.index)

    # --- Metrics ---
    metrics = compute_backtest_metrics(
        equity_curve=equity_curve,
        returns=daily_returns,
        initial_capital=initial_capital,
        n_trades=n_trades,
        n_wins=n_wins,
    )

    # --- Buy & Hold Comparison ---
    bh = compute_buy_and_hold(df, initial_capital)

    logger.info(
        f"Backtest [{strategy_name}]: return={metrics['total_return_pct']:.1f}%, "
        f"sharpe={metrics['sharpe_ratio']:.2f}, maxdd={metrics['max_drawdown_pct']:.1f}%, "
        f"trades={n_trades}"
    )

    return {
        "strategy_name": strategy_name,
        "strategy_params": strategy_params,
        "signals": signals,
        "execution_df": exec_df,
        "equity_curve": equity_curve,
        "daily_returns": daily_returns,
        "metrics": metrics,
        "bh_metrics": bh,
        "initial_capital": initial_capital,
    }
