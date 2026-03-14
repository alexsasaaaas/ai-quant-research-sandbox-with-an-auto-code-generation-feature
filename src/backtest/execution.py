"""
execution.py
Trade execution simulation: converts signals to trades with commissions & slippage.
Signal generated at close T → trade executed at open T+1.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def simulate_execution(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 1_000_000,
    commission_rate: float = 0.001425,   # buyer side commission
    tax_rate: float = 0.003,             # seller side transaction tax (TW)
    slippage: float = 0.001,            # 0.1% slippage on execution
) -> pd.DataFrame:
    """
    Simulate long-only trade execution given binary signals (0 or 1).

    Execution rules:
    - Signal on day T → buy/sell at day T open (already shifted in signal gen)
    - Full-capital investment (no fractional shares here; simplified)
    - Commission on both buy and sell; tax on sell only

    Returns:
        DataFrame with columns: signal, price, shares, cash, equity, daily_return
    """
    # Use open price as execution price (next day's open after signal)
    price = df["open"].copy()

    n = len(df)
    equity = np.zeros(n)
    cash = np.zeros(n)
    shares = np.zeros(n)
    trade_prices = []

    cash[0] = initial_capital
    current_shares = 0.0
    current_cash = float(initial_capital)
    n_trades = 0
    n_wins = 0
    last_buy_price = 0.0

    for i in range(n):
        sig = signals.iloc[i]
        px = price.iloc[i]

        if pd.isna(px) or px <= 0:
            # No valid price — maintain position
            shares[i] = current_shares
            cash[i] = current_cash
            equity[i] = current_cash + current_shares * df["close"].iloc[i]
            continue

        # Effective prices after slippage
        buy_px = px * (1 + slippage)
        sell_px = px * (1 - slippage)

        if sig == 1 and current_shares == 0:
            # BUY: invest all available cash
            gross_shares = current_cash / (buy_px * (1 + commission_rate))
            cost = gross_shares * buy_px * (1 + commission_rate)
            if cost <= current_cash:
                current_shares = gross_shares
                current_cash -= cost
                last_buy_price = buy_px
                n_trades += 1

        elif sig == 0 and current_shares > 0:
            # SELL: close entire position
            proceeds = current_shares * sell_px * (1 - commission_rate - tax_rate)
            current_cash += proceeds
            if sell_px > last_buy_price:
                n_wins += 1
            current_shares = 0.0

        shares[i] = current_shares
        cash[i] = current_cash
        equity[i] = current_cash + current_shares * df["close"].iloc[i]

    equity_series = pd.Series(equity, index=df.index, name="equity")
    equity_series = equity_series.replace(0, np.nan).ffill().fillna(initial_capital)

    daily_return = equity_series.pct_change().fillna(0)

    result = df[["open", "high", "low", "close", "volume"]].copy()
    result["signal"] = signals.values
    result["equity"] = equity_series.values
    result["daily_return"] = daily_return.values
    result["drawdown"] = (
        equity_series / equity_series.cummax() - 1
    ).values

    return result, n_trades, n_wins
