"""
charts.py
Visualization functions for the Streamlit dashboard using Plotly and Matplotlib.
Includes prediction plots, equity curves, and feature importance charts.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_prediction_vs_actual(y_test, y_pred, dates, ticker: str, task_type: str):
    """Plot predicted vs actual returns/classes."""
    fig = go.Figure()
    
    if task_type == "regression":
        fig.add_trace(go.Scatter(x=dates, y=y_test, name="Actual", line=dict(color="blue", width=1.5)))
        fig.add_trace(go.Scatter(x=dates, y=y_pred, name="Predicted", line=dict(color="red", width=1.5, dash="dot")))
        fig.update_layout(title=f"{ticker} - Predicted vs Actual Returns", yaxis_title="Return")
    else:
        # For classification, plot as points or bars
        fig.add_trace(go.Scatter(x=dates, y=y_test, name="Actual Class", mode="markers", marker=dict(symbol="circle", size=8, color="blue", opacity=0.5)))
        fig.add_trace(go.Scatter(x=dates, y=y_pred, name="Predicted Class", mode="markers", marker=dict(symbol="x", size=8, color="red")))
        fig.update_layout(title=f"{ticker} - Predicted vs Actual Class (1=Up, 0=Down)", yaxis_title="Class")

    fig.update_layout(height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def plot_equity_curve(exec_df: pd.DataFrame, bh_equity: pd.Series, strategy_name: str):
    """Plot Strategy Equity Curve vs Buy & Hold."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=exec_df.index, y=exec_df["equity"], name=f"Strategy: {strategy_name}", line=dict(color="green", width=2)))
    fig.add_trace(go.Scatter(x=bh_equity.index, y=bh_equity, name="Buy & Hold", line=dict(color="gray", width=1.5, dash="dash")))
    
    fig.update_layout(
        title="Equity Curve Comparison",
        yaxis_title="Portfolio Value (NTD)",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_drawdown(exec_df: pd.DataFrame):
    """Plot Strategy Drawdown."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=exec_df.index, 
        y=exec_df["drawdown"] * 100, 
        name="Drawdown", 
        fill="tozeroy",
        line=dict(color="red", width=1)
    ))
    
    fig.update_layout(
        title="Strategy Drawdown (%)",
        yaxis_title="Drawdown (%)",
        height=300,
        showlegend=False
    )
    return fig


def plot_feature_importance(fi_df: pd.DataFrame):
    """Plot Feature Importance bar chart."""
    if fi_df is None or fi_df.empty:
        return None
        
    top_fi = fi_df.head(15).iloc[::-1] # Top 15, largest at top
    
    fig = go.Figure(go.Bar(
        x=top_fi["importance"],
        y=top_fi["feature"],
        orientation="h",
        marker=dict(color="rgba(50, 171, 96, 0.6)", line=dict(color="rgba(50, 171, 96, 1.0)", width=1))
    ))
    
    fig.update_layout(
        title="Feature Importance (Top 15)",
        xaxis_title="Relative Importance",
        height=500,
        margin=dict(l=150)
    )
    return fig


def plot_ohlcv_with_signals(exec_df: pd.DataFrame, ticker: str):
    """Plot OHLCV with Buy/Sell signals."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=exec_df.index,
        open=exec_df["open"], high=exec_df["high"], low=exec_df["low"], close=exec_df["close"],
        name="OHLC"
    ), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=exec_df.index, y=exec_df["volume"], name="Volume", marker_color="rgba(100,100,100,0.5)"), row=2, col=1)
    
    # Entry/Exit Signals
    entries = exec_df[(exec_df["signal"] == 1) & (exec_df["signal"].shift(1) == 0)]
    exits = exec_df[(exec_df["signal"] == 0) & (exec_df["signal"].shift(1) == 1)]
    
    fig.add_trace(go.Scatter(
        x=entries.index, y=entries["low"] * 0.98,
        mode="markers", name="Buy Signal",
        marker=dict(symbol="triangle-up", size=12, color="green")
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=exits.index, y=exits["high"] * 1.02,
        mode="markers", name="Sell Signal",
        marker=dict(symbol="triangle-down", size=12, color="red")
    ), row=1, col=1)
    
    fig.update_layout(
        title=f"{ticker} - OHLCV & Signals",
        height=700,
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    return fig
