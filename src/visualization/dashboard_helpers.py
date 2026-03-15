"""
dashboard_helpers.py
Utility functions for the Streamlit UI components (sidebar, metrics, formatting).
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def initialize_session_state():
    """Ensure all required session state variables are initialized."""
    if "experiment_data" not in st.session_state:
        st.session_state.experiment_data = None
    if "backtest_results" not in st.session_state:
        st.session_state.backtest_results = None
    if "research_summary" not in st.session_state:
        st.session_state.research_summary = None


def setup_page_config(title: str):
    """Set standard page config and styling."""
    st.set_page_config(
        page_title=f"{title} - AI Quant Sandbox",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS for a premium feel
    st.markdown("""
    <style>
    /* Metric Card Styling */
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* Ensure label color is visible */
    [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 600 !important;
    }
    
    /* Ensure value color is visible */
    [data-testid="stMetricValue"] {
        color: #1e293b !important;
    }

    .stButton>button { border-radius: 8px; font-weight: 500; }
    h1, h2, h3 { color: #1e293b; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)


def display_research_banner():
    st.title("🧪 AI Quant Research Sandbox")
    st.markdown("""
    *建立股票預測模型與模板化策略的回撤沙盒。本工具僅供研究用途，非投資建議。*
    """)
    st.divider()


def render_metric_cards(metrics_dict: dict, cols=4):
    """Helper to render a grid of metrics cards."""
    # Convert flat dict to readable keys
    display_names = {
        "rmse": "RMSE (誤差 ↓)",
        "mae": "MAE (誤差 ↓)",
        "r2": "R² (得分 ↑)",
        "direction_accuracy": "方向準確率",
        "accuracy": "準確率",
        "precision": "精確率",
        "recall": "召回率",
        "f1_score": "F1 Score",
        "total_return_pct": "總報酬",
        "annualized_return_pct": "年化報酬",
        "sharpe_ratio": "夏普比率",
        "max_drawdown_pct": "最大回撤",
        "win_rate_pct": "勝率",
        "n_trades": "交易次數"
    }
    
    items = list(metrics_dict.items())
    for i in range(0, len(items), cols):
        row_items = items[i:i+cols]
        cols_obj = st.columns(len(row_items))
        for col_idx, (k, v) in enumerate(row_items):
            label = display_names.get(k, k.replace("_", " ").title())
            
            # Formatting
            try:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    val_str = "N/A"
                elif "pct" in k or k in ["accuracy", "precision", "recall", "direction_accuracy"]:
                    # Ensure we handle values that are already percentages or decimal fractions
                    display_val = v * 100.0 if abs(v) <= 1.0 else v
                    val_str = f"{display_val:.2f}%"
                elif k in ["n_trades"]:
                    val_str = f"{int(v)}"
                else:
                    val_str = f"{v:.4f}"
            except Exception as e:
                logger.error(f"Error formatting metric {k}: {e}")
                val_str = str(v)
                
            cols_obj[col_idx].metric(label, val_str)


def sidebar_stock_selector():
    """Generic stock selection sidebar."""
    from src.data.tw_stock_api import get_available_tickers
    
    st.sidebar.header("📁 研究對象")
    
    popular = get_available_tickers()
    ticker_input = st.sidebar.selectbox("選擇或輸入股票代碼", popular + ["自定義..."])
    
    if ticker_input == "自定義...":
        ticker = st.sidebar.text_input("輸入台股代碼 (例: 2454.TW)", "2330.TW")
    else:
        ticker = ticker_input
        
    start_date = st.sidebar.date_input(
        "起始日期", 
        datetime.now() - timedelta(days=365*3),
        min_value=datetime(2000, 1, 1),
        max_value=datetime.now()
    )
    end_date = st.sidebar.date_input(
        "結束日期", 
        datetime.now(),
        min_value=datetime(2000, 1, 1),
        max_value=datetime.now()
    )
    
    return ticker, str(start_date), str(end_date)
