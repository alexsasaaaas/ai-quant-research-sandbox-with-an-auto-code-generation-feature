import os
import sys

# 修正 Python 路徑，確保 Streamlit Cloud 能找到 src 模組
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

"""
2_Strategy_Sandbox.py
Backtesting interface for template strategies and model-based strategies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging

from src.visualization.dashboard_helpers import setup_page_config, render_metric_cards, initialize_session_state
from src.backtest.engine import run_backtest
from src.visualization.charts import plot_equity_curve, plot_drawdown, plot_ohlcv_with_signals

logger = logging.getLogger(__name__)

def run():
    initialize_session_state()
    setup_page_config("Strategy Sandbox")
    
    st.header("📉 Strategy Sandbox")
    st.caption("模板化回測框架：驗證預測模型或技術指標的實戰價值")

    # Check if we have data from Forecast Builder
    has_model_data = st.session_state.experiment_data is not None
    
    col_set1, col_set2 = st.columns([1, 2])
    
    with col_set1:
        st.subheader("🛠️ 策略配置")
        
        # Strategy selection
        strat_options = ["MA Cross", "RSI Mean Reversion"]
        if has_model_data:
            strat_options.append("Prediction-based")
            
        strategy_name = st.selectbox("選擇交易策略", strat_options, index=len(strat_options)-1)
        
        # Parameters depending on strategy
        params = {}
        with st.expander("📝 策略參數設定", expanded=True):
            if strategy_name == "MA Cross":
                params["fast_window"] = st.number_input("快速均線 (Fast)", 5, 60, 5)
                params["slow_window"] = st.number_input("慢速均線 (Slow)", 10, 240, 20)
            elif strategy_name == "RSI Mean Reversion":
                params["rsi_period"] = st.number_input("RSI 週期", 2, 30, 14)
                params["oversold"] = st.slider("超賣線 (Buy)", 10, 50, 30)
                params["overbought"] = st.slider("超買線 (Sell)", 50, 90, 70)
            elif strategy_name == "Prediction-based":
                st.info(f"模型：{st.session_state.experiment_data['meta']['model_name']} | 預測天數：{st.session_state.experiment_data['meta']['horizon']}")
                if st.session_state.experiment_data['meta']['task_type'] == "regression":
                    params["threshold"] = st.number_input("預測報酬門檻 (%)", -5.0, 5.0, 0.0, step=0.1) / 100.0
                else:
                    st.write("分類模型：預測為 1 即買入。")
        
        st.subheader("💰 帳戶設定")
        initial_capital = st.number_input("初始資金 (NTD)", 100000, 100000000, 1000000, step=100000)
        
        with st.expander("⚖️ 費用與滑點", expanded=False):
            commission = st.number_input("手續費率 (%)", 0.0, 1.0, 0.1425, step=0.01) / 100.0
            tax = st.number_input("交易稅率 (%)", 0.0, 1.0, 0.3, step=0.1) / 100.0
            slippage = st.number_input("假設滑點 (%)", 0.0, 1.0, 0.1, step=0.05) / 100.0

        if st.button("📈 執行回測", type="primary", use_container_width=True):
            if not has_model_data and strategy_name == "Prediction-based":
                st.error("請先前往 Forecast Builder 訓練模型！")
            else:
                # Get predictive data if needed
                predictions = None
                task_type = "regression"
                df = None
                
                if has_model_data:
                    exp = st.session_state.experiment_data
                    df = exp["df_final"]
                    task_type = exp["meta"]["task_type"]
                    # Get predictions for the whole df if possible, or just the test set
                    # Here we re-predict on the full df used in trainer (df_final)
                    model = exp["train_res"]["model"]
                    X = df[exp["feature_cols"]].values
                    predictions = pd.Series(model.predict(X), index=df.index)
                else:
                    # Fallback if no forecast builder was run - need to load default data
                    from src.data.loader import load_stock_data
                    from src.utils.constants import DEFAULT_TICKER
                    df = load_stock_data(DEFAULT_TICKER, "2021-01-01", "2023-12-31")
                
                with st.spinner("計算中..."):
                    bt_res = run_backtest(
                        df, 
                        strategy_name, 
                        params, 
                        initial_capital=initial_capital,
                        commission_rate=commission,
                        tax_rate=tax,
                        slippage=slippage,
                        predictions=predictions,
                        task_type=task_type
                    )
                    st.session_state.backtest_results = bt_res
                    st.success("回測完成！")

    with col_set2:
        if st.session_state.backtest_results:
            res = st.session_state.backtest_results
            st.subheader(f"📊 回測表現: {res['strategy_name']}")
            
            # Key metrics cards
            render_metric_cards(res["metrics"])
            
            # Main chart: Equity
            fig_equity = plot_equity_curve(res["execution_df"], res["bh_metrics"]["equity_curve"], res["strategy_name"])
            st.plotly_chart(fig_equity, use_container_width=True)
            
            # Secondary chart: Drawdown
            fig_dd = plot_drawdown(res["execution_df"])
            st.plotly_chart(fig_dd, use_container_width=True)
            
            with st.expander("📍 查看具體進出場訊號圖", expanded=False):
                from src.utils.constants import DEFAULT_TICKER
                t = st.session_state.experiment_data["meta"]["ticker"] if has_model_data else "2330.TW"
                fig_sig = plot_ohlcv_with_signals(res["execution_df"], t)
                st.plotly_chart(fig_sig, use_container_width=True)
                
            st.info("✨ **下一步建議**: 前往「Report Center」生成完整的研究總結與 AI 分析報告。")
        else:
            st.info("請在左側側邊欄設定策略參數並點擊「執行回測」。")

if __name__ == "__main__":
    run()
