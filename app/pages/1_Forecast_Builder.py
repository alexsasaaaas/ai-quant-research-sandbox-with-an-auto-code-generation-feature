import os
import sys

# 修正 Python 路徑，確保 Streamlit Cloud 能找到 src 模組
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

"""
1_Forecast_Builder.py
Model training and evaluation interface.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import logging

from src.visualization.dashboard_helpers import (
    setup_page_config, 
    sidebar_stock_selector, 
    render_metric_cards, 
    initialize_session_state
)
from src.data.loader import load_stock_data, get_data_summary
from src.data.preprocess import build_target
from src.data.feature_engineering import build_features, get_feature_columns
from src.models.trainer import train_model
from src.models.evaluator import evaluate_model, get_feature_importance
from src.visualization.charts import plot_prediction_vs_actual, plot_feature_importance

logger = logging.getLogger(__name__)

def run():
    initialize_session_state()
    setup_page_config("Forecast Builder")
    
    st.header("🎯 Forecast Builder")
    st.caption("若想要重新訓練模型，請重新整理網頁，若有 bug 也請再次重新整理網頁")
    
    # --- Sidebar Configuration ---
    ticker, start_date, end_date = sidebar_stock_selector()
    
    st.sidebar.divider()
    st.sidebar.subheader("⚙️ 任務設定")
    task_type = st.sidebar.radio("任務類型", ["regression", "classification"], index=0)
    horizon = st.sidebar.select_slider("預測 Horizon (天)", options=[1, 5, 20], value=1)
    
    st.sidebar.subheader("🧠 模型設定")
    model_name = st.sidebar.selectbox("預設的模型演算法", ["lightgbm", "xgboost", "linear", "baseline"], index=0)
    
    st.sidebar.subheader("🛠️ 特徵群組")
    use_price = st.sidebar.checkbox("價格特徵 (SMA, Return, etc.)", value=True)
    use_vol = st.sidebar.checkbox("量能特徵 (MA, Ratio, OBV)", value=True)
    use_tech = st.sidebar.checkbox("技術指標 (MACD, RSI, BB, etc.)", value=True)
    use_fund = st.sidebar.checkbox("基本面特徵 (PE, PB, MarketCap)", value=False)
    
    if st.sidebar.button("🚀 開始運行預測任務", type="primary", use_container_width=True):
        with st.status("正在執行預測任務...", expanded=True) as status:
            # 1. Load Data
            st.write("📥 獲取股票資料...")
            df_raw = load_stock_data(ticker, start_date, end_date)
            data_sum = get_data_summary(df_raw)
            
            # Show fundamentals found in info
            fund_info = data_sum.get("fundamentals", {})
            if fund_info:
                st.write(f"ℹ️ 發現基本面資料: {fund_info.get('long_name', 'Unknown')}")
            
            # 2. Tech Indicators
            st.write("🔧 計算特徵工程...")
            df_feat = build_features(
                df_raw, 
                use_price=use_price, 
                use_volume=use_vol, 
                use_technical=use_tech, 
                use_fundamental=use_fund
            )
            feat_cols = get_feature_columns(
                df_feat, 
                use_price=use_price, 
                use_volume=use_vol, 
                use_technical=use_tech, 
                use_fundamental=use_fund
            )
            
            # 3. Target Building
            st.write("🎯 建立目標變數...")
            df_target = build_target(df_feat, task_type, horizon)
            
            # --- Data Preview (Crucial for Debugging) ---
            st.write("🗂️ 資料篩選預覽...")
            df_final = df_target.dropna(subset=feat_cols + ["target"])
            
            col_d1, col_d2, col_d3 = st.columns(3)
            col_d1.metric("原始資料", f"{len(df_raw)} 筆")
            col_d2.metric("特徵後", f"{len(df_feat)} 筆")
            col_d3.metric("過濾後 (可用)", f"{len(df_final)} 筆")
            
            if len(df_final) == 0:
                st.error("重大錯誤：過濾後完全沒有可用資料！請檢查日期範圍是否過短 (建議至少一年)。")
                status.update(label="❌ 任務失敗", state="error")
                return

            # 4. Training
            st.write(f"🧠 訓練 {model_name} 模型...")
            try:
                train_res = train_model(df_final, feat_cols, task_type, model_name)
                
                # 5. Evaluation
                st.write("📊 評估模型表現...")
                eval_res = evaluate_model(train_res)
                
                status.update(label="✅ 任務完成！", state="complete", expanded=False)
                
                # Store in session state for next pages
                st.session_state.experiment_data = {
                    "meta": {
                        "ticker": ticker, 
                        "start_date": start_date, 
                        "end_date": end_date, 
                        "task_type": task_type, 
                        "horizon": horizon, 
                        "model_name": model_name,
                        "use_fundamental": use_fund
                    },
                    "data_summary": data_sum,
                    "feature_cols": feat_cols,
                    "train_res": train_res,
                    "eval_res": eval_res,
                    "df_final": df_final
                }
            except ValueError as e:
                status.update(label="❌ 任務失敗", state="error", expanded=True)
                st.error(f"模型訓練失敗: {str(e)}")
                st.session_state.experiment_data = None
    
    # --- Display Results ---
    if st.session_state.experiment_data:
        exp = st.session_state.experiment_data
        eval_res = exp["eval_res"]
        
        st.subheader("📊 模型評估結果 (測試集)")
        
        with st.expander("🔍 數據診斷 (Metrics Debug)", expanded=False):
            st.write("原始評估數值：", eval_res["test_metrics"])
            
        render_metric_cards(eval_res["test_metrics"])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("##### 預測趨勢圖 (Prediction vs Actual)")
            fig_pred = plot_prediction_vs_actual(
                eval_res["y_test"], 
                eval_res["y_pred_test"], 
                eval_res["test_dates"],
                exp["meta"]["ticker"],
                exp["meta"]["task_type"]
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
        with col2:
            st.markdown("##### 特徵重要性 (Top 15)")
            fi_df = get_feature_importance(exp["train_res"]["model"], exp["feature_cols"])
            if fi_df is not None:
                fig_fi = plot_feature_importance(fi_df)
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("當前模型不支持導出特徵重要性。")
                
        st.success("✨ **下一步建議**: 前往「Strategy Sandbox」將此模型的預測轉化為交易策略進行回測。")

    else:
        st.info("請在左側側邊欄設定研究參數並點擊「開始運行預測任務」。")

if __name__ == "__main__":
    run()
