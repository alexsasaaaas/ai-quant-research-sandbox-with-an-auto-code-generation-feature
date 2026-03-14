import os
import sys

# 修正 Python 路徑，確保 Streamlit Cloud 能找到 src 模組
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import streamlit as st
import pandas as pd
import logging
from datetime import datetime

from src.visualization.dashboard_helpers import (
    setup_page_config, 
    render_metric_cards, 
    initialize_session_state
)
from src.agent.prompt_parser import generate_modeling_code
from src.agent.workflow_runner import run_agent_workflow
from src.agent.agent_summary import generate_agent_summary
from src.visualization.charts import plot_prediction_vs_actual, plot_feature_importance, plot_equity_curve
from src.models.evaluator import get_feature_importance
from src.backtest.engine import run_backtest
from src.report.summary_builder import build_research_summary
from src.report.exporter import generate_report, _md_to_html

logger = logging.getLogger(__name__)

def run():
    # 重要：此頁面不持久化 session_state，模擬重新整理即重置
    initialize_session_state()
    setup_page_config("Agent Model Builder")
    
    st.header("🤖 Agent Model Builder (Code Gen Mode)")
    st.caption("AI 根據您的需求動態編寫 Python 程式碼並執行建模工作流")
    
    with st.expander("ℹ️ 關於 Code Generation 模式", expanded=False):
        st.write("""
        1. **Code Gen**: Agent 會解析您的 Prompt 並直接**編寫**一段自定義的 Python 研究腳本。
        2. **Execution**: 使用 Python 的 `exec()` 功能動態執行這段腳本。
        3. **Auto Repair**: 如果腳本執行噴錯，Agent 會讀取 Error Message 並嘗試**修改程式碼**重新執行。
        5. **Rate Limit Reminder**: 目前展示使用免費 LLM 模型，若頻繁執行可能觸發 API 限制。
        """)
    st.info("💡 **提示**：目前展示使用的是免費級別的 LLM 模型 (Groq/OpenAI)，較易觸發呼叫次數限制 (Rate Limit)。建議每次執行間隔稍作停頓。")

    # --- Step 1: Prompt Input ---
    st.subheader("📝 Step 1: 輸入研究需求")
    user_prompt = st.text_area(
        "請描述您想要建立的模型", 
        placeholder="例如：建立一個預測 2330 明天漲跌的分類任務，模型用穩定一點的線性模型就好",
        height=100
    )
    
    col_run, col_reset = st.columns([4, 1])
    
    if col_run.button("🚀 生成並執行程式碼", type="primary", use_container_width=True):
        if not user_prompt:
            st.warning("請先輸入需求描述！")
            return
            
        # Reset relative states (in-page logic)
        st.session_state.agent_result = None
        st.session_state.backtest_results = None
        if "final_report" in st.session_state: del st.session_state["final_report"]
        
        with st.status("🤖 Agent 正在編寫程式碼...", expanded=True) as status:
            # A. Code Generation
            st.write("✍️ 正在編寫研究腳本...")
            generated_code = generate_modeling_code(user_prompt)
            
            # B. Code Execution (with Auto-Repair)
            st.write("⚙️ 正在動態執行產出的內容...")
            result = run_agent_workflow(generated_code)
            
            st.session_state.agent_result = result
            
            if result.get("success"):
                # Sync with project conventions
                st.session_state.experiment_data = {
                    "meta": {
                        "ticker": result["ticker"],
                        "task_type": result["task_type"],
                        "horizon": result["horizon"],
                        "model_name": result["model_name"]
                    },
                    "feature_cols": result["feature_cols"],
                    "train_res": result["train_res"],
                    "eval_res": result["eval_res"],
                    "df_final": result["df_final"],
                    "data_summary": {"ticker": result["ticker"]}
                }
                status.update(label="✅ 程式碼執行成功！", state="complete", expanded=False)
            else:
                status.update(label="❌ 程式碼執行失敗", state="error", expanded=True)
                st.error(f"Agent 已嘗試修復但仍發生錯誤: {result.get('error')}")

    if col_reset.button("🔄 重置頁面"):
        st.rerun()

    # --- Step 2: Display Results & Code ---
    if "agent_result" in st.session_state and st.session_state.agent_result:
        res = st.session_state.agent_result
        st.divider()
        st.markdown(generate_agent_summary(res))
        
        with st.expander("💻 查看 Agent 編寫的程式碼", expanded=False):
            st.code(res.get("code", ""), language="python")

        if res.get("success"):
            # Evaluation Section
            st.subheader("📊 模型表現與分析")
            eval_res = res.get("eval_res")
            
            if isinstance(eval_res, dict) and "test_metrics" in eval_res and "y_test" in eval_res:
                render_metric_cards(eval_res["test_metrics"])
                
                col_eval1, col_eval2 = st.columns([2, 1])
                with col_eval1:
                    fig_pred = plot_prediction_vs_actual(
                        eval_res["y_test"], 
                        eval_res["y_pred_test"], 
                        eval_res["test_dates"],
                        res.get("ticker", "N/A"),
                        res.get("task_type", "regression")
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                with col_eval2:
                    fi_df = get_feature_importance(res["train_res"]["model"], res["feature_cols"])
                    if fi_df is not None:
                        fig_fi = plot_feature_importance(fi_df)
                        st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.warning("⚠️ 無法讀取完整的模型評估指標（如 y_test 或 test_metrics 缺失）。")
                st.write("這通常是因為自定義模型程式碼未能正確產出標準格式的 `eval_res`。")
                with st.expander("查看原始評估輸出", expanded=False):
                    st.write(eval_res)

            # --- Step 3: Fast Backtest ---
            st.divider()
            st.subheader("📉 Step 2: 快速策略回測")
            
            col_bt_set, col_bt_res = st.columns([1, 2])
            with col_bt_set:
                strategy_name = st.selectbox("策略模式", ["Prediction-based", "MA Cross"], index=0)
                initial_capital = st.number_input("初始資金", 1000000)
                
                if st.button("📈 運行回測", type="primary", use_container_width=True):
                    with st.spinner("回測中..."):
                        df = res["df_final"]
                        model = res["train_res"]["model"]
                        feat_cols = res["feature_cols"]
                        X = df[feat_cols].values
                        full_preds = pd.Series(model.predict(X), index=df.index)
                        
                        bt_res = run_backtest(
                            df, 
                            strategy_name, 
                            {}, # Simple default params
                            initial_capital=initial_capital,
                            predictions=full_preds,
                            task_type=res["task_type"]
                        )
                        st.session_state.backtest_results = bt_res
            
            with col_bt_res:
                if "backtest_results" in st.session_state and st.session_state.backtest_results:
                    bt = st.session_state.backtest_results
                    render_metric_cards(bt["metrics"])
                    fig_equity = plot_equity_curve(bt["execution_df"], bt["bh_metrics"]["equity_curve"], bt["strategy_name"])
                    st.plotly_chart(fig_equity, use_container_width=True)

            # --- Step 4: Final Report ---
            if "backtest_results" in st.session_state and st.session_state.backtest_results:
                st.divider()
                st.subheader("📝 Step 3: 自動回報研究成果")
                if st.button("🪄 生成 AI 研究報告", type="primary", use_container_width=True):
                    with st.spinner("撰寫中..."):
                        exp = st.session_state.experiment_data
                        bt = st.session_state.backtest_results
                        summary = build_research_summary(
                            ticker=exp["meta"]["ticker"],
                            start_date="Unknown", end_date="Today",
                            task_type=exp["meta"]["task_type"],
                            horizon=exp["meta"]["horizon"],
                            model_name=exp["meta"]["model_name"],
                            data_summary={"ticker_info": {"symbol": exp["meta"]["ticker"]}},
                            feature_cols=exp["feature_cols"],
                            eval_result=exp["eval_res"],
                            backtest_result=bt
                        )
                        report_text, used_mode = generate_report(summary, mode="auto")
                        st.session_state.final_report = report_text
                        st.session_state.used_mode = used_mode
                
                if "final_report" in st.session_state:
                    st.success(f"報告已產出 (使用模式: {st.session_state.used_mode})")
                    st.markdown(st.session_state.final_report)
                    st.download_button("📥 下載報告 (MD)", st.session_state.final_report, file_name="agent_report.md")

if __name__ == "__main__":
    run()
