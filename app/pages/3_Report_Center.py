import os
import sys

# 修正 Python 路徑，確保 Streamlit Cloud 能找到 src 模組
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

"""
3_Report_Center.py
Research summary generation and natural language reporting (Template or LLM).
"""

import streamlit as st
import json
import logging

from src.visualization.dashboard_helpers import setup_page_config, initialize_session_state
from src.report.summary_builder import build_research_summary
from src.report.exporter import generate_report, save_report

logger = logging.getLogger(__name__)

def run():
    initialize_session_state()
    setup_page_config("Report Center")
    
    st.header("📝 Report Center")
    st.caption("自動生成結構化的量化研究摘要與 AI 報告")

    # Check dependencies
    if not st.session_state.experiment_data:
        st.warning("⚠️ 查無預測模型資料。請先前往 「Forecast Builder」 完成模型訓練。")
        return

    exp = st.session_state.experiment_data
    bt = st.session_state.backtest_results
    
    st.subheader("📋 步驟 1: 建立研究摘要")
    
    if st.button("🏗️ 生成結構化研究摘要", type="primary"):
        with st.spinner("彙整資料中..."):
            summary = build_research_summary(
                ticker=exp["meta"]["ticker"],
                start_date=exp["meta"]["start_date"],
                end_date=exp["meta"]["end_date"],
                task_type=exp["meta"]["task_type"],
                horizon=exp["meta"]["horizon"],
                model_name=exp["meta"]["model_name"],
                data_summary=exp["data_summary"],
                feature_cols=exp["feature_cols"],
                eval_result=exp["eval_res"],
                backtest_result=bt
            )
            st.session_state.research_summary = summary
            st.success("摘要已生成！")

    if st.session_state.research_summary:
        summary = st.session_state.research_summary
        
        with st.expander("🔍 查看結構化 JSON 數據", expanded=False):
            st.json(summary)
            
        st.divider()
        st.subheader("✍️ 步驟 2: 生成自然語言報告")
        
        mode = st.radio("報告模式", ["auto", "template", "llm"], index=0, 
                        help="auto: 有 API key 則使用 LLM，否則使用模板。")
        
        if st.button("🪄 開始撰寫報告", use_container_width=True):
            with st.spinner("AI 正在分析數據並撰寫報告中..."):
                report_text, used_mode = generate_report(summary, mode=mode)
                st.session_state.final_report = report_text
                st.session_state.used_mode = used_mode
                
        if "final_report" in st.session_state:
            st.markdown("---")
            st.success(f"已使用 「{st.session_state.used_mode}」 模式生成報告")
            
            # Display report
            st.markdown(st.session_state.final_report)
            
            st.divider()
            st.subheader("📥 步驟 3: 下載報告")
            
            col1, col2 = st.columns(2)
            
            # Save to disk first to get the local path (optional, for exporter)
            with col1:
                md_path = save_report(st.session_state.final_report, summary["experiment_metadata"]["ticker"], format="md")
                st.download_button(
                    "📥 下載 Markdown 報告",
                    st.session_state.final_report,
                    file_name=f"Quant_Report_{summary['experiment_metadata']['ticker']}.md",
                    mime="text/markdown"
                )
                
            with col2:
                # For HTML, we use the internal exporter helper
                from src.report.exporter import _md_to_html
                html_text = _md_to_html(st.session_state.final_report, summary["experiment_metadata"]["ticker"])
                st.download_button(
                    "📥 下載 HTML 報告",
                    html_text,
                    file_name=f"Quant_Report_{summary['experiment_metadata']['ticker']}.html",
                    mime="text/html"
                )

if __name__ == "__main__":
    run()
