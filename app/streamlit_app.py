"""
streamlit_app.py
Main entry point for the AI Quant Research Sandbox.
Handles navigation and high-level configuration.
"""

import os
import sys

# 修正 Python 路徑，確保 Streamlit Cloud 能找到 src 模組 (Updated 2026-03-15)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import streamlit as st
from src.visualization.dashboard_helpers import setup_page_config, display_research_banner, initialize_session_state
from src.utils.logger import setup_logger
from dotenv import load_dotenv

# Initialize session state for experiment data persistence across pages
initialize_session_state()

def main():
    # Load environment variables (API keys)
    load_dotenv()
    
    # Configure logging
    setup_logger()
    
    # Page setup
    setup_page_config("主導引")
    display_research_banner()
    
    st.markdown("""
    ### 👋 歡迎來到 AI 量化研究沙盒
    
    本平台旨在提供一個模組化的環境，讓您快速建立與評估基於機器學習的台股預測任務。
    
    #### 🚀 快速開始流程：
    
    1.  **Forecast Builder (預測建模器)**: 
        - 選擇股票與天數。
        - 訓練模型 (Linear, LightGBM, XGBoost)。
        - 查看預測準確度與特徵重要性。
    
    2.  **Strategy Sandbox (策略沙盒)**:
        - 根據模型訊號或傳統指標建立交易策略。
        - 進行回測並觀察績效、回撤與風險。
    
    3.  **Report Center (報告中心)**:
        - 自動生成結構化的研究報告。
        - 支援 LLM (OpenAI/Anthropic) 自然語言分析或自動模板。
        - 下載 Markdown 或 HTML 格式報告。
        
    #### ⚖️ 重要聲明
    - 本工具僅供**量化研究與開發原型**使用。
    - 內建回測已考量交易稅與手續費，但未考慮市場流動性與極端風險。
    - **非投資建議**。所有預測結果均基於歷史資料，不代表未來表現。
    
    ---
    *請點選左側側邊欄開始您的量化研究之旅。*
    """)
    
    with st.expander("🛠️ 系統狀態與設定", expanded=False):
        import os
        has_groq = "✅ 加載成功" if os.getenv("GROQ_API_KEY") else "❌ 未設定"
        has_openai = "✅ 加載成功" if os.getenv("OPENAI_API_KEY") else "❌ 未設定"
        has_anthropic = "✅ 加載成功" if os.getenv("ANTHROPIC_API_KEY") else "❌ 未設定"
        
        st.write(f"- Groq API Key: {has_groq}")
        st.write(f"- OpenAI API Key: {has_openai}")
        st.write(f"- Anthropic API Key: {has_anthropic}")
        st.write("- 資料緩存目錄: `data/cache/` (清理此目錄可強制更新資料)")

if __name__ == "__main__":
    main()
