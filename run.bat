@echo off
title 預測股票趨勢 - Streamlit Launcher

echo [INFO] 正在檢查環境...

echo [INFO] 啟動 Streamlit...
python -m streamlit run "C:\Users\alex\Desktop\AI Quant Research Sandbox for Taiwan Stocks\ai-quant-research-sandbox\app\streamlit_app.py" --server.port 8501

echo.
echo [結束] 應用程式已關閉，按任意鍵離開...
pause
