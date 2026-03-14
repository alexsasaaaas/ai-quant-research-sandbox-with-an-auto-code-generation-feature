"""
auto_debugger.py
Uses LLM to automatically repair generated code with full traceback and data context.
"""

import logging
import re
import json
from src.agent.llm_client import call_llm

logger = logging.getLogger(__name__)

DEBUGGER_SYSTEM_PROMPT = """你是一位專業的 Python 調試專家與量化工程師。
你的任務是讀取一段執行失敗的程式碼、完整的 Error Traceback 以及當前的資料上下文 (Data Context)，並對程式碼進行修復。

你必須遵循以下規範：
1. **深度分析 Traceback**：不要只看最後一行錯誤。從 Traceback 中找出具體報錯的行數與函式調用鏈。
2. **利用 Data Context**：如果 Traceback 顯示與資料維度、特徵名稱、或 shape 相關的錯誤，請參考提供的 Data Context (如 X_train 的維度) 來修正程式碼。
3. **處理缺失庫 (Dependency Issues)**：如果 Traceback 顯示 `ModuleNotFoundError` 或指出某個庫 (如 `tensorflow`, `keras`, `torch`) 未安裝，你可以嘗試在修復後的程式碼中先調用 `install_package("庫名稱")`（例如：`install_package("tensorflow")`）。
4. **驗證庫簽名**：如果錯誤涉及參數缺失 (missing argument)，請參考提供的 Data Context 並修正簽名。
5. **修補原則**：
   - 找出根源並修補。
   - 保持原有的核心模型邏輯 (例如：不要隨意把 LSTM 改成 Linear)。
   - 確保修復後的程式碼仍包含 `execute_research()` 函式與 `result = execute_research()`。
6. **核級選項 (Nuclear Option)**：如果是**重試後期 (Attempt 12+)** 且錯誤依然無法解決，請徹底簡化模型（如改用 Linear），以確保使用者至少能看到一份基準報告。
7. **僅輸出 Markdown 程式碼塊**：不要有額外解釋，只輸出程式碼內容。
"""

def repair_code_from_error(code: str, error_msg: str, attempt: int, context: dict = None) -> tuple:
    """
    使用 LLM 自動化分析並修復程式碼。
    """
    logger.info(f"Attempting LLM repair for attempt {attempt}.")
    
    # 建立上下文描述
    context_str = json.dumps(context, indent=2, ensure_ascii=False) if context else "No extra context available."
    
    is_last_chance = "【這是最後幾次重試機會，若持續失敗請考慮改用最穩定的模型結構】" if attempt >= 12 else ""

    # 預處理：常見錯誤類型的特定提示
    extra_hint = ""
    if "too many indices for array" in error_msg.lower():
        extra_hint = "\n💡 提示：發生了維度錯誤。請檢查 NumPy 陣列的 shape。注意 `y` 通常是 1D `(N,)`，而 `X` 是 2D `(N, features)`。若對 1D 使用 `y[:, 0]` 會報錯。"
    elif "not subscriptable" in error_msg.lower():
        extra_hint = "\n💡 提示：你可能把一個數字當成了 Dictionary 使用。請檢查你的函式傳回值格式。"
    elif "input_shape" in error_msg.lower():
        extra_hint = "\n💡 提示：深度學習維度錯誤。對於 LSTM，輸入應為 `(samples, time_steps, features)`。"
    elif "nan" in error_msg.lower() or "na" in error_msg.lower():
        extra_hint = "\n💡 提示：偵測到空值 (NaN)。請確保在訓練與預測前使用了 `.dropna()`，且對於深度學習模型，請務必進行特徵縮放 (Scaling)。"
    elif "n_samples=0" in error_msg or "train set will be empty" in error_msg:
        extra_hint = "\n💡 提示：資料清空了！這通常是因為特徵太多（特別是基本面）或預測天數太長，導致 `dropna()` 後沒有剩餘資料。請嘗試減少特徵（例如關閉基本面）或縮短預測距離。"
    elif "unexpected keyword argument" in error_msg.lower() and "evaluate_model" in error_msg.lower():
        extra_hint = "\n💡 提示：`evaluate_model` 函式只接受一個參數 `train_result`。請移除多餘的參數（如 `y_test=` 或 `X_test=`）。"
    elif "X_train" in error_msg or "y_train" in error_msg or "baseline_model" in error_msg:
        extra_hint = "\n💡 提示：回傳的 `train_res` 字典缺少了必要的鍵值（如 X_train, y_train 或 baseline_model）。請確保 `train_res` 包含 evaluate_model 函式所需的所有資訊。"

    user_prompt = f"""
{is_last_chance}
當前執行的程式碼：
```python
{code}
```

完整的 Error Traceback：
```text
{error_msg}
```
{extra_hint}

當前的資料狀態 (Data Context)：
```json
{context_str}
```

請根據以上資訊修復這段程式碼，確保它能成功執行。
"""
    
    # 使用較低溫度的 LLM 以獲得更穩定的修復
    response = call_llm(DEBUGGER_SYSTEM_PROMPT, user_prompt, temperature=0.1)
    
    if not response:
        return _fallback_static_repair(code, error_msg, attempt)
    
    # 提取程式碼塊
    code_match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
    if code_match:
        repaired_code = code_match.group(1).strip()
        return repaired_code, f"LLM-Traceback-Repair (Attempt {attempt})"
    
    return response.strip(), f"LLM-Traceback-Repair (No tags) (Attempt {attempt})"

def _fallback_static_repair(code: str, error_msg: str, attempt: int) -> tuple:
    """
    LLM 失敗時的靜態降級機制。
    強制回傳一個保證能執行的最簡版本。
    """
    logger.warning(f"LLM failed to respond during repair. Triggering hardcoded Safe-Mode fallback.")
    
    # 提取 ticker (從報錯程式碼中嘗試找尋)
    ticker_match = re.search(r'ticker = "(.*?)"', code)
    ticker = ticker_match.group(1) if ticker_match else "2330"
    
    new_code = f"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.loader import load_stock_data
from src.data.preprocess import build_target
from src.data.feature_engineering import build_features, get_feature_columns
from src.models.trainer import train_model
from src.models.evaluator import evaluate_model

def execute_research():
    # Safe-Mode: 降級為最穩定的線性模型與基本特徵
    ticker = "{ticker}"
    task_type = "regression"
    horizon = 5
    model_name = "linear"
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # 不使用基本面以防資料清空
    df_raw = load_stock_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    df_feat = build_features(df_raw, use_price=True, use_volume=True, use_technical=True, use_fundamental=False)
    feat_cols = get_feature_columns(df_feat, use_price=True, use_volume=True, use_technical=True, use_fundamental=False)
    df_target = build_target(df_feat, task_type, horizon)
    df_final = df_target.dropna(subset=feat_cols + ["target"])
    
    if len(df_final) < 20:
        # 如果資料太少，嘗試擴大範圍或回報失敗
        return {{"success": False, "error": "Insufficient data even in safe mode."}}

    train_res = train_model(df_final, feat_cols, task_type, model_name)
    eval_res = evaluate_model(train_res)
    
    return {{
        "success": True, "ticker": ticker, "task_type": task_type, "horizon": horizon,
        "model_name": model_name, "train_res": train_res, "eval_res": eval_res,
        "df_final": df_final, "feature_cols": feat_cols
    }}

result = execute_research()
"""
    return new_code, f"Safe-Mode Hardcoded Fallback (Attempt {attempt})"
