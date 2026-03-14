"""
prompt_parser.py
Uses LLM to generate custom modeling code based on user prompts.
"""

import re
import logging
from src.agent.llm_client import call_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一位專業的台股量化研究員與 Python 工程師。
你的任務是根據使用者的需求，編寫一段 Python 程式碼來建立並訓練一個預測模型。

你必須遵循以下規範：
1. **程式碼結構**：定義一個 `execute_research()` 函式。
2. **依賴模組**：
   - 你**必須**自行導入所有需要的標準庫 (如 `import os`, `import sys`, `import re`)。
   - 你可以且應使用專案中已有的模組：
   - `from src.data.loader import load_stock_data` -> `load_stock_data(ticker, start_date, end_date)`
   - `src.data.loader.load_stock_data(ticker, start_date, end_date)`
   - `from src.data.feature_engineering import build_features, get_feature_columns` -> `build_features(df, use_price, use_volume, use_technical, use_fundamental)`, `get_feature_columns(df, use_price, use_volume, use_technical, use_fundamental)`
   - `src.data.feature_engineering.build_features(df, use_price, use_volume, use_technical, use_fundamental)`
   - `from src.data.preprocess import build_target` -> `build_target(df, task_type, horizon)`
   - `from src.models.trainer import train_model` -> `train_model(df, feature_cols, task_type, model_name)`
   - `src.models.trainer.train_model(df, feature_cols, task_type, model_name)`
   - `from src.models.evaluator import evaluate_model` -> `evaluate_model(train_result)`
   - **注意**：`evaluate_model` **僅接受一個參數**：即 `train_result` 字典。千萬不要傳入額外的 keyword arguments (如 y_test=...)。
3. **處理缺失庫 (Dependency Issues)**：如果 Traceback 顯示 `ModuleNotFoundError` 或指出某個庫 (如 `tensorflow`, `keras`, `torch`) 未安裝，你可以嘗試在修復後的程式碼中先調用 `install_package("庫名稱")`（例如：`install_package("tensorflow")`）。
4. **驗證庫簽名**：如果錯誤涉及參數缺失 (missing argument)，請參考提供的 Data Context 並修正簽名。
   - 如果使用者要求的是內建模型 (lightgbm, xgboost, linear, baseline)，請使用 `train_model` 函式。
   - 如果使用者要求的是自定義模型 (如 LSTM, RNN, CNN, 或特定論文中的結構)，請在程式碼中自行實作模型類別與訓練邏輯。
   - **重要：自定義模型必須包含資料標準化 (Scaling)**：深度學習模型對數值敏感，請務必使用 `StandardScaler` 或 `MinMaxScaler` 對特徵進行處理。
   - **重要：嚴格處理空值**：在使用 `dropna()` 後，請檢查資料長度是否足夠訓練。
   - 請確保資料處理部分仍使用專案模組以確保格式相容。
5. **傳回值**：函式最後必須回傳一個 dict，包含：
   - `success`: bool
   - `ticker`: str
   - `task_type`: 'regression' 或 'classification'
   - `horizon`: int (天數)
   - `model_name`: str (模型簡稱)
   - `train_res`: dict (必須包含 'model', 'baseline_model', 'X_train', 'X_test', 'y_train', 'y_test', 'test_df', 'feature_cols', 'task_type')
   - `eval_res`: dict (必須由 evaluate_model 產生，包含 'test_metrics', 'y_test', 'y_pred_test', 'test_dates' 等子字典。絕對不能只回傳一個分數或 float)
   - **防錯提醒**：嚴禁將 `eval_res` 設為單一數字。如果你手動計算指標，也必須包裝成 `{ 'test_metrics': { 'rmse': ... }, 'y_test': ..., ... }` 的格式。
   - `df_final`: pd.DataFrame (過濾後的最終資料)
   - `feature_cols`: list[str] (特徵清單)
6. **資料處理**：
   - 資料下載：如果使用者沒指定期間，預設抓取過去 1 年資料。
   - `train_model` 預期 `df_final` 包含特徵欄位與名為 `target` 的目標欄位。
   - **特徵預設**：如果使用者未說明要使用哪些特徵，請預設使用 `price`, `volume`, `technical`。**注意：基本面 (fundamental) 資料通常較稀疏且頻率低，若資料區間短 (如1年) 容易導致 dropna() 後資料清空，請謹慎使用。**
   - **預測值檢查**：確保 `y_pred_test` 中沒有 NaN 或 Inf，否則評估指標會變成 N/A。
7. **僅輸出 Markdown 程式碼塊**：不要有額外解釋，只輸出程式碼內容。

範例結構：
```python
import pandas as pd
import numpy as np
# ... imports ...

def execute_research():
    ticker = "2330" 
    # ... logic ...
    return { ... }

result = execute_research()
```
"""

def generate_modeling_code(prompt: str) -> str:
    """
    使用 LLM 解析需求並生成 Python 程式碼。
    """
    logger.info(f"Generating code for prompt: {prompt}")
    
    # 調用 LLM
    response = call_llm(SYSTEM_PROMPT, f"使用者需求：{prompt}", temperature=0.2)
    
    if not response:
        # Fallback to a basic template if LLM fails
        return _fallback_static_code(prompt)
    
    # 提取程式碼塊
    code_match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # 如果沒有 code block 標籤，則嘗試直接回傳
    return response.strip()

def _fallback_static_code(prompt: str) -> str:
    """
    LLM 失敗時的靜態降級機制。
    """
    prompt_lower = prompt.lower()
    stock_id = "2330"
    # The following lines are syntactically incorrect as they are placed directly after the docstring.
    # Assuming they were intended to be part of an if/elif chain that processes an error_msg,
    # but error_msg is not defined in this function.
    # To faithfully apply the change as given, they are inserted as provided,
    # which will result in a syntax error.
    stock_match = re.search(r'(\d{4})', prompt)
    if stock_match:
        stock_id = stock_match.group(1)

    return f"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.loader import load_stock_data
from src.data.preprocess import build_target
from src.data.feature_engineering import build_features, get_feature_columns
from src.models.trainer import train_model
from src.models.evaluator import evaluate_model

def execute_research():
    ticker = "{stock_id}"
    task_type = "regression"
    horizon = 5
    model_name = "linear"
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    df_raw = load_stock_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    df_feat = build_features(df_raw, use_price=True, use_volume=True, use_technical=True, use_fundamental=False)
    feat_cols = get_feature_columns(df_feat, use_price=True, use_volume=True, use_technical=True, use_fundamental=False)
    df_target = build_target(df_feat, task_type, horizon)
    df_final = df_target.dropna(subset=feat_cols + ["target"])
    
    train_res = train_model(df_final, feat_cols, task_type, model_name)
    eval_res = evaluate_model(train_res)
    
    return {{
        "success": True, "ticker": ticker, "task_type": task_type, "horizon": horizon,
        "model_name": model_name, "train_res": train_res, "eval_res": eval_res,
        "df_final": df_final, "feature_cols": feat_cols
    }}

result = execute_research()
"""
