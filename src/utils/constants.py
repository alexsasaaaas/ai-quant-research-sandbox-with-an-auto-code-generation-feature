"""
constants.py
Global constants for the application.
"""

# Taiwan Market specific
TW_COMMISSION_RATE = 0.001425  # 0.1425%
TW_TAX_RATE = 0.003            # 0.3%

# Application Defaults
DEFAULT_INITIAL_CAPITAL = 1_000_000
DEFAULT_TICKER = "2330.TW"
DEFAULT_HORIZON = 1

# File paths
CONFIG_DIR = "config"
DATA_DIR = "data"
ARTIFACTS_DIR = "artifacts"
PROMPTS_DIR = "prompts"

# Model types
MODEL_TYPES = ["baseline", "linear", "lightgbm", "xgboost"]
TASK_TYPES = ["regression", "classification"]
STRATEGY_TYPES = ["MA Cross", "RSI Mean Reversion", "Prediction-based"]
