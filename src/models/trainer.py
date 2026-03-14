"""
trainer.py
Model training pipeline: selects model, handles NaN dropping, fits, returns results.
"""

import logging
import numpy as np
import pandas as pd

from src.models.baseline import get_baseline_model
from src.models.linear_model import get_linear_model
from src.models.lightgbm_model import get_lightgbm_model
from src.models.xgboost_model import get_xgboost_model

logger = logging.getLogger(__name__)


def get_model(model_name: str, task_type: str, model_params: dict = None):
    """Factory function to return the requested model."""
    model_map = {
        "baseline": lambda: get_baseline_model(task_type),
        "linear":   lambda: get_linear_model(task_type),
        "lightgbm": lambda: get_lightgbm_model(task_type, model_params),
        "xgboost":  lambda: get_xgboost_model(task_type, model_params),
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_map.keys())}")
    return model_map[model_name]()


def train_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    task_type: str,
    model_name: str,
    test_ratio: float = 0.2,
    model_params: dict = None,
) -> dict:
    """
    Full training pipeline:
    1. Drop rows with NaN in features or target
    2. Chronological train/test split
    3. Fit baseline + selected model on train
    4. Return both models and split data

    Returns a dict with keys:
        model, baseline_model, X_train, X_test, y_train, y_test, train_df, test_df
    """
    # --- Drop NaN rows ---
    cols_needed = feature_cols + ["target"]
    df_clean = df[cols_needed].dropna().copy()
    
    if len(df_clean) < 10:
        raise ValueError(
            f"Insufficient data: After dropping NaNs (due to indicators/horizon), only {len(df_clean)} rows remain. "
            "Please try a longer date range or fewer technical indicators (like SMA 60)."
        )
    
    logger.info(f"After dropping NaN: {len(df_clean)} rows (from {len(df)})")

    # --- Chronological split ---
    split_idx = int(len(df_clean) * (1 - test_ratio))
    
    if split_idx < 2:
        raise ValueError(
            f"Training set too small: split_idx={split_idx}. Not enough data for training. "
            "Suggest increasing the date range."
        )

    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["target"].values

    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # --- Fit selected model ---
    model = get_model(model_name, task_type, model_params)
    model.fit(X_train, y_train)
    logger.info(f"Trained [{model_name}] for task=[{task_type}]")

    # --- Fit baseline ---
    baseline = get_baseline_model(task_type)
    baseline.fit(X_train, y_train)

    return {
        "model": model,
        "model_name": model_name,
        "baseline_model": baseline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_df": train_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "task_type": task_type,
    }
