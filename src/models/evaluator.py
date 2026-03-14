"""
evaluator.py
Compute evaluation metrics for regression and classification tasks.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> dict:
    """
    Compute evaluation metrics based on task type.
    Returns a flat dict of metric_name -> value.
    """
    if task_type == "regression":
        return _regression_metrics(y_true, y_pred)
    elif task_type == "classification":
        return _classification_metrics(y_true, y_pred)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Direction accuracy: did we predict the sign of return correctly?
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred))

    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mae),
        "r2": float(r2),
        "direction_accuracy": float(dir_acc),
    }


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    # Use zero_division=0 to avoid warnings when a class is absent
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
    }


def evaluate_model(train_result: dict) -> dict:
    """
    Run evaluation for both the trained model and baseline on the test set.
    Returns structured dict with test metrics and baseline comparison.
    """
    model = train_result["model"]
    baseline = train_result["baseline_model"]
    X_test = train_result["X_test"]
    y_test = train_result["y_test"]
    X_train = train_result["X_train"]
    y_train = train_result["y_train"]
    task_type = train_result["task_type"]

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    y_pred_baseline = baseline.predict(X_test)

    test_metrics = compute_metrics(y_test, y_pred_test, task_type)
    train_metrics = compute_metrics(y_train, y_pred_train, task_type)
    baseline_metrics = compute_metrics(y_test, y_pred_baseline, task_type)

    return {
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "baseline_metrics": baseline_metrics,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "y_pred_baseline": y_pred_baseline,
        "test_dates": train_result["test_df"].index.tolist(),
    }


def get_feature_importance(model, feature_cols: list[str]) -> pd.DataFrame | None:
    """
    Extract feature importances if the model supports it.
    Returns a DataFrame sorted by importance (descending), or None.
    """
    importance = None

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            coef = np.abs(coef).mean(axis=0)
        importance = np.abs(coef)
    # Pipeline: check the last step
    elif hasattr(model, "steps"):
        last_step = model.steps[-1][1]
        if hasattr(last_step, "feature_importances_"):
            importance = last_step.feature_importances_
        elif hasattr(last_step, "coef_"):
            coef = last_step.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            importance = np.abs(coef)

    if importance is None:
        return None

    fi_df = pd.DataFrame(
        {"feature": feature_cols, "importance": importance}
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    return fi_df
