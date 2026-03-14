"""
validator.py
Walk-forward cross-validation for time series.
Strictly chronological — no data leakage.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def time_series_split_indices(
    n: int,
    n_splits: int = 5,
    test_size: float = 0.1,
) -> list[tuple]:
    """
    Generate (train_indices, test_indices) tuples for walk-forward validation.
    Each fold's test set is strictly after its train set.
    """
    fold_size = max(int(n * test_size), 20)  # at least 20 rows per test fold
    splits = []

    for i in range(n_splits, 0, -1):
        test_end = n - (i - 1) * fold_size
        test_start = test_end - fold_size
        train_end = test_start

        if train_end < 50:  # need enough training data
            continue

        splits.append((
            np.arange(0, train_end),
            np.arange(test_start, test_end),
        ))

    return splits


def walk_forward_validate(
    df: pd.DataFrame,
    feature_cols: list[str],
    task_type: str,
    model_factory,
    n_splits: int = 5,
) -> dict:
    """
    Run walk-forward validation. Returns mean metrics across folds.

    Args:
        model_factory: callable that returns a fresh unfitted model
    """
    from src.models.evaluator import compute_metrics

    cols_needed = feature_cols + ["target"]
    df_clean = df[cols_needed].dropna().copy()
    X = df_clean[feature_cols].values
    y = df_clean["target"].values

    splits = time_series_split_indices(len(X), n_splits=n_splits)
    fold_metrics = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])

        metrics = compute_metrics(y[test_idx], y_pred, task_type)
        metrics["fold"] = fold_i + 1
        metrics["train_size"] = len(train_idx)
        metrics["test_size"] = len(test_idx)
        fold_metrics.append(metrics)
        logger.debug(f"Fold {fold_i+1}: {metrics}")

    # Aggregate
    result_df = pd.DataFrame(fold_metrics)
    numeric_cols = result_df.select_dtypes(include=np.number).columns
    mean_metrics = result_df[numeric_cols].mean().to_dict()
    mean_metrics["n_folds"] = len(fold_metrics)
    mean_metrics["fold_details"] = fold_metrics

    return mean_metrics
