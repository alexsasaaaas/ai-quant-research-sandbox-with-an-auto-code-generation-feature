"""
baseline.py
Baseline models: DummyRegressor (mean prediction) and DummyClassifier (majority class).
Used as comparison benchmarks for all trained models.
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor, DummyClassifier


def get_baseline_model(task_type: str):
    """
    Return an sklearn-compatible baseline model.
    - regression:    predicts the training mean
    - classification: predicts the majority class
    """
    if task_type == "regression":
        return DummyRegressor(strategy="mean")
    elif task_type == "classification":
        return DummyClassifier(strategy="most_frequent")
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
