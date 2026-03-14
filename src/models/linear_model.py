"""
linear_model.py
Ridge regression and Logistic Regression wrappers.
"""

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_linear_model(task_type: str, alpha: float = 1.0):
    """
    Return a pipeline with StandardScaler + linear model.
    Linear models require feature scaling — handled here transparently.
    """
    if task_type == "regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha, fit_intercept=True)),
            ]
        )
    elif task_type == "classification":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(C=1.0 / alpha, max_iter=1000, random_state=42)),
            ]
        )
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
