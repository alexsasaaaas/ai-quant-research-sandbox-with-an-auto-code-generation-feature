"""
xgboost_model.py
XGBoost model wrapper with graceful fallback to sklearn GradientBoosting.
"""

import logging
logger = logging.getLogger(__name__)


def get_xgboost_model(task_type: str, params: dict = None):
    """
    Return an XGBoost model. Falls back to sklearn GradientBoosting if xgb not installed.
    """
    default_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbosity": 0,
    }
    if params:
        default_params.update(params)

    try:
        import xgboost as xgb
        if task_type == "regression":
            return xgb.XGBRegressor(**default_params)
        elif task_type == "classification":
            return xgb.XGBClassifier(**default_params)
    except ImportError:
        logger.warning("xgboost not installed. Falling back to GradientBoosting.")
        return _fallback_model(task_type, default_params)


def _fallback_model(task_type: str, params: dict):
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    fb = {
        "n_estimators": params.get("n_estimators", 100),
        "learning_rate": params.get("learning_rate", 0.1),
        "max_depth": min(params.get("max_depth", 6), 5),
        "random_state": params.get("random_state", 42),
    }
    if task_type == "regression":
        return GradientBoostingRegressor(**fb)
    return GradientBoostingClassifier(**fb)
