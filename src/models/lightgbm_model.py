"""
lightgbm_model.py
LightGBM model wrapper with graceful fallback to sklearn GradientBoosting.
"""

import logging
logger = logging.getLogger(__name__)


def get_lightgbm_model(task_type: str, params: dict = None):
    """
    Return a LightGBM model. Falls back to sklearn GradientBoosting if lgbm not installed.
    """
    default_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    try:
        import lightgbm as lgb
        if task_type == "regression":
            return lgb.LGBMRegressor(**default_params)
        elif task_type == "classification":
            return lgb.LGBMClassifier(**default_params)
    except ImportError:
        logger.warning("lightgbm not installed. Falling back to GradientBoosting.")
        return _fallback_model(task_type, default_params)


def _fallback_model(task_type: str, params: dict):
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    fb = {
        "n_estimators": params.get("n_estimators", 100),
        "learning_rate": params.get("learning_rate", 0.1),
        "max_depth": max(params.get("max_depth", 3), 3),
        "random_state": params.get("random_state", 42),
    }
    if task_type == "regression":
        return GradientBoostingRegressor(**fb)
    return GradientBoostingClassifier(**fb)
