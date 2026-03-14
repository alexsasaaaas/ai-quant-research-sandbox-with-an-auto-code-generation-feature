"""
summary_builder.py
Builds a structured research summary JSON/dict from experiment results.
This serves as the canonical input to both LLM and template reporters.
"""

import json
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd


def build_research_summary(
    # Metadata
    ticker: str,
    start_date: str,
    end_date: str,
    task_type: str,
    horizon: int,
    model_name: str,
    # Data
    data_summary: dict,
    feature_cols: list[str],
    # Model results
    eval_result: dict,
    # Backtest results (optional)
    backtest_result: Optional[dict] = None,
) -> dict:
    """
    Build a structured research summary dict suitable for report generation.
    All float values are rounded for readability.
    """

    def safe_float(v):
        """Convert to float, handle numpy types, NaN → None."""
        if v is None:
            return None
        try:
            v = float(v)
            return round(v, 4) if not (v != v) else None  # NaN check
        except Exception:
            return None

    test_metrics = {k: safe_float(v) for k, v in eval_result["test_metrics"].items()}
    train_metrics = {k: safe_float(v) for k, v in eval_result["train_metrics"].items()}
    baseline_metrics = {k: safe_float(v) for k, v in eval_result["baseline_metrics"].items()}

    # Detect potential overfitting (train much better than test)
    overfitting_flag = False
    if task_type == "regression":
        train_r2 = train_metrics.get("r2", 0)
        test_r2 = test_metrics.get("r2", 0)
        if train_r2 is not None and test_r2 is not None:
            overfitting_flag = (train_r2 - test_r2) > 0.3
    else:
        train_acc = train_metrics.get("accuracy", 0)
        test_acc = test_metrics.get("accuracy", 0)
        if train_acc is not None and test_acc is not None:
            overfitting_flag = (train_acc - test_acc) > 0.15

    # Beat baseline check
    beat_baseline = False
    if task_type == "regression":
        model_rmse = test_metrics.get("rmse")
        base_rmse = baseline_metrics.get("rmse")
        if model_rmse and base_rmse:
            beat_baseline = model_rmse < base_rmse
    else:
        model_acc = test_metrics.get("accuracy")
        base_acc = baseline_metrics.get("accuracy")
        if model_acc and base_acc:
            beat_baseline = model_acc > base_acc

    summary = {
        "generated_at": datetime.now().isoformat(),
        "experiment_metadata": {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "task_type": task_type,
            "horizon_days": horizon,
            "model_name": model_name,
        },
        "data_summary": {
            "n_rows": data_summary.get("n_rows"),
            "start_date": data_summary.get("start_date"),
            "end_date": data_summary.get("end_date"),
            "close_start": safe_float(data_summary.get("close_start")),
            "close_end": safe_float(data_summary.get("close_end")),
            "total_return_pct": safe_float(data_summary.get("total_return_pct")),
            "missing_pct": safe_float(data_summary.get("missing_pct")),
        },
        "feature_summary": {
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
        },
        "model_metrics": {
            "test": test_metrics,
            "train": train_metrics,
            "baseline": baseline_metrics,
            "beat_baseline": beat_baseline,
            "overfitting_suspected": overfitting_flag,
        },
        "backtest_metrics": None,
        "risk_warnings": _generate_risk_warnings(
            data_summary, test_metrics, train_metrics, baseline_metrics,
            task_type, overfitting_flag, beat_baseline
        ),
        "final_observations": _generate_observations(
            task_type, model_name, beat_baseline, overfitting_flag,
            test_metrics, backtest_result
        ),
    }

    if backtest_result:
        m = backtest_result["metrics"]
        bh = backtest_result["bh_metrics"]
        summary["backtest_metrics"] = {
            "strategy": backtest_result["strategy_name"],
            "total_return_pct": safe_float(m.get("total_return_pct")),
            "annualized_return_pct": safe_float(m.get("annualized_return_pct")),
            "sharpe_ratio": safe_float(m.get("sharpe_ratio")),
            "max_drawdown_pct": safe_float(m.get("max_drawdown_pct")),
            "win_rate_pct": safe_float(m.get("win_rate_pct")),
            "n_trades": m.get("n_trades"),
            "final_equity": safe_float(m.get("final_equity")),
            "bh_total_return_pct": safe_float(bh.get("total_return_pct")),
            "bh_annualized_return_pct": safe_float(bh.get("annualized_return_pct")),
        }

    return summary


def _generate_risk_warnings(
    data_summary, test_metrics, train_metrics, baseline_metrics,
    task_type, overfitting_flag, beat_baseline
) -> list[str]:
    warnings = []
    n_rows = data_summary.get("n_rows", 0)

    if n_rows < 500:
        warnings.append("樣本量較小（少於 500 個交易日），模型的泛化能力可能有限。")
    elif n_rows < 1000:
        warnings.append("樣本量中等（不足 1000 個交易日），建議在更長時間跨度上驗證結果。")

    if overfitting_flag:
        warnings.append("訓練集表現明顯優於測試集，有過擬合（Overfitting）跡象，請謹慎解讀測試結果。")

    if not beat_baseline:
        warnings.append("模型表現未超越基線（Baseline），建議重新審視特徵工程或模型選擇。")

    if task_type == "regression":
        dir_acc = test_metrics.get("direction_accuracy", 0) or 0
        if dir_acc < 0.52:
            warnings.append(f"方向預測準確率 {dir_acc:.1%} 接近隨機水準，預測能力有限。")

    warnings.append("本工具僅供研究用途，所有結果均不構成投資建議，過去表現不代表未來結果。")
    warnings.append("回測未考量流動性風險、市場衝擊成本與極端市場事件。")
    return warnings


def _generate_observations(
    task_type, model_name, beat_baseline, overfitting_flag, test_metrics, backtest_result
) -> list[str]:
    obs = []

    if beat_baseline:
        obs.append(f"{model_name} 模型在測試集上超越了基線模型，顯示有一定的預測能力。")
    else:
        obs.append(f"{model_name} 模型在測試集上未能超越基線模型，可能需要更多特徵或調整超參數。")

    if task_type == "regression":
        da = test_metrics.get("direction_accuracy", 0) or 0
        obs.append(f"方向準確率 {da:.1%}：{'優於隨機，有一定參考價值。' if da > 0.55 else '接近隨機，需謹慎解讀。'}")

    if overfitting_flag:
        obs.append("建議嘗試增加正則化強度或減少特徵數量以緩解過擬合問題。")

    if backtest_result:
        ret = backtest_result["metrics"].get("total_return_pct", 0) or 0
        bh_ret = backtest_result["bh_metrics"].get("total_return_pct", 0) or 0
        if ret > bh_ret:
            obs.append(f"策略回測總報酬 {ret:.1f}% 優於買入持有 {bh_ret:.1f}%，但需注意回測偏差。")
        else:
            obs.append(f"策略回測總報酬 {ret:.1f}% 低於買入持有 {bh_ret:.1f}%，策略效果有限。")

    return obs


def summary_to_json(summary: dict) -> str:
    """Convert summary dict to formatted JSON string."""
    return json.dumps(summary, ensure_ascii=False, indent=2, default=str)
