"""
template_reporter.py
Generates readable Markdown research reports without LLM.
High-quality, well-structured reports for non-technical users.
"""

from datetime import datetime
from typing import Optional


def generate_template_report(summary: dict) -> str:
    """
    Generate a comprehensive Markdown report from the research summary dict.
    Designed to be readable by non-technical users.
    """
    meta = summary.get("experiment_metadata", {})
    data = summary.get("data_summary", {})
    feat = summary.get("feature_summary", {})
    model_info = summary.get("model_metrics", {})
    bt = summary.get("backtest_metrics")
    warnings = summary.get("risk_warnings", [])
    observations = summary.get("final_observations", [])

    test_m = model_info.get("test", {})
    train_m = model_info.get("train", {})
    base_m = model_info.get("baseline", {})
    task_type = meta.get("task_type", "regression")

    lines = []

    # ── Header ───────────────────────────────────────────────────────────────
    lines += [
        "# AI 量化研究報告",
        "",
        f"> 🏷️ **股票**: {meta.get('ticker', 'N/A')}  ",
        f"> 📅 **研究期間**: {meta.get('start_date', 'N/A')} ～ {meta.get('end_date', 'N/A')}  ",
        f"> 🤖 **模型**: {meta.get('model_name', 'N/A')} | **任務**: {task_type} | **預測天數**: {meta.get('horizon_days', 'N/A')} 天  ",
        f"> 🕒 **報告生成時間**: {summary.get('generated_at', datetime.now().isoformat())[:19]}",
        "",
        "> ⚠️ **免責聲明**：本報告僅供研究用途，所有結果均不構成投資建議，不保證獲利。請自行承擔投資風險。",
        "",
        "---",
        "",
    ]

    # ── 1. Executive Summary ──────────────────────────────────────────────────
    lines += [
        "## 1. 執行摘要",
        "",
    ]
    beat = model_info.get("beat_baseline", False)
    overfit = model_info.get("overfitting_suspected", False)

    if observations:
        for obs in observations:
            lines.append(f"- {obs}")
    else:
        if beat:
            lines.append(f"- `{meta.get('model_name')}` 模型在測試集上超越了基線（Baseline）模型。")
        else:
            lines.append(f"- `{meta.get('model_name')}` 模型在測試集上**未能**超越基線，需要進一步改善。")
        if overfit:
            lines.append("- ⚠️ 偵測到潛在的過擬合跡象，請謹慎解讀指標。")

    lines += ["", "---", ""]

    # ── 2. Research Setup ─────────────────────────────────────────────────────
    lines += [
        "## 2. 研究設定",
        "",
        "### 資料概況",
        f"| 項目 | 數值 |",
        f"|------|------|",
        f"| 交易日數量 | {data.get('n_rows', 'N/A')} 天 |",
        f"| 資料起始日 | {data.get('start_date', 'N/A')} |",
        f"| 資料結束日 | {data.get('end_date', 'N/A')} |",
        f"| 期初收盤價 | {_fmt_price(data.get('close_start'))} 元 |",
        f"| 期末收盤價 | {_fmt_price(data.get('close_end'))} 元 |",
        f"| 期間總報酬 | {_fmt_pct(data.get('total_return_pct'))} |",
        "",
        "### 特徵工程",
        f"- 共使用 **{feat.get('n_features', 0)} 個特徵**",
        f"- 特徵列表：`{'`, `'.join(feat.get('feature_names', [])[:10])}` {'...' if feat.get('n_features', 0) > 10 else ''}",
        "",
        "---",
        "",
    ]

    # ── 3. Model Results ──────────────────────────────────────────────────────
    lines += [
        "## 3. 模型結果分析",
        "",
    ]

    if task_type == "regression":
        lines += [
            "### 迴歸指標說明",
            "- **RMSE**（均方根誤差）：越小越好，代表預測值與真實值的平均差距",
            "- **MAE**（平均絕對誤差）：直觀的平均預測誤差",
            "- **R²**（決定係數）：越接近 1 越好，代表模型解釋了多少變異",
            "- **方向準確率**：預測漲/跌方向的正確比例",
            "",
            "| 指標 | 訓練集 | 測試集 | 基線（均值預測） |",
            "|------|--------|--------|----------------|",
            f"| RMSE | {_fmt(train_m.get('rmse'))} | {_fmt(test_m.get('rmse'))} | {_fmt(base_m.get('rmse'))} |",
            f"| MAE  | {_fmt(train_m.get('mae'))} | {_fmt(test_m.get('mae'))} | {_fmt(base_m.get('mae'))} |",
            f"| R²   | {_fmt(train_m.get('r2'))} | {_fmt(test_m.get('r2'))} | {_fmt(base_m.get('r2'))} |",
            f"| 方向準確率 | {_fmt_pct(train_m.get('direction_accuracy', 0) * 100 if train_m.get('direction_accuracy') else None)} | {_fmt_pct(test_m.get('direction_accuracy', 0) * 100 if test_m.get('direction_accuracy') else None)} | {_fmt_pct(base_m.get('direction_accuracy', 0) * 100 if base_m.get('direction_accuracy') else None)} |",
        ]
    else:
        lines += [
            "### 分類指標說明",
            "- **準確率（Accuracy）**：預測正確的比例",
            "- **精確率（Precision）**：預測為「上漲」中真的上漲的比例",
            "- **召回率（Recall）**：真實上漲中被正確預測的比例",
            "- **F1 Score**：精確率與召回率的調和平均",
            "",
            "| 指標 | 訓練集 | 測試集 | 基線（多數類別） |",
            "|------|--------|--------|----------------|",
            f"| 準確率 | {_fmt_pct(train_m.get('accuracy', 0) * 100 if train_m.get('accuracy') else None)} | {_fmt_pct(test_m.get('accuracy', 0) * 100 if test_m.get('accuracy') else None)} | {_fmt_pct(base_m.get('accuracy', 0) * 100 if base_m.get('accuracy') else None)} |",
            f"| 精確率 | {_fmt_pct(train_m.get('precision', 0) * 100 if train_m.get('precision') else None)} | {_fmt_pct(test_m.get('precision', 0) * 100 if test_m.get('precision') else None)} | — |",
            f"| 召回率 | {_fmt_pct(train_m.get('recall', 0) * 100 if train_m.get('recall') else None)} | {_fmt_pct(test_m.get('recall', 0) * 100 if test_m.get('recall') else None)} | — |",
            f"| F1 Score | {_fmt(train_m.get('f1_score'))} | {_fmt(test_m.get('f1_score'))} | — |",
        ]

    if overfit:
        lines += [
            "",
            "> ⚠️ **過擬合警告**：訓練集表現明顯優於測試集，模型可能對歷史資料過度擬合，在未見資料上的預測能力可能有限。",
        ]

    lines += ["", "---", ""]

    # ── 4. Backtest ───────────────────────────────────────────────────────────
    if bt:
        lines += [
            "## 4. 策略回測分析",
            "",
            f"策略名稱：**{bt.get('strategy', 'N/A')}**",
            "",
            "| 指標 | 策略 | 買入持有 |",
            "|------|------|---------|",
            f"| 總報酬 | {_fmt_pct(bt.get('total_return_pct'))} | {_fmt_pct(bt.get('bh_total_return_pct'))} |",
            f"| 年化報酬 | {_fmt_pct(bt.get('annualized_return_pct'))} | {_fmt_pct(bt.get('bh_annualized_return_pct'))} |",
            f"| 夏普比率 | {_fmt(bt.get('sharpe_ratio'))} | — |",
            f"| 最大回撤 | {_fmt_pct(bt.get('max_drawdown_pct'))} | — |",
            f"| 勝率 | {_fmt_pct(bt.get('win_rate_pct'))} | — |",
            f"| 交易次數 | {bt.get('n_trades', 'N/A')} | — |",
            "",
            _bt_interpretation(bt),
            "",
            "---",
            "",
        ]
    else:
        lines += [
            "## 4. 策略回測分析",
            "",
            "_本次研究未執行回測。_",
            "",
            "---",
            "",
        ]

    # ── 5. Risk & Limitations ─────────────────────────────────────────────────
    lines += [
        "## 5. 風險與限制",
        "",
    ]
    for w in warnings:
        lines.append(f"- ⚠️ {w}")
    lines += ["", "---", ""]

    # ── 6. Conclusion ─────────────────────────────────────────────────────────
    lines += [
        "## 6. 結論與後續建議",
        "",
        "### 本次研究結論",
    ]
    for obs in observations:
        lines.append(f"- {obs}")

    lines += [
        "",
        "### 建議的後續步驟",
        "1. 嘗試不同的特徵組合，觀察哪些技術指標對此股票最有預測性",
        "2. 調整模型超參數以改善泛化能力",
        "3. 在更長的時間跨度上測試策略的穩健性",
        "4. 考慮加入基本面指標或總經環境變數",
        "5. 任何研究結果在轉化為實際操作前，務必經過嚴格的樣本外測試",
        "",
        "---",
        "",
        "_本報告由 AI Quant Research Sandbox 自動生成，僅供研究學習用途。_",
        "_⚠️ 非投資建議 · 不保證獲利 · 投資一定有風險_",
    ]

    return "\n".join(lines)


# ── Formatting Helpers ────────────────────────────────────────────────────────

def _fmt(v, decimals: int = 4) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return "N/A"


def _fmt_pct(v) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.2f}%"
    except Exception:
        return "N/A"


def _fmt_price(v) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):,.1f}"
    except Exception:
        return "N/A"


def _bt_interpretation(bt: dict) -> str:
    """Plain-language interpretation of backtest results."""
    total_ret = bt.get("total_return_pct") or 0
    bh_ret = bt.get("bh_total_return_pct") or 0
    sharpe = bt.get("sharpe_ratio") or 0
    maxdd = bt.get("max_drawdown_pct") or 0
    win_rate = bt.get("win_rate_pct") or 0

    parts = []
    if total_ret > bh_ret:
        parts.append(f"策略報酬（{total_ret:.1f}%）優於買入持有（{bh_ret:.1f}%）。")
    else:
        parts.append(f"策略報酬（{total_ret:.1f}%）**低於**買入持有（{bh_ret:.1f}%），策略效益有待提升。")

    if sharpe > 1.0:
        parts.append(f"夏普比率 {sharpe:.2f} 表現良好（>1）。")
    elif sharpe > 0.5:
        parts.append(f"夏普比率 {sharpe:.2f} 尚可（0.5~1）。")
    else:
        parts.append(f"夏普比率 {sharpe:.2f} 偏低，表示每單位風險的報酬不理想。")

    if abs(maxdd) > 20:
        parts.append(f"⚠️ 最大回撤 {maxdd:.1f}% 相當顯著，代表策略在某段期間曾大幅虧損。")

    return " ".join(parts)
