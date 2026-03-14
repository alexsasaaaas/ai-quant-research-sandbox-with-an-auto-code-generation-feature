"""
exporter.py
Unified report generation interface: generate_report(summary, mode="auto").
Auto-detects LLM availability and falls back to template.
Also handles saving reports to disk.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def generate_report(summary: dict, mode: str = "auto") -> tuple[str, str]:
    """
    Unified report generation interface.

    Args:
        summary: research summary dict from summary_builder.build_research_summary()
        mode:    'auto' | 'llm' | 'template'
                 'auto' = try LLM first, fall back to template

    Returns:
        (report_text: str, mode_used: str)
    """
    from src.report.llm_reporter import generate_llm_report, detect_llm_provider
    from src.report.template_reporter import generate_template_report

    if mode == "llm":
        report = generate_llm_report(summary)
        if report:
            return report, "llm"
        logger.warning("LLM not available, falling back to template.")
        return generate_template_report(summary), "template"

    elif mode == "template":
        return generate_template_report(summary), "template"

    else:  # auto
        provider = detect_llm_provider()
        if provider:
            logger.info(f"LLM provider detected: {provider}. Using LLM report.")
            report = generate_llm_report(summary)
            if report:
                return report, "llm"
        logger.info("No LLM available. Using template report.")
        return generate_template_report(summary), "template"


def save_report(
    report_text: str,
    ticker: str,
    output_dir: str = "artifacts/reports",
    format: str = "md",
) -> str:
    """
    Save report to disk. Returns the file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_ticker = ticker.replace(".", "_")
    filename = f"report_{safe_ticker}_{timestamp}.{format}"
    filepath = output_dir / filename

    if format == "html":
        html_content = _md_to_html(report_text, ticker)
        filepath.write_text(html_content, encoding="utf-8")
    else:
        filepath.write_text(report_text, encoding="utf-8")

    logger.info(f"Report saved to {filepath}")
    return str(filepath)


def _md_to_html(md_text: str, ticker: str) -> str:
    """
    Convert Markdown report to a simple styled HTML page.
    Uses basic regex replacement — no external dependencies.
    """
    try:
        import markdown
        body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    except ImportError:
        # Minimal fallback: wrap in <pre> if markdown library not available
        body = f"<pre>{md_text}</pre>"

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI 量化研究報告 - {ticker}</title>
<style>
  body {{ font-family: 'Microsoft JhengHei', 'Noto Sans TC', sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; line-height: 1.7; color: #2d3748; }}
  h1 {{ color: #1a365d; border-bottom: 3px solid #3182ce; padding-bottom: 10px; }}
  h2 {{ color: #2b6cb0; border-left: 4px solid #3182ce; padding-left: 12px; margin-top: 40px; }}
  h3 {{ color: #2c5282; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th {{ background: #ebf8ff; color: #2b6cb0; padding: 10px 14px; text-align: left; }}
  td {{ padding: 8px 14px; border-bottom: 1px solid #e2e8f0; }}
  tr:hover {{ background: #f7fafc; }}
  blockquote {{ background: #fff3cd; border-left: 4px solid #f6ad55; padding: 12px 16px; margin: 16px 0; }}
  code {{ background: #edf2f7; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
  hr {{ border: none; border-top: 1px solid #e2e8f0; margin: 32px 0; }}
  .footer {{ text-align: center; color: #a0aec0; font-size: 0.85em; margin-top: 60px; }}
</style>
</head>
<body>
{body}
<div class="footer">⚠️ 本報告僅供研究用途，非投資建議 · AI Quant Research Sandbox</div>
</body>
</html>"""
    return html
