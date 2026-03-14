"""
llm_reporter.py
LLM-powered report generation. Falls back gracefully to template if LLM is unavailable.
Supports Groq, OpenAI, and Anthropic providers.
"""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Load the prompt template
_PROMPT_PATH = "prompts/report_prompt.txt"


def _load_prompt_template() -> str:
    try:
        with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return _DEFAULT_PROMPT


_DEFAULT_PROMPT = """你是一位專業但謹慎的量化研究助理，請根據我提供的研究結果 JSON，撰寫一份對非技術使用者也清楚易懂的研究報告。

請遵守以下原則：
1. 不要誇大模型能力
2. 不要把結果寫成投資建議
3. 要明確指出限制、風險與可能的偏差
4. 若模型或策略表現不佳，要誠實說明
5. 用清楚、有條理的中文撰寫
6. 遇到指標時，要用一句白話解釋它的意義

請輸出以下章節：
1. 執行摘要
2. 研究設定
3. 模型結果分析
4. 策略回測分析
5. 風險與限制
6. 結論與後續建議

以下是研究結果 JSON：
{{research_summary_json}}"""


def detect_llm_provider() -> Optional[str]:
    """Check which LLM provider is available based on environment variables."""
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def generate_llm_report(summary: dict) -> Optional[str]:
    """
    Call LLM to generate a natural language research report.
    Returns None if LLM is unavailable or errors out.
    """
    provider = detect_llm_provider()
    if not provider:
        return None

    from src.report.summary_builder import summary_to_json
    summary_json = summary_to_json(summary)
    prompt_template = _load_prompt_template()
    prompt = prompt_template.replace("{{research_summary_json}}", summary_json)

    try:
        if provider == "groq":
            return _call_groq(prompt)
        elif provider == "openai":
            return _call_openai(prompt)
        elif provider == "anthropic":
            return _call_anthropic(prompt)
    except Exception as e:
        logger.warning(f"LLM call failed ({provider}): {e}. Falling back to template report.")
        return None


def _call_groq(prompt: str) -> str:
    import groq
    client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是一位謹慎、誠實的量化研究助理，不誇大能力，不輕易給出投資建議。",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=1200,
    )
    return response.choices[0].message.content


def _call_openai(prompt: str) -> str:
    import openai
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是一位謹慎、誠實的量化研究助理，不誇大能力，不輕易給出投資建議。",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=4000,
    )
    return response.choices[0].message.content


def _call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    model = os.getenv("LLM_MODEL", "claude-3-haiku-20240307")

    response = client.messages.create(
        model=model,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
