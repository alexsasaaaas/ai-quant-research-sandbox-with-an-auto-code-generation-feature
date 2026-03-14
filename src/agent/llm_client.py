"""
llm_client.py
LLM interface for the Agent Model Builder.
Handles code generation and repair requests.
"""

import os
import logging
import json
from typing import Optional

logger = logging.getLogger(__name__)

def detect_llm_provider() -> Optional[str]:
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None

def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> Optional[str]:
    provider = detect_llm_provider()
    if not provider:
        return None

    try:
        if provider == "groq":
            import groq
            client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
            model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content
            
        elif provider == "openai":
            import openai
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model = os.getenv("LLM_MODEL", "gpt-4o-mini")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content
            
        elif provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            model = os.getenv("LLM_MODEL", "claude-3-haiku-20240307")
            response = client.messages.create(
                model=model,
                max_tokens=4000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
            )
            return response.content[0].text
            
    except Exception as e:
        logger.error(f"LLM call failed ({provider}): {e}")
        return None
    return None
