"""Lightweight, configurable AI invocation helper.

This wrapper prefers OpenRouter (if `OPENROUTER_API_KEY` is set). If not,
it will attempt to use the OpenAI Python client when available. The function
accepts basic settings like `model` and `temperature` and always sends the
system+user messages in the same shape used elsewhere in the repo.

To use OpenRouter (recommended if you don't want to use OpenAI):
    export OPENROUTER_API_KEY="<your-openrouter-key>"

To use OpenAI (fallback):
  export OPENAI_API_KEY="<your-key>"
"""

import os
import json
from typing import Optional

import requests


def invoke_ai(system_message: str, user_message: str, *, model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    """Invoke an LLM and return the assistant content string.

    Uses OpenRouter by default when `OPENROUTER_API_KEY` is set. Falls back
    to OpenAI client if available and configured. Raises a helpful error
    if no provider is configured.

    Args:
        system_message: system prompt content
        user_message: user prompt content (string)
        model: model name (provider-specific). If omitted, a sensible default is used.
        temperature: sampling temperature
        max_tokens: maximum tokens to generate

    Returns:
        assistant text content (string)
    """

    # prefer OpenRouter (more provider choices and often free-tier alternatives)
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        url = "https://openrouter.ai/api/v1/chat/completions"
        hdr = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model or "mistralai/mistral-7b-instruct:free",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        resp = requests.post(url, headers=hdr, json=body, timeout=60)
        resp.raise_for_status()
        j = resp.json()
        # OpenRouter response shape: {choices: [{message: {role:..., content: ...}}], ...}
        try:
            return j["choices"][0]["message"]["content"]
        except Exception:
            # some versions return text directly in 'choices[0].message.content'
            return json.dumps(j)

    # fallback: try OpenAI python client if installed and configured
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=openai_key)
            model_name = model or "gpt-4o-mini"
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )
            return response.choices[0].message.content
        except Exception:
            # fall through to error below
            pass

    raise EnvironmentError("No LLM provider configured. Set OPENROUTER_API_KEY or OPENAI_API_KEY.")
