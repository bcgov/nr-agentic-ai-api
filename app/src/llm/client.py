from __future__ import annotations
import json
import logging
import os
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI

from app.src.memory.session_store import _env_flag

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

USE_GPT4O = _env_flag("USE_GPT4O")
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "600"))
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0"))

_client: Optional[AzureOpenAI] = None

logger = logging.getLogger(__name__)


def _client_or_none() -> Optional[AzureOpenAI]:
    """Return an Azure OpenAI client if configured+enabled; otherwise None."""
    global _client
    if not USE_GPT4O:
        return None
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        return None
    if _client is None:
        _client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2024-08-01-preview",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )
    return _client


def json_complete(messages: List[Dict[str, Any]], response_format: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Call GPT-4o and return a parsed JSON object.
    If 'response_format' is provided, it is used with json_schema mode.
    When disabled or misconfigured, returns a No-Op structure.
    """
    cli = _client_or_none()
    if cli is None:
        return {"updatedFields": {}, "clarifications": []}

    kwargs: Dict[str, Any] = {
        "model": AZURE_OPENAI_DEPLOYMENT,
        "messages": messages,
        "temperature": AGENT_TEMPERATURE,
        "max_tokens": AGENT_MAX_TOKENS,
    }
    if response_format:
        kwargs["response_format"] = {"type": "json_schema", "json_schema": response_format}

    try:
        resp = cli.chat.completions.create(**kwargs)
    except Exception as exc:
        logger.warning("Azure OpenAI chat completion failed: %s", exc)
        return {"updatedFields": {}, "clarifications": []}

    content = resp.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except Exception:
        # If model didn't strictly follow JSON, hand back raw content for debugging
        return {"updatedFields": {}, "clarifications": [], "_raw": content}
