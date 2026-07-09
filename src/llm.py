"""Single factory for producing a chat model from config/settings.yaml.

Swapping provider (ollama <-> anthropic) is a config-only change; no
agent code should import a provider-specific class directly.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from src.config import LLMSettings, load_settings

T = TypeVar("T", bound=BaseModel)


def _build_llm(cfg: LLMSettings, json_mode: bool = False) -> BaseChatModel:
    if cfg.provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {"format": "json"} if json_mode else {}
        return ChatOllama(model=cfg.model, base_url=cfg.base_url, temperature=cfg.temperature, **kwargs)
    if cfg.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=cfg.model, temperature=cfg.temperature)
    raise ValueError(f"Unknown llm provider: {cfg.provider!r}")


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    settings = load_settings()
    return _build_llm(settings.llm)


@lru_cache(maxsize=1)
def _get_structured_llm() -> BaseChatModel:
    """A JSON-mode variant used only by call_structured, where supported (Ollama)."""
    settings = load_settings()
    if settings.llm.provider == "ollama":
        return _build_llm(settings.llm, json_mode=True)
    return get_llm()


_JSON_BLOCK_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```\s*$", "", text)
    match = _JSON_BLOCK_RE.search(text)
    return match.group(0) if match else text


class StructuredCallError(RuntimeError):
    pass


def call_structured(prompt: str, model: type[T], llm: BaseChatModel | None = None, max_retries: int = 2) -> T:
    """Ask the LLM for JSON matching `model`'s schema and parse it robustly.

    Works across providers (small local Ollama models rarely support reliable
    native function-calling) by instructing JSON output, using Ollama's JSON
    mode where available, repairing common formatting issues (code fences,
    leading prose), and retrying on parse failure with the error fed back to
    the model.
    """
    llm = llm or _get_structured_llm()
    schema = model.model_json_schema()
    base_prompt = (
        f"{prompt}\n\n"
        "Respond with ONLY a single JSON object matching this JSON schema, "
        "no prose, no markdown code fences:\n"
        f"{json.dumps(schema)}"
    )

    last_error: Exception | None = None
    attempt_prompt = base_prompt
    for attempt in range(max_retries + 1):
        response = llm.invoke(attempt_prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        json_str = _extract_json(raw)
        try:
            if not json_str.strip():
                raise ValueError("empty response")
            return model.model_validate(json.loads(json_str))
        except Exception as exc:  # noqa: BLE001 - retry loop, re-raised below if exhausted
            last_error = exc
            attempt_prompt = (
                f"{base_prompt}\n\n"
                f"Your previous response could not be parsed as valid JSON ({exc}). "
                f"Previous response was:\n{raw}\n\nReturn ONLY the corrected JSON object."
            )

    raise StructuredCallError(f"LLM did not return parseable JSON after {max_retries + 1} attempts: {last_error}")
