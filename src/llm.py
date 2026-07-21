"""Single factory for producing a chat model from config/settings.yaml.

Swapping provider (ollama <-> anthropic) is a config-only change; no
agent code should import a provider-specific class directly.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from src import llm_cache, telemetry
from src.config import LLMSettings, load_settings

T = TypeVar("T", bound=BaseModel)

# Cap concurrency on the fan-out so we don't hammer the LLM backend (Ollama
# cloud rate limits). Lower this if concurrent invokes start erroring.
_MAX_FANOUT_WORKERS = 8


def _build_llm(cfg: LLMSettings, json_mode: bool = False) -> BaseChatModel:
    if cfg.provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {"format": "json"} if json_mode else {}
        api_key = os.environ.get("OLLAMA_API_KEY")
        if api_key:
            kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {api_key}"}}
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
    # Telemetry: one record per call_structured, covering all retries. Recorded
    # in `finally` so timed-out or failed calls still show up in the UI.
    started_at = time.perf_counter()
    attempts = 0
    prompt_chars = 0
    response_chars = 0
    ok = False
    cache_hit = False
    cache_key: str | None = None
    if llm_cache.enabled():
        settings = load_settings()
        cache_key = llm_cache.cache_key(
            prompt=base_prompt,
            provider=settings.llm.provider,
            model=settings.llm.model,
            schema_name=model.__name__,
        )
    try:
        if cache_key is not None:
            cached = llm_cache.load(cache_key, model)
            if cached is not None:
                attempts = 1
                prompt_chars = len(base_prompt)
                response_chars = len(cached.model_dump_json())
                ok = True
                cache_hit = True
                return cached  # type: ignore[return-value]

        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            prompt_chars = len(attempt_prompt)
            response = llm.invoke(attempt_prompt)
            raw = response.content if hasattr(response, "content") else str(response)
            response_chars = len(raw)
            json_str = _extract_json(raw)
            try:
                if not json_str.strip():
                    raise ValueError("empty response")
                parsed = model.model_validate(json.loads(json_str))
                ok = True
                if cache_key is not None:
                    llm_cache.store(cache_key, parsed)
                return parsed
            except Exception as exc:  # noqa: BLE001 - retry loop, re-raised below if exhausted
                last_error = exc
                attempt_prompt = (
                    f"{base_prompt}\n\n"
                    f"Your previous response could not be parsed as valid JSON ({exc}). "
                    f"Previous response was:\n{raw}\n\nReturn ONLY the corrected JSON object."
                )

        raise StructuredCallError(
            f"LLM did not return parseable JSON after {max_retries + 1} attempts: {last_error}"
        )
    finally:
        telemetry.record_llm(
            schema=model.__name__,
            model=load_settings().llm.model,
            duration_s=time.perf_counter() - started_at,
            attempts=attempts,
            ok=ok,
            prompt_chars=prompt_chars,
            response_chars=response_chars,
            started_at=started_at,
            error=None if ok else (str(last_error) if last_error else "call failed"),
            cache_hit=cache_hit,
        )


def map_structured(jobs: list[Callable[[], T]]) -> list[T]:
    """Run independent LLM-producing callables concurrently, preserving order.

    Each job is any zero-arg callable that ultimately makes one or more
    `call_structured` calls (e.g. a `partial(run_gap_finder, ...)`). LLM calls
    are network-bound, so a thread pool gives real wall-clock parallelism while
    keeping the graph nodes synchronous.

    Each worker re-binds the enclosing telemetry node (see telemetry.bound_to)
    so the fanned-out LLM calls attribute to the right node instead of
    "(hors nœud)". `ThreadPoolExecutor.map` preserves input order, so callers
    can zip results back against their inputs deterministically.
    """
    if not jobs:
        return []
    run = telemetry.current_run()

    def _worker(job: Callable[[], T]) -> T:
        with telemetry.bound_to(run):
            return job()

    with ThreadPoolExecutor(max_workers=min(_MAX_FANOUT_WORKERS, len(jobs))) as ex:
        return list(ex.map(_worker, jobs))
