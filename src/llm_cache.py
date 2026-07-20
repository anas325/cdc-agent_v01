"""Content-addressed disk cache for structured LLM calls.

Purpose is the dev loop: iterating on the same CDC costs 5+ minutes of LLM
latency before the first question batch appears (one gap-finder call per
section in initial_scan, then ~9 calls in the first gap_filler turn). Every
call in the codebase goes through src.llm.call_structured with temperature=0,
so keying the parsed result on the assembled prompt replays a whole warm-up
in ~0s.

Off unless CDC_LLM_CACHE=1, so real runs are never served stale output. The
env vars are read at call time, not import time, so tests can toggle them.

The cache only ever hits when ids embedded in prompts are reproducible — see
src/ids.py::stable_id.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from pydantic import BaseModel

_DEFAULT_DIR = Path(".cache/llm")


def enabled() -> bool:
    return os.environ.get("CDC_LLM_CACHE") == "1"


def cache_dir() -> Path:
    override = os.environ.get("CDC_LLM_CACHE_DIR")
    return Path(override) if override else _DEFAULT_DIR


def cache_key(*, prompt: str, provider: str, model: str, schema_name: str) -> str:
    payload = "\x1f".join((prompt, provider, model, schema_name))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _path_for(key: str) -> Path:
    # Shard by first byte so the directory stays browsable after a few
    # thousand entries.
    return cache_dir() / key[:2] / f"{key}.json"


def load(key: str, model: type[BaseModel]) -> BaseModel | None:
    """Return the cached result, or None on miss / unusable entry.

    A corrupt or schema-drifted entry is treated as a miss rather than an
    error: the caller falls through to a live call and overwrites it.
    """
    path = _path_for(key)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return model.model_validate_json(raw)
    except Exception:  # noqa: BLE001 - any unusable entry is just a miss
        return None


def store(key: str, value: BaseModel) -> None:
    """Best-effort write; a cache failure must never break a real run."""
    path = _path_for(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value.model_dump_json(), encoding="utf-8")
    except OSError:
        pass
