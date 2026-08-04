"""Shared plumbing for the offline eval harnesses.

Two concerns live here:

1. **Provider forcing** — `override_llm_provider` rebinds `call_structured` inside
   every agent module so a whole eval run goes to one provider regardless of
   config/settings.yaml.
2. **Per-case isolation** — a benchmark case must not see another case's RAG
   corpus, Chroma index, or output files. `case_settings` + `isolate` swap the
   process-wide Settings and drop every cache derived from it.

Plus `run_manifest`, the reproducibility record described in roadmap §17.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

from src import config as config_module
from src import llm as llm_module
from src import prompts, rag
from src.config import Settings

ROOT = Path(__file__).resolve().parent.parent

# Only the model needs to change when forcing a provider; base_url/temperature
# stay whatever config/settings.yaml already has.
PROVIDER_DEFAULT_MODEL = {
    "ollama": "gpt-oss:20b",
    "anthropic": "claude-sonnet-5",
}

# Agent modules that bind `call_structured` into their own namespace at import
# time. Keep in sync with src/agents/ — a module missing here silently keeps
# using the configured provider.
AGENT_MODULE_PATHS = [
    "src.agents.orchestrator",
    "src.agents.gap_finder",
    "src.agents.gap_filler",
    "src.agents.critic",
    "src.agents.synthesizer",
    "src.agents.final_validator",
]


# ---------------------------------------------------------------------------
# Provider forcing
# ---------------------------------------------------------------------------


def override_llm_provider(provider: str, *, model: str | None = None) -> None:
    """Force every call_structured() call in the agent modules onto `provider`.

    Agent modules do `from src.llm import call_structured` at import time, so the
    name lives in each agent module's namespace (same pattern used by
    tests/test_graph_flow.py's ScriptedLLM) — patch it there, not on src.llm.

    The replacement must mirror src.llm.call_structured's signature exactly,
    keyword-only `prompt_id` included: agents pass it on every call, and dropping
    it would raise TypeError instead of overriding anything.
    """
    import importlib

    settings = config_module.load_settings()
    cfg = settings.llm.model_copy(
        update={"provider": provider, "model": model or PROVIDER_DEFAULT_MODEL[provider]}
    )
    fixed_llm = llm_module._build_llm(cfg, json_mode=(provider == "ollama"))

    def call_structured_fixed(prompt, model, llm=None, max_retries=2, *, prompt_id=None):
        return llm_module.call_structured(
            prompt, model, llm=fixed_llm, max_retries=max_retries, prompt_id=prompt_id
        )

    for path in AGENT_MODULE_PATHS:
        module = importlib.import_module(path)
        if hasattr(module, "call_structured"):
            module.call_structured = call_structured_fixed


# ---------------------------------------------------------------------------
# Per-case isolation
# ---------------------------------------------------------------------------


def _rel(path: Path | str) -> str:
    """Settings paths are interpreted relative to the repo root by src/rag.py and
    src/agents/final_validator.py, so store them that way when we can."""
    p = Path(path)
    if p.is_absolute():
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)
    return str(p)


def case_settings(
    base: Settings,
    *,
    source_dir: Path | str,
    persist_dir: Path | str,
    output_dir: Path | str,
    top_k: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> Settings:
    """A copy of `base` pointing RAG and document output at case-private dirs."""
    rag_update: dict = {"source_dir": _rel(source_dir), "persist_dir": _rel(persist_dir)}
    if top_k is not None:
        rag_update["top_k"] = top_k

    llm_update: dict = {}
    if provider is not None:
        llm_update["provider"] = provider
        llm_update["model"] = model or PROVIDER_DEFAULT_MODEL[provider]
    elif model is not None:
        llm_update["model"] = model
    if temperature is not None:
        llm_update["temperature"] = temperature

    return base.model_copy(
        update={
            "rag": base.rag.model_copy(update=rag_update),
            "quarto": base.quarto.model_copy(update={"output_dir": _rel(output_dir)}),
            "llm": base.llm.model_copy(update=llm_update) if llm_update else base.llm,
        }
    )


def _drop_settings_derived_caches() -> None:
    """Every memoized object built from Settings, dropped in one place.

    src.rag.get_collection pins a Chroma client to a persist_dir; src.llm's two
    caches pin a chat model to a provider/model. All three would otherwise
    survive a settings swap and keep pointing at the previous case.
    """
    rag.get_collection.cache_clear()
    llm_module.get_llm.cache_clear()
    llm_module._get_structured_llm.cache_clear()


@contextmanager
def isolate(settings: Settings):
    """Run a block against `settings` instead of config/settings.yaml."""
    previous = config_module._override
    config_module.set_settings_override(settings)
    _drop_settings_derived_caches()
    try:
        yield settings
    finally:
        config_module.set_settings_override(previous)
        _drop_settings_derived_caches()


# ---------------------------------------------------------------------------
# Reproducibility manifest (roadmap §17)
# ---------------------------------------------------------------------------


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def run_manifest(
    *,
    run_id: str,
    dataset_version: str,
    case_ids: list[str],
    simulator_mode: str,
    seed: int,
    settings: Settings,
    started_at: str,
) -> dict:
    """Everything needed to reproduce (or fairly compare against) this run."""
    dirty = _git("status", "--porcelain")
    return {
        "run_id": run_id,
        "started_at": started_at,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(dirty) if dirty is not None else None,
        "dataset_version": dataset_version,
        "case_ids": case_ids,
        "simulator_mode": simulator_mode,
        "seed": seed,
        "llm": {
            "provider": settings.llm.provider,
            "model": settings.llm.model,
            "base_url": settings.llm.base_url,
            "temperature": settings.llm.temperature,
        },
        "embeddings": {
            "provider": settings.embeddings.provider,
            "model": settings.embeddings.model,
            "fallback_model": settings.embeddings.fallback_model,
        },
        "rag": {
            "top_k": settings.rag.top_k,
            # Chunking is hard-coded in src/rag.py::_chunk_text; record the
            # defaults so a later change is visible when comparing runs.
            "chunk_size": rag._chunk_text.__defaults__[0],
            "chunk_overlap": rag._chunk_text.__defaults__[1],
        },
        "loop": settings.loop.model_dump(),
        "prompt_versions": dict(prompts.PROMPT_VERSIONS),
        "config_hashes": {
            "sections.yaml": _sha256(config_module.CONFIG_DIR / "sections.yaml"),
            "settings.yaml": _sha256(config_module.CONFIG_DIR / "settings.yaml"),
        },
        "llm_cache": {
            "enabled": os.environ.get("CDC_LLM_CACHE") == "1",
            "dir": os.environ.get("CDC_LLM_CACHE_DIR"),
        },
    }
