"""Loads config/sections.yaml and config/settings.yaml into typed models."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from src.state import LoopSettings, SectionConfig

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"

load_dotenv(Path(__file__).resolve().parent / ".env")


class LLMSettings(BaseModel):
    provider: str = "ollama"
    model: str = "mistral"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2


class EmbeddingSettings(BaseModel):
    provider: str = "ollama"
    model: str = "nomic-embed-text"
    fallback_model: str = "all-MiniLM-L6-v2"
    base_url: str = "http://localhost:11434"


class RagSettings(BaseModel):
    persist_dir: str = ".chroma"
    source_dir: str = "data/source_docs"
    top_k: int = 4


class QuartoSettings(BaseModel):
    template_path: str = "templates/cdc_template.qmd"
    output_dir: str = "output"
    output_basename: str = "cdc_final"
    target_format: str = "docx"


class Settings(BaseModel):
    llm: LLMSettings = LLMSettings()
    embeddings: EmbeddingSettings = EmbeddingSettings()
    loop: LoopSettings = LoopSettings()
    rag: RagSettings = RagSettings()
    quarto: QuartoSettings = QuartoSettings()


# Memoized: these are read on nearly every LLM call (call_structured reads
# settings up to 4×), yet the YAML never changes during a run. Callers must
# treat the returned models as read-only — they are shared instances. Tests that
# rewrite config on disk should call clear_config_cache() (or monkeypatch the
# loader at its import site, as the existing suite does).
@lru_cache(maxsize=None)
def _load_settings_cached(path: Path | None = None) -> Settings:
    path = path or CONFIG_DIR / "settings.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings.model_validate(raw)


# An in-process substitute for the on-disk settings, used by the evaluation
# harness to give each benchmark case its own RAG corpus / index / output dir
# without rewriting YAML. It sits *in front of* the memoized loader rather than
# inside it, so installing one doesn't have to invalidate the cache.
_override: Settings | None = None


def load_settings(path: Path | None = None) -> Settings:
    """Typed settings, honouring an active override (evaluation harness / tests)."""
    if _override is not None:
        return _override
    return _load_settings_cached(path)


def set_settings_override(settings: Settings | None) -> None:
    """Install a process-wide Settings override, or clear it with None.

    Consumers that memoize objects built from settings (src.rag.get_collection,
    src.llm.get_llm) must drop their own caches after calling this — see
    evals/harness.py::isolate, which does both.
    """
    global _override
    _override = settings


@lru_cache(maxsize=None)
def load_sections(path: Path | None = None) -> list[SectionConfig]:
    path = path or CONFIG_DIR / "sections.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return [SectionConfig.model_validate(s) for s in raw["sections"]]


def clear_config_cache() -> None:
    """Drop memoized settings/sections (call after editing the YAML on disk)."""
    _load_settings_cached.cache_clear()
    load_sections.cache_clear()
    set_settings_override(None)
