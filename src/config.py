"""Loads config/sections.yaml and config/settings.yaml into typed models."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from src.state import LoopSettings, SectionConfig

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"


class LLMSettings(BaseModel):
    provider: str = "ollama"
    model: str = "mistral"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2


class EmbeddingSettings(BaseModel):
    provider: str = "ollama"
    model: str = "nomic-embed-text"
    fallback_model: str = "all-MiniLM-L6-v2"


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


def load_settings(path: Path | None = None) -> Settings:
    path = path or CONFIG_DIR / "settings.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings.model_validate(raw)


def load_sections(path: Path | None = None) -> list[SectionConfig]:
    path = path or CONFIG_DIR / "sections.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return [SectionConfig.model_validate(s) for s in raw["sections"]]
