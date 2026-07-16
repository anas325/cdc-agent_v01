"""Unit tests for per-section CDC splitting in ingest_node.

Config loading and RAG ingestion are monkeypatched — no filesystem, no Chroma.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.graph as graph
from src.state import LoopSettings, SectionConfig


def make_section(id_: str, title: str, aliases: list[str] | None = None) -> SectionConfig:
    return SectionConfig(
        id=id_,
        title=title,
        description="",
        template_slot=id_,
        aliases=aliases or [],
    )


SECTIONS = [
    make_section("problem", "Contexte", aliases=["Problème"]),
    make_section("functional", "Besoins fonctionnels"),
]


@pytest.fixture(autouse=True)
def stub_ingest_dependencies(monkeypatch):
    monkeypatch.setattr(graph, "load_settings", lambda: SimpleNamespace(loop=LoopSettings()))
    monkeypatch.setattr(graph, "load_sections", lambda: SECTIONS)
    monkeypatch.setattr(graph, "ingest_source_docs", lambda: None)


def test_ingest_creates_one_item_per_matched_section():
    doc = "## Contexte\nTexte contexte.\n\n## Besoins fonctionnels\nTexte fonctionnel.\n"
    result = graph.ingest_node({"initial_cdc_text": doc})
    items = result["context_items"]
    assert len(items) == 2
    by_sections = {tuple(it.section_ids): it for it in items}
    assert "Texte contexte." in by_sections[("problem",)].content
    assert "Texte fonctionnel." in by_sections[("functional",)].content
    assert all(it.source == "initial_cdc" for it in items)
    assert result["turn_log"][0].details["matched_sections"] == ["problem", "functional"]


def test_ingest_preamble_becomes_untagged_item():
    doc = "Titre du document\n\n## Contexte\nTexte contexte.\n"
    result = graph.ingest_node({"initial_cdc_text": doc})
    items = result["context_items"]
    untagged = [it for it in items if it.section_ids == []]
    assert len(untagged) == 1
    assert "Titre du document" in untagged[0].content
    assert result["turn_log"][0].details["unmatched_chunks"] == 1


def test_ingest_unstructured_doc_falls_back_to_whole_doc_item():
    doc = "Prose libre sans aucun titre reconnaissable."
    result = graph.ingest_node({"initial_cdc_text": doc})
    items = result["context_items"]
    assert len(items) == 1
    assert items[0].content == doc
    assert set(items[0].section_ids) == {"problem", "functional"}


def test_ingest_empty_text_creates_no_items():
    result = graph.ingest_node({"initial_cdc_text": "  \n "})
    assert result["context_items"] == []
