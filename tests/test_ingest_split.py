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
# --- Mode « sans CDC initial » : l'auteur décrit chaque section lui-même -----


def test_declared_sections_become_one_item_each():
    result = graph.ingest_node(
        {"initial_section_texts": {"problem": "Le stock est suivi sur papier.", "functional": "Saisie mobile."}}
    )
    items = result["context_items"]
    assert [it.section_ids for it in items] == [["problem"], ["functional"]]
    assert [it.content for it in items] == ["Le stock est suivi sur papier.", "Saisie mobile."]
    # Même statut qu'un CDC déposé, mais attribué à l'auteur, pas au découpeur.
    assert all(it.source == "initial_cdc" and it.created_by == "user" for it in items)
    assert all(it.confidence == 1.0 and it.validation_status == "accepted" for it in items)
    assert all(not it.fresh and it.turn_added == 0 for it in items)
    assert result["turn_log"][0].details["declared_sections"] == ["problem", "functional"]


def test_declared_sections_are_ordered_by_config_not_by_input():
    """L'ordre des items est checkpointé : il ne doit pas suivre l'ordre de saisie."""
    result = graph.ingest_node(
        {"initial_section_texts": {"functional": "Saisie mobile.", "problem": "Suivi papier."}}
    )
    assert [it.section_ids[0] for it in result["context_items"]] == ["problem", "functional"]


def test_declared_sections_ignore_blanks_and_unknown_ids():
    result = graph.ingest_node(
        {"initial_section_texts": {"problem": "Suivi papier.", "functional": "   ", "inexistante": "Texte."}}
    )
    items = result["context_items"]
    assert [it.section_ids for it in items] == [["problem"]]
    assert result["turn_log"][0].details["declared_sections"] == ["problem"]
    # Toutes les sections gardent un statut, y compris celles laissées vides.
    assert set(result["section_statuses"]) == {"problem", "functional"}
    assert result["section_statuses"]["functional"].status == "empty"


def test_declared_sections_and_uploaded_doc_are_cumulative():
    """Un brouillon partiel peut être complété à la main : rien n'est écrasé."""
    doc = """## Contexte
Texte contexte.
"""
    result = graph.ingest_node(
        {"initial_cdc_text": doc, "initial_section_texts": {"functional": "Saisie mobile."}}
    )
    items = result["context_items"]
    assert [it.section_ids for it in items] == [["problem"], ["functional"]]
    details = result["turn_log"][0].details
    assert details["matched_sections"] == ["problem"]
    assert details["declared_sections"] == ["functional"]


def test_no_input_at_all_still_yields_a_usable_state():
    """Mode guidé sans un mot saisi : la boucle démarre sur des sections vides."""
    result = graph.ingest_node({})
    assert result["context_items"] == []
    assert set(result["section_statuses"]) == {"problem", "functional"}
    assert result["gaps"] == []
