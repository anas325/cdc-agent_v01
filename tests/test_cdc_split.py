"""Unit tests for split_cdc_by_sections / extract_section in src.utils.cdc_sections.

Pure text-parsing tests — no LLM, no graph, no filesystem.
"""

from __future__ import annotations

import pytest

from src.state import SectionConfig
from src.utils.cdc_sections import extract_section, split_cdc_by_sections


def make_section(id_: str, title: str, aliases: list[str] | None = None) -> SectionConfig:
    return SectionConfig(
        id=id_,
        title=title,
        description="",
        template_slot=id_,
        aliases=aliases or [],
    )


SECTIONS = [
    make_section("problem", "Problème et contexte", aliases=["Contexte", "Contexte et objectifs"]),
    make_section("functional", "Besoins fonctionnels", aliases=["Spécifications fonctionnelles"]),
    make_section("technical", "Architecture technique", aliases=["Technique"]),
]


# ---------------------------------------------------------------------------
# split_cdc_by_sections
# ---------------------------------------------------------------------------


def test_split_assigns_each_section_its_text():
    doc = """## Contexte
Le projet vise à optimiser les stocks.

## Besoins fonctionnels
- Suivi temps réel
- Alertes de rupture

## Technique
Stack Python + Postgres.
"""
    split = split_cdc_by_sections(doc, SECTIONS)
    assert split.matched_any
    assert set(split.sections) == {"problem", "functional", "technical"}
    assert "optimiser les stocks" in split.sections["problem"]
    assert "Alertes de rupture" in split.sections["functional"]
    assert "Postgres" in split.sections["technical"]
    assert split.unmatched == []


def test_split_numbered_headings():
    doc = """1. Contexte
Texte du contexte.

II) Spécifications fonctionnelles
Texte fonctionnel.
"""
    split = split_cdc_by_sections(doc, SECTIONS)
    assert "Texte du contexte." in split.sections["problem"]
    assert "Texte fonctionnel." in split.sections["functional"]


def test_split_preamble_goes_to_unmatched():
    doc = """Cahier des charges SmartStock
Version 1.2 — brouillon

## Contexte
Le vrai contexte.
"""
    split = split_cdc_by_sections(doc, SECTIONS)
    assert len(split.unmatched) == 1
    assert "SmartStock" in split.unmatched[0]
    assert "brouillon" in split.unmatched[0]
    assert "Le vrai contexte." in split.sections["problem"]
    assert "SmartStock" not in split.sections["problem"]


def test_split_unrecognized_same_level_heading_goes_to_unmatched():
    doc = """## Contexte
Contenu du contexte.

## Annexes
Contenu annexe qui ne doit pas être perdu.
"""
    split = split_cdc_by_sections(doc, SECTIONS)
    assert "Contenu du contexte." in split.sections["problem"]
    assert "Annexes" not in split.sections["problem"]
    assert len(split.unmatched) == 1
    assert "Annexes" in split.unmatched[0]
    assert "ne doit pas être perdu" in split.unmatched[0]


def test_split_unstructured_doc_matches_nothing():
    doc = "Un document en prose libre.\nSans aucun titre reconnaissable ici même."
    split = split_cdc_by_sections(doc, SECTIONS)
    assert not split.matched_any
    assert split.sections == {}
    assert split.unmatched == [doc]


def test_split_repeated_section_headings_merge():
    doc = """## Contexte
Première partie.

## Besoins fonctionnels
Fonctionnel.

## Contexte
Deuxième partie.
"""
    split = split_cdc_by_sections(doc, SECTIONS)
    assert "Première partie." in split.sections["problem"]
    assert "Deuxième partie." in split.sections["problem"]
    assert list(split.sections["problem"]).count("\n") >= 1


def test_split_empty_text():
    split = split_cdc_by_sections("", SECTIONS)
    assert not split.matched_any
    assert split.unmatched == []


# ---------------------------------------------------------------------------
# extract_section (parity after the segmenter refactor)
# ---------------------------------------------------------------------------


def test_extract_section_returns_first_occurrence_without_heading_line():
    doc = """## Contexte
Première partie.

## Besoins fonctionnels
Fonctionnel.

## Contexte
Deuxième partie.
"""
    extracted = extract_section(doc, "problem", SECTIONS)
    assert extracted == "Première partie."


def test_extract_section_stops_at_unrecognized_same_level_heading():
    doc = """## Contexte
Contenu.

## Annexes
Annexe.
"""
    assert extract_section(doc, "problem", SECTIONS) == "Contenu."


def test_extract_section_keeps_deeper_unrecognized_headings():
    doc = """## Contexte
Intro.

### Sous-partie
Détail.

## Technique
Stack.
"""
    extracted = extract_section(doc, "problem", SECTIONS)
    assert "Sous-partie" in extracted
    assert "Détail." in extracted
    assert "Stack." not in extracted


def test_extract_section_returns_none_when_absent():
    assert extract_section("Prose sans titres.", "problem", SECTIONS) is None


def test_extract_section_raises_on_unknown_id():
    with pytest.raises(KeyError):
        extract_section("## Contexte\nx", "nope", SECTIONS)
