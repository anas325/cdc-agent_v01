"""Provenance on ContextItems: where did each piece of information come from?

The audit story only holds if every accepted fact carries its origin, its
evidence, and a review status derived from confidence — regardless of whether it
came from RAG, a human, or an LLM-proposed default.
"""

from __future__ import annotations

import pytest

from src.agents.gap_filler import AssumptionDraft, RagGrade, build_assumption, fill_gaps
from src.decisions import confidence_band, evidence_from_hit, validation_for
from src.state import CDCState, Gap, PendingQuestion, SectionConfig


def make_state(gaps: list[Gap]) -> CDCState:
    return {
        "sections_config": [
            SectionConfig(id="sec_a", title="Section A", description="desc A", required=True, template_slot="a")
        ],
        "context_items": [],
        "gaps": gaps,
        "asked_questions": [],
        "section_statuses": {},
        "turn": 3,
    }


def make_gap(gap_id: str = "gap_1", **overrides) -> Gap:
    fields = {
        "id": gap_id,
        "section_ids": ["sec_a"],
        "category": "business_rule",
        "description": gap_id,
        "severity": "blocking",
        "status": "open",
        "rag_attempted": False,
    }
    fields.update(overrides)
    return Gap(**fields)


HIT = {
    "content": "Le responsable de magasin peut modifier les quantités en stock.",
    "chunk_id": "stock_process.pdf::p14::3",
    "document": "stock_process.pdf",
    "source": "stock_process.pdf",
    "page": 14,
    "distance": 0.15,
    "score": 0.87,
}


@pytest.fixture
def one_rag_hit(monkeypatch):
    monkeypatch.setattr("src.agents.gap_filler.retrieve", lambda query, top_k=None: [HIT])


def patch_grade(monkeypatch, grade: RagGrade) -> None:
    monkeypatch.setattr(
        "src.agents.gap_filler.call_structured",
        lambda prompt, model, llm=None, max_retries=2, *, prompt_id=None: grade,
    )


# ---------------------------------------------------------------------------
# RAG answers carry their evidence
# ---------------------------------------------------------------------------


def test_rag_answer_cites_the_document_and_page_it_came_from(monkeypatch, one_rag_hit):
    patch_grade(
        monkeypatch,
        RagGrade(
            sufficient=True,
            answer_summary="Le responsable de magasin peut modifier le stock.",
            confidence=0.91,
            evidence_grade="sufficient",
        ),
    )

    result = fill_gaps(make_state([make_gap()]), turn=3, max_batch=3, max_per_gap=2)

    assert len(result.new_context_items) == 1
    item = result.new_context_items[0]
    assert item.source == "rag"
    assert item.created_by == "rag"
    assert item.timestamp is not None
    assert item.evidence_grade == "sufficient"
    assert item.confidence == 0.91
    assert item.model and item.prompt_version  # audit needs both
    assert [ev.chunk_id for ev in item.evidence] == ["stock_process.pdf::p14::3"]
    assert item.evidence[0].document == "stock_process.pdf"
    assert item.evidence[0].page == 14
    assert item.evidence[0].retrieval_score == pytest.approx(0.87)


def test_high_confidence_rag_answer_is_auto_accepted(monkeypatch, one_rag_hit):
    patch_grade(monkeypatch, RagGrade(sufficient=True, answer_summary="ok", confidence=0.95))

    result = fill_gaps(make_state([make_gap()]), turn=3, max_batch=3, max_per_gap=2)

    assert result.new_context_items[0].validation_status == "accepted"


def test_low_confidence_rag_answer_is_flagged_for_review(monkeypatch, one_rag_hit):
    """A weakly-supported answer must never slip into the CDC unreviewed."""
    patch_grade(monkeypatch, RagGrade(sufficient=True, answer_summary="peut-être", confidence=0.55))

    result = fill_gaps(make_state([make_gap()]), turn=3, max_batch=3, max_per_gap=2)

    assert result.new_context_items[0].validation_status == "needs_review"


def test_confidence_reported_on_a_0_to_100_scale_is_normalized(monkeypatch, one_rag_hit):
    """Models routinely answer 95 instead of 0.95; that must not read as 'max'."""
    patch_grade(monkeypatch, RagGrade(sufficient=True, answer_summary="ok", confidence=95))

    item = fill_gaps(make_state([make_gap()]), turn=3, max_batch=3, max_per_gap=2).new_context_items[0]

    assert item.confidence == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Assumptions are never silently trusted
# ---------------------------------------------------------------------------


def test_assumption_is_always_flagged_for_review(monkeypatch):
    """Even a confidently-stated default is an assumption, not a validated fact."""
    monkeypatch.setattr(
        "src.agents.gap_filler.call_structured",
        lambda prompt, model, llm=None, max_retries=2, *, prompt_id=None: AssumptionDraft(
            assumption_text="ASSUMPTION: rétention de 12 mois.", confidence=0.99
        ),
    )

    item = build_assumption(make_state([make_gap()]), make_gap(), turn=3)

    assert item.source == "assumption"
    assert item.created_by == "llm"
    assert item.validation_status == "needs_review"
    assert item.confidence == pytest.approx(0.99)
    assert item.model and item.prompt_version


# ---------------------------------------------------------------------------
# Human answers are authoritative
# ---------------------------------------------------------------------------


def test_user_answer_is_attributed_to_the_user_and_accepted(monkeypatch):
    from src import graph as graph_module

    state = make_state([make_gap("gap_1", rag_attempted=True)])
    state["pending_user_questions"] = [
        PendingQuestion(gap_id="gap_1", text="Qui peut modifier le stock ?")
    ]
    state["_raw_answers"] = {"gap_1": {"text": "Le responsable de magasin.", "skip": False}}

    updates = graph_module.integrate_answers_node(state)

    item = updates["context_items"][-1]
    assert item.source == "user_answer"
    assert item.created_by == "user"
    assert item.confidence == 1.0
    assert item.validation_status == "accepted"
    assert item.timestamp is not None
    # No LLM produced this text, so nothing may be attributed to one.
    assert item.model is None and item.prompt_version is None


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------


def test_confidence_bands_and_acceptance_policy():
    assert confidence_band(0.9) == "high"
    assert confidence_band(0.6) == "medium"
    assert confidence_band(0.2) == "low"
    assert confidence_band(None) == "unknown"

    assert validation_for(0.9) == "accepted"
    # Everything below the high band, including "unknown", needs a human look.
    assert validation_for(0.6) == "needs_review"
    assert validation_for(None) == "needs_review"


def test_evidence_from_hit_truncates_long_excerpts():
    from src.state import EXCERPT_MAX_CHARS

    ev = evidence_from_hit({**HIT, "content": "x" * (EXCERPT_MAX_CHARS + 500)})

    assert len(ev.excerpt) <= EXCERPT_MAX_CHARS + 1  # +1 for the ellipsis
    assert ev.excerpt.endswith("…")


def test_evidence_survives_a_hit_missing_optional_metadata():
    """md/txt sources have no page, and old chunks may lack an explicit id."""
    ev = evidence_from_hit({"content": "texte", "source": "notes.md", "score": 0.4})

    assert ev.document == "notes.md"
    assert ev.page is None
    assert ev.retrieval_score == pytest.approx(0.4)
