"""Tests for the simple baseline and the run comparison.

The baseline exists to be *worse* than the graph, so the thing worth pinning is
not its quality — it is that its output is indistinguishable in shape from a
benchmark run. If a field drifts, the scorer does not crash: it silently reports
a zero, and the comparison then credits the graph with a win it did not earn.

So: the record it produces is checked against the scorer, the naive behaviours
that make it a baseline are checked as behaviours (a retrieval nobody grades, a
question batch nobody dedups, an assumption nobody validates), and the
comparison is checked for refusing pairs that do not measure the same thing.

Everything is offline — the single LLM call is stubbed.
"""

from __future__ import annotations

import json

import pytest

from evals import baseline, compare_runs, scoring
from evals.baseline import BaselineGap, BaselineOutput
from evals.dataset import GroundTruth
from src.config import load_sections

SECTIONS = load_sections()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _output(*gaps: dict) -> BaselineOutput:
    return BaselineOutput(gaps=[BaselineGap(**g) for g in gaps])


@pytest.fixture
def stub_llm(monkeypatch):
    """Replace the one call the baseline makes. Returns the recorded prompt."""

    def install(output: BaselineOutput) -> dict:
        seen: dict = {}

        def fake(prompt, model, *args, prompt_id=None, **kwargs):
            seen["prompt"] = prompt
            seen["prompt_id"] = prompt_id
            return output

        monkeypatch.setattr(baseline, "call_structured", fake)
        monkeypatch.setattr(baseline, "current_model_name", lambda: "test-model")
        return seen

    return install


def _hit(chunk_id: str, score: float, content: str = "extrait") -> dict:
    document, *_ = chunk_id.split("::")
    return {
        "content": content,
        "chunk_id": chunk_id,
        "document": document,
        "source": document,
        "page": None,
        "distance": 1.0,
        "score": score,
    }


# ---------------------------------------------------------------------------
# The single pass
# ---------------------------------------------------------------------------


def test_analyze_builds_a_gap_and_its_question_in_one_call(stub_llm):
    seen = stub_llm(
        _output(
            {
                "section_ids": ["functional"],
                "category": "business_rule",
                "description": "Le seuil d'alerte n'est pas défini.",
                "severity": "blocking",
                "question": "Quel est le seuil d'alerte de réapprovisionnement ?",
            }
        )
    )

    gaps, questions, decisions = baseline.analyze("CDC vague", SECTIONS)

    assert seen["prompt_id"] == baseline.PROMPT_ID
    assert len(gaps) == len(questions) == 1
    assert gaps[0].status == "open"
    assert questions[0].gap_id == gaps[0].id
    # One gap, one question: the graph needs gap_finder and gap_filler for this.
    assert {d.decision_type for d in decisions} == {"gap_detected", "question_drafted"}


def test_analyze_drops_invented_section_ids(stub_llm):
    stub_llm(
        _output(
            {
                "section_ids": ["functional", "chapitre_3"],
                "category": "nfr",
                "description": "Aucune exigence de latence.",
                "severity": "important",
                "question": "Quelle latence maximale ?",
            }
        )
    )

    gaps, _, _ = baseline.analyze("CDC", SECTIONS)

    # A section id nothing recognises would break every section lookup downstream
    # and quietly cost the scorer's section bonus.
    assert gaps[0].section_ids == ["functional"]


def test_analyze_keeps_identically_worded_gaps_apart(stub_llm):
    same = {
        "section_ids": ["functional"],
        "category": "edge_case",
        "description": "Cas non traité.",
        "severity": "important",
        "question": "Que se passe-t-il ?",
    }
    stub_llm(_output(same, same))

    gaps, _, _ = baseline.analyze("CDC", SECTIONS)

    # Content-hashed ids collapse duplicates unless the index is in the hash, and
    # two gaps sharing an id would make the record unreadable to the scorer.
    assert len({g.id for g in gaps}) == 2


def test_prompt_carries_the_whole_cdc_and_every_section():
    prompt = baseline.build_prompt("Le stock doit être suivi.", SECTIONS)

    assert "Le stock doit être suivi." in prompt
    for section in SECTIONS:
        assert section.id in prompt


# ---------------------------------------------------------------------------
# Naive retrieval
# ---------------------------------------------------------------------------


def _one_gap(stub_llm, **overrides):
    stub_llm(
        _output(
            {
                "section_ids": ["technical"],
                "category": "integration",
                "description": "Le format d'échange n'est pas spécifié.",
                "severity": "blocking",
                "question": "Quel format d'échange ?",
                **overrides,
            }
        )
    )
    gaps, questions, _ = baseline.analyze("CDC", SECTIONS)
    return gaps, questions


def test_retrieval_is_measured_but_does_not_close_a_gap_by_default(stub_llm):
    gaps, _ = _one_gap(stub_llm)

    items, decisions = baseline.fill_from_rag(
        gaps, lambda q: [_hit("annexe.pdf::0", 0.9), _hit("annexe.pdf::1", 0.4)]
    )

    # Deciding a chunk *answers* a gap is gap_filler's grading call; the baseline
    # retrieves so Recall@K is comparable, then asks the human anyway. The cost
    # of that is the point — every question RAG could have saved is still asked.
    assert gaps[0].status == "open"
    assert items == []
    assert [d.decision_type for d in decisions] == ["rag_rejected"]
    assert decisions[0].evidence_ids == ["annexe.pdf::0", "annexe.pdf::1"]


def test_rag_closes_a_gap_on_the_retriever_score_alone_when_asked(stub_llm):
    gaps, _ = _one_gap(stub_llm)

    items, decisions = baseline.fill_from_rag(
        gaps,
        lambda q: [_hit("annexe.pdf::0", 0.9), _hit("annexe.pdf::1", 0.4)],
        close_gaps=True,
    )

    assert gaps[0].status == "rag_answered"
    assert items[0].source == "rag"
    assert items[0].id in gaps[0].answer_item_ids
    # No model read the chunks — that is what makes this variant naive, and the
    # reason its retrieval judgment accuracy is worth comparing.
    assert [d.decision_type for d in decisions] == ["rag_answer"]


def test_rag_below_the_floor_leaves_the_gap_open_but_keeps_the_rank_order(stub_llm):
    gaps, _ = _one_gap(stub_llm)

    _, decisions = baseline.fill_from_rag(
        gaps, lambda q: [_hit("annexe.pdf::7", 0.1)], close_gaps=True, score_floor=0.5
    )

    assert gaps[0].status == "open"
    assert gaps[0].rag_attempted is True
    # A rejected retrieval leaves no ContextItem, so scoring.score_retrieval can
    # only see its ranks here.
    assert decisions[0].decision_type == "rag_rejected"
    assert decisions[0].evidence_ids == ["annexe.pdf::7"]


def test_rag_failure_does_not_kill_the_case(stub_llm):
    gaps, _ = _one_gap(stub_llm)

    def boom(query):
        raise RuntimeError("chroma indisponible")

    _, decisions = baseline.fill_from_rag(gaps, boom)

    assert gaps[0].status == "open"
    assert decisions[0].decision_type == "rag_rejected"


# ---------------------------------------------------------------------------
# Naive integration
# ---------------------------------------------------------------------------


def test_an_answer_closes_its_gap_and_a_skip_becomes_an_assumption(stub_llm):
    stub_llm(
        _output(
            {
                "section_ids": ["functional"],
                "category": "business_rule",
                "description": "Seuil non défini.",
                "severity": "blocking",
                "question": "Quel seuil ?",
            },
            {
                "section_ids": ["technical"],
                "category": "nfr",
                "description": "Latence non chiffrée.",
                "severity": "important",
                "question": "Quelle latence ?",
            },
        )
    )
    gaps, _, _ = baseline.analyze("CDC", SECTIONS)

    items, decisions = baseline.integrate(
        gaps,
        [
            {"gap_id": gaps[0].id, "text": "Le seuil est de 20 unités.", "skip": False},
            {"gap_id": gaps[1].id, "text": "", "skip": True},
        ],
    )

    assert [g.status for g in gaps] == ["user_answered", "assumed"]
    assert [i.source for i in items] == ["user_answer", "assumption"]
    # Nothing is dropped on the floor: the "je ne sais pas" path is the same one
    # the Streamlit button takes.
    assert {d.decision_type for d in decisions} == {"answer_integrated", "assumption_built"}


def test_a_blank_answer_counts_as_a_skip(stub_llm):
    gaps, _ = _one_gap(stub_llm)

    baseline.integrate(gaps, [{"gap_id": gaps[0].id, "text": "   ", "skip": False}])

    assert gaps[0].status == "assumed"


def test_section_is_complete_only_when_nothing_important_stays_open(stub_llm):
    gaps, _ = _one_gap(stub_llm)

    statuses = baseline.section_statuses(gaps, SECTIONS)
    assert statuses["technical"].status == "in_progress"
    # A section no gap ever touched is empty, not complete.
    assert statuses["problem"].status == "empty"

    gaps[0].status = "user_answered"
    assert baseline.section_statuses(gaps, SECTIONS)["technical"].status == "complete"

    # Only required sections are tracked, because that is the set the graph is
    # started with — and therefore the denominator the scorer compares against.
    optional = {s.id for s in SECTIONS if not s.required}
    assert optional.isdisjoint(statuses)


# ---------------------------------------------------------------------------
# The record the scorer reads
# ---------------------------------------------------------------------------


@pytest.fixture
def ground_truth() -> GroundTruth:
    return GroundTruth.model_validate(
        {
            "case_id": "unit",
            "title": "unit",
            "dataset_version": "v1",
            "annotator": "test",
            "gaps": [
                {
                    "id": "GT-GAP-001",
                    "section_id": "functional",
                    "category": "business_rule",
                    "severity": "blocking",
                    "description": "Le seuil d'alerte n'est pas défini.",
                    "keywords": ["seuil", "alerte"],
                }
            ],
            "expected_answers": [
                {"gap_ref": "GT-GAP-001", "answer": "Le seuil est de 20 unités.",
                 "keywords": ["seuil"]},
            ],
        }
    )


def _record(stub_llm) -> dict:
    """A baseline run's record, built through the real code path."""
    stub_llm(
        _output(
            {
                "section_ids": ["functional"],
                "category": "business_rule",
                "description": "Le seuil d'alerte de réapprovisionnement n'est pas défini.",
                "severity": "blocking",
                "question": "Quel est le seuil d'alerte de réapprovisionnement ?",
            }
        )
    )
    from evals.run_baseline import initial_context

    gaps, questions, decisions = baseline.analyze("Le stock est suivi.", SECTIONS)
    items, answer_decisions = baseline.integrate(
        gaps, [{"gap_id": gaps[0].id, "text": "Le seuil est de 20 unités.", "skip": False}]
    )
    return {
        "case_id": "unit",
        "title": "unit",
        "status": "ok",
        "finished": True,
        "turns": 1,
        "rounds": 1,
        "wall_s": 1.0,
        "gaps": [g.model_dump() for g in gaps],
        "context_items": [initial_context("Le stock est suivi.").model_dump()]
        + [i.model_dump() for i in items],
        "asked_questions": [q.model_dump() for q in questions],
        "decision_log": [d.model_dump() for d in decisions + answer_decisions],
        "section_statuses": {
            sid: ss.model_dump() for sid, ss in baseline.section_statuses(gaps, SECTIONS).items()
        },
        "transcript": [
            {
                "round": 0,
                "turn": 1,
                "questions": [{"gap_id": gaps[0].id, "text": questions[0].text}],
                "answers": [
                    {"gap_id": gaps[0].id, "text": "Le seuil est de 20 unités.",
                     "skip": False, "reason": "matched"}
                ],
            }
        ],
    }


def test_the_record_scores_like_a_benchmark_run(stub_llm, ground_truth):
    """The whole point: the scorer must not be able to tell who produced this."""
    record = _record(stub_llm)

    score = scoring.score_case(record, ground_truth)

    assert score["gaps"]["overall"]["tp"] == 1
    assert score["gaps"]["blocking_recall"] == 1.0
    assert score["questions"]["asked"] == 1
    assert score["questions"]["mean_score"] is not None
    assert score["completeness"]["gt_gap_coverage"] == 1.0
    assert score["effort"]["resolved_by_human"] == 1


def test_the_record_survives_summarize_and_a_json_round_trip(stub_llm):
    """`summarize` builds summary.csv for both systems; a missing key is a crash."""
    from evals.run_benchmark import _json_default, summarize

    record = _record(stub_llm)
    record["stop_reason"] = "baseline_complete"

    row = summarize(json.loads(json.dumps(record, default=_json_default)))

    assert row["gaps_total"] == 1
    assert row["questions_asked"] == 1
    assert row["resolved_by_user"] == 1
    assert row["sections_total"] == sum(1 for s in SECTIONS if s.required)


# ---------------------------------------------------------------------------
# Comparing two runs
# ---------------------------------------------------------------------------


def _scores(run_id: str, system: str, f1: float, **overrides) -> dict:
    payload = {
        "run_id": run_id,
        "system": system,
        "scorer_version": "v1",
        "match_threshold": 0.5,
        "dataset_version": "v1",
        "simulator_mode": "oracle",
        "scored_cases": ["unit"],
        "llm": {"model": "test-model"},
        "totals": {
            "gaps": {"micro": {"f1": f1, "precision": f1, "recall": f1},
                     "blocking_recall": f1, "predicted": 10},
            "contradictions": {"micro": {"f1": 0.0, "recall": 0.0}},
            "retrieval": {"recall_at_k": {"@3": 0.5}, "mrr": 0.4,
                          "sufficiency_judgment": {"accuracy": 0.5}},
            "questions": {"asked": 12, "mean_score": 1.2, "questions_per_resolved_gap": 1.0},
            "effort": {"human_intervention_reduction": 0.2, "wall_s_per_case": 60.0},
            "completeness": {"gt_gap_coverage": f1, "quality_score_delta": 0.3},
        },
    }
    payload.update(overrides)
    return payload


def test_comparison_reports_a_signed_delta_in_the_metrics_own_units():
    report = compare_runs.render(
        _scores("base_1", "baseline", 0.40), _scores("bench_1", "graph", 0.55), []
    )

    assert "+15.0 pts" in report
    assert "ahead on 3/4" in report  # question quality is level in this pair


def test_comparison_refuses_runs_that_do_not_measure_the_same_thing():
    problems = compare_runs.incomparable(
        _scores("base_1", "baseline", 0.4),
        _scores("bench_1", "graph", 0.5, simulator_mode="realistic"),
    )

    assert any("simulator mode" in p for p in problems)


def test_comparison_refuses_two_different_models():
    """Otherwise the delta answers "which model?", not "is the architecture worth it?"."""
    problems = compare_runs.incomparable(
        _scores("base_1", "baseline", 0.4),
        _scores("bench_1", "graph", 0.5, llm={"model": "other-model"}),
    )

    assert problems == ["model: 'test-model' vs 'other-model'"]


def test_comparison_refuses_a_run_against_itself():
    assert compare_runs.incomparable(
        _scores("bench_1", "graph", 0.5), _scores("bench_1", "graph", 0.5)
    ) == ["both sides are the same run"]


def test_an_unmeasured_metric_is_a_dash_not_a_zero_delta():
    """A metric neither run measured must not read as "no change"."""
    reference = _scores("base_1", "baseline", 0.4)
    candidate = _scores("bench_1", "graph", 0.5)
    reference["totals"]["retrieval"]["mrr"] = None

    assert compare_runs.fmt_delta(None, 0.4, "ratio") == "—"
    assert "—" in compare_runs.render(reference, candidate, [])
