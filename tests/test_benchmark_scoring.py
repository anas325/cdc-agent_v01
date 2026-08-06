"""Tests for the Phase 4 scorer: matching, every metric family, and the CLI.

The scorer is what turns "the system found 30 gaps" into "it found 9 of the 11 a
human annotated". That makes its arithmetic load-bearing — a matching rule that
silently stops matching would not crash, it would just report a worse system.
So the tests here pin the *judgment calls* rather than the plumbing: what counts
as a detection, what counts as a false positive, which denominator each ratio
uses, and which numbers are deliberately not scored at all.

Everything is offline: hand-built records shaped like predictions.json, and the
real benchmark dataset for the CLI test.
"""

from __future__ import annotations

import csv
import json

import pytest

from evals import scoring
from evals.dataset import GroundTruth
from evals.scoring import (
    aggregate,
    collect_contradictions,
    content_tokens,
    first_relevant_rank,
    match_gaps,
    prf,
    score_case,
    score_completeness,
    score_contradictions,
    score_effort,
    score_gaps,
    score_questions,
    score_retrieval,
)


# ---------------------------------------------------------------------------
# Fixtures — the smallest annotation and record that exercise every branch
# ---------------------------------------------------------------------------


def _ground_truth(**overrides) -> GroundTruth:
    data = {
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
            },
            {
                "id": "GT-GAP-002",
                "section_id": "technical",
                "category": "nfr",
                "severity": "important",
                "description": "Aucune cible de disponibilité.",
                "keywords": ["disponibilite"],
                "resolvable_by": "rag",
                "expected_evidence": {"document": "politique.md", "quote": None},
            },
            {
                "id": "GT-GAP-003",
                "section_id": "functional",
                "category": "edge_case",
                "severity": "nice_to_have",
                "description": "La gestion des conflits n'est pas décrite.",
                "keywords": ["conflit"],
            },
        ],
        "contradictions": [
            {
                "id": "GT-CON-001",
                "statement_a": "Budget estimé : 150 000 €",
                "statement_b": "Le budget ne devra pas dépasser 80 000 €",
                "section_ids": ["constraints"],
                "severity": "blocking",
                "keywords": ["budget"],
            }
        ],
    }
    data.update(overrides)
    return GroundTruth.model_validate(data)


def _gap(gap_id: str, description: str, **overrides) -> dict:
    gap = {
        "id": gap_id,
        "section_ids": ["functional"],
        "category": "business_rule",
        "description": description,
        "severity": "blocking",
        "status": "user_answered",
        "question_text": None,
        "answer_item_ids": [],
        "questions_asked": 1,
        "rag_attempted": True,
    }
    gap.update(overrides)
    return gap


def _record(**overrides) -> dict:
    record = {
        "case_id": "unit",
        "title": "unit",
        "status": "ok",
        "finished": True,
        "turns": 3,
        "rounds": 1,
        "wall_s": 12.0,
        "gaps": [],
        "context_items": [],
        "asked_questions": [],
        "section_statuses": {"functional": {"status": "complete"}},
        "transcript": [],
        "decision_log": [],
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# 1. Matching
# ---------------------------------------------------------------------------


def test_a_paraphrase_matches_and_an_unrelated_gap_does_not():
    gt = _ground_truth()
    predictions = [
        _gap("g1", "Le seuil d'alerte de stock faible dépend du produit, sans règle."),
        _gap("g2", "La charte graphique n'est pas fournie."),
    ]

    match = match_gaps(gt.gaps, predictions)

    assert match.pairs == {"GT-GAP-001": "g1"}
    assert match.spurious == ["g2"]
    assert set(match.missed) == {"GT-GAP-002", "GT-GAP-003"}


def test_one_annotation_cannot_be_claimed_by_two_predictions():
    """Otherwise recall could exceed 1.0 and precision would reward duplicates."""
    gt = _ground_truth()
    predictions = [
        _gap("g1", "Le seuil d'alerte n'est pas défini."),
        _gap("g2", "Le seuil d'alerte reste à préciser."),
    ]

    match = match_gaps(gt.gaps, predictions)

    assert len(match.pairs) == 1
    assert len(match.spurious) == 1


def test_matching_is_order_independent_and_deterministic():
    gt = _ground_truth()
    predictions = [
        _gap("g1", "Le seuil d'alerte de stock dépend du produit."),
        _gap("g2", "Les conflits de modification ne sont pas traités."),
    ]

    forward = match_gaps(gt.gaps, predictions).pairs
    backward = match_gaps(gt.gaps, list(reversed(predictions))).pairs

    assert forward == backward == {"GT-GAP-001": "g1", "GT-GAP-003": "g2"}


def test_a_gap_filed_under_the_wrong_section_still_counts_as_detected():
    """The section is a ranking bonus, never a gate — a filing error is not a miss."""
    gt = _ground_truth()
    predictions = [_gap("g1", "Le seuil d'alerte est indéfini.", section_ids=["technical"])]

    assert match_gaps(gt.gaps, predictions).pairs == {"GT-GAP-001": "g1"}


def test_the_question_text_is_part_of_the_haystack():
    """A gap described vaguely but asked about precisely is still that gap."""
    gt = _ground_truth()
    predictions = [
        _gap("g1", "Un paramètre métier reste flou.",
             question_text="Comment est calculé le seuil d'alerte de réapprovisionnement ?")
    ]

    assert match_gaps(gt.gaps, predictions).pairs == {"GT-GAP-001": "g1"}


# ---------------------------------------------------------------------------
# 2. Metric primitives
# ---------------------------------------------------------------------------


def test_prf_arithmetic():
    assert prf(3, 1, 1) == {
        "tp": 3, "fp": 1, "fn": 1,
        "precision": 0.75, "recall": 0.75, "f1": 0.75,
    }


def test_nothing_to_find_and_nothing_reported_scores_one_not_zero():
    """The negative control reports no contradictions and is right, not undefined."""
    empty = prf(0, 0, 0)
    assert (empty["precision"], empty["recall"], empty["f1"]) == (1.0, 1.0, 1.0)
    # But inventing findings where there are none is still punished.
    assert prf(0, 4, 0)["precision"] == 0.0


# ---------------------------------------------------------------------------
# 3. Gap detection (§7, §8)
# ---------------------------------------------------------------------------


def test_gap_scoring_reports_blocking_recall_and_the_severity_matrix():
    gt = _ground_truth()
    record = _record(
        gaps=[
            # Found, but the system called it merely important.
            _gap("g1", "Le seuil d'alerte n'est pas défini.", severity="important"),
            _gap("g2", "Les conflits de modification ne sont pas traités.",
                 category="edge_case", severity="nice_to_have"),
            _gap("g3", "La charte graphique manque.", category="scope", severity="important"),
        ]
    )

    scores = score_gaps(record, gt)

    assert scores["overall"] == prf(2, 1, 1)
    # GT-GAP-002 (blocking? no — GT-GAP-001 is the only blocking one) was found.
    assert (scores["blocking_found"], scores["blocking_total"]) == (1, 1)
    assert scores["blocking_recall"] == 1.0
    # ...but its severity was underestimated, which is what the matrix is for.
    assert scores["severity_confusion"]["blocking"]["important"] == 1
    assert scores["severity_confusion"]["blocking"]["blocking"] == 0
    assert scores["by_category"]["business_rule"]["tp"] == 1
    assert scores["by_category"]["scope"]["fp"] == 1
    assert [u["id"] for u in scores["unmatched_predictions"]] == ["g3"]
    assert [m["id"] for m in scores["missed"]] == ["GT-GAP-002"]


def test_contradictions_are_kept_out_of_the_gap_pool():
    """They have their own annotation list; counting them twice would skew both."""
    gt = _ground_truth()
    record = _record(
        gaps=[
            _gap("g1", "Le seuil d'alerte n'est pas défini."),
            _gap("c1", "Le budget de 150 000 € contredit le plafond de 80 000 €.",
                 category="contradiction"),
        ]
    )

    scores = score_gaps(record, gt)

    assert scores["predicted"] == 1
    assert [u["id"] for u in scores["unmatched_predictions"]] == []


def test_category_agreement_is_measured_separately_from_detection():
    """Finding a gap and labelling it right are different failures."""
    gt = _ground_truth()
    record = _record(
        gaps=[_gap("g1", "Le seuil d'alerte n'est pas défini.", category="scope")]
    )

    scores = score_gaps(record, gt)

    assert scores["overall"]["tp"] == 1  # detection succeeded
    assert scores["category_agreement"] == 0.0  # classification did not


def test_a_mislabelled_gap_is_charged_to_both_classes_but_still_counts_as_found():
    """Per-class P/R/F1 is strict multi-class: a match with the wrong label is a
    miss for the annotated class *and* a false positive for the predicted one.
    Mixing the two (TP by annotation, FP by prediction) would make per-class
    precision meaningless. `detection_recall` is the lenient reading beside it."""
    gt = _ground_truth()
    record = _record(
        gaps=[_gap("g1", "Le seuil d'alerte n'est pas défini.", category="scope")]
    )

    by_category = score_gaps(record, gt)["by_category"]

    assert by_category["business_rule"]["tp"] == 0
    assert by_category["business_rule"]["fn"] == 1
    assert by_category["scope"]["fp"] == 1
    # ...but the author was still told about the gap.
    assert by_category["business_rule"]["detection_recall"] == 1.0
    assert by_category["business_rule"]["annotated"] == 1


def test_blocking_recall_asks_whether_the_gap_was_surfaced_not_how_it_was_labelled():
    """Roadmap §8's safety metric: never mentioning a blocking gap is the
    dangerous failure; under-calling its severity is a milder, separate one that
    the confusion matrix reports."""
    gt = _ground_truth()
    record = _record(
        gaps=[_gap("g1", "Le seuil d'alerte n'est pas défini.", severity="nice_to_have")]
    )

    scores = score_gaps(record, gt)

    assert scores["blocking_recall"] == 1.0
    assert scores["severity_confusion"]["blocking"]["nice_to_have"] == 1
    # The strict per-class view disagrees, on purpose.
    assert scores["by_severity"]["blocking"]["recall"] == 0.0


# ---------------------------------------------------------------------------
# 4. Contradictions (§9)
# ---------------------------------------------------------------------------


def _decision(decision_type: str, **overrides) -> dict:
    entry = {
        "id": "dec_1",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "turn": 1,
        "agent": "critic",
        "decision_type": decision_type,
        "summary": "",
        "input_ids": [],
        "output_ids": [],
        "evidence_ids": [],
        "details": {},
    }
    entry.update(overrides)
    return entry


def test_contradictions_are_collected_from_the_loop_and_the_final_pass():
    record = _record(
        gaps=[_gap("c1", "Le budget est contradictoire.", category="contradiction")],
        decision_log=[
            _decision("gap_detected", agent="gap_finder", output_ids=["c1"]),
            _decision(
                "final_check",
                agent="final_validator",
                details={"contradictions": ["Le délai de livraison diffère entre deux sections."]},
            ),
        ],
    )

    found = collect_contradictions(record)

    assert [c["origin"] for c in found] == ["cdc", "final"]
    # The final validator's finding exists nowhere in `gaps` — reading only gaps
    # would credit the system with none of them.
    assert "délai" in found[1]["text"]


def test_the_final_pass_restating_a_known_contradiction_is_not_counted_twice():
    text = "Le budget de 150 000 euros contredit le plafond de 80 000 euros annonce ailleurs."
    record = _record(
        gaps=[_gap("c1", text, category="contradiction")],
        decision_log=[
            _decision("gap_detected", agent="gap_finder", output_ids=["c1"]),
            _decision("final_check", details={"contradictions": [text]}),
        ],
    )

    assert len(collect_contradictions(record)) == 1


def test_a_contradiction_between_the_runs_own_answers_is_never_a_false_positive():
    """The benchmark annotates contradictions *inside the CDC*. The critic's
    answer-vs-answer findings are a different population — realistic mode exists
    to provoke them — so they may raise recall but must not sink precision."""
    gt = _ground_truth()
    record = _record(
        gaps=[
            _gap("c1", "L'hypothèse d'annulation contredit la réponse précédente.",
                 category="contradiction"),
            _gap("c2", "Le budget annoncé contredit le plafond fixé.",
                 category="contradiction"),
        ],
        decision_log=[
            _decision("contradiction_found", agent="critic", output_ids=["c1"]),
            _decision("gap_detected", agent="gap_finder", output_ids=["c2"]),
        ],
    )

    scores = score_contradictions(record, gt)

    assert scores["by_origin"] == {"answer": 1, "cdc": 1}
    assert scores["answer_level"] == 1
    # c2 matched the annotation; c1 is neither credited nor charged.
    assert scores["overall"] == prf(1, 0, 0)
    assert scores["critical_recall"] == 1.0


def test_an_unattributed_contradiction_is_scored_strictly():
    """An old run has no decision log; over-counting a false positive beats
    silently inflating precision."""
    gt = _ground_truth()
    record = _record(
        gaps=[_gap("c1", "Une incohérence quelconque.", category="contradiction")]
    )

    assert score_contradictions(record, gt)["by_origin"] == {"cdc": 1}
    assert score_contradictions(record, gt)["overall"]["fp"] == 1


# ---------------------------------------------------------------------------
# 5. Retrieval (§10)
# ---------------------------------------------------------------------------


def test_first_relevant_rank_understands_both_chunk_id_schemes():
    chunks = ["autre.md::0", "politique.pdf::p14::3", "politique.md::2"]

    assert first_relevant_rank(chunks, "politique.md", None) == 3
    assert first_relevant_rank(chunks, "politique.pdf", 14) == 2
    # The annotation pins a page the retrieval never returned.
    assert first_relevant_rank(chunks, "politique.pdf", 9) is None
    assert first_relevant_rank([], "politique.md", None) is None


def test_recall_at_k_and_mrr_come_from_the_rank_of_the_annotated_chunk():
    gt = _ground_truth()
    record = _record(
        gaps=[_gap("g2", "Aucune cible de disponibilité n'est fixée.",
                   category="nfr", severity="important", section_ids=["technical"])],
        decision_log=[
            _decision(
                "rag_answer",
                agent="gap_filler",
                input_ids=["g2"],
                evidence_ids=["autre.md::0", "autre.md::1", "politique.md::4"],
            )
        ],
    )
    match = match_gaps(gt.gaps, record["gaps"])

    scores = score_retrieval(record, gt, match)

    assert scores["attempted"] == 1
    assert scores["recall_at_k"] == {"@1": 0.0, "@3": 1.0, "@5": 1.0}
    assert scores["mrr"] == pytest.approx(1 / 3, abs=1e-4)
    assert scores["sufficiency_judgment"]["correct_accept"] == 1


def test_a_rejected_retrieval_still_counts_and_names_the_failure_mode():
    """A rejected retrieval leaves no ContextItem, so only the decision log knows
    it happened — and it is exactly the retrieval-vs-reasoning split of §10."""
    gt = _ground_truth()
    record = _record(
        gaps=[_gap("g2", "Aucune cible de disponibilité n'est fixée.",
                   category="nfr", section_ids=["technical"])],
        decision_log=[
            _decision("rag_rejected", agent="gap_filler", input_ids=["g2"],
                      evidence_ids=["politique.md::4"])
        ],
    )
    match = match_gaps(gt.gaps, record["gaps"])

    scores = score_retrieval(record, gt, match)

    assert scores["recall_at_k"]["@1"] == 1.0  # retrieval worked
    assert scores["sufficiency_judgment"]["wrong_reject"] == 1  # the grader did not
    assert scores["sufficiency_judgment"]["accuracy"] == 0.0


def test_a_rag_gap_that_was_never_detected_is_separated_from_a_retrieval_miss():
    gt = _ground_truth()
    record = _record(gaps=[])  # nothing detected at all

    scores = score_retrieval(record, gt, match_gaps(gt.gaps, []))

    assert (scores["annotated"], scores["detected"], scores["attempted"]) == (1, 0, 0)
    # Nothing was attempted, so there is no retrieval quality to report...
    assert scores["recall_at_k"]["@3"] is None
    # ...but end to end the document did go unread.
    assert scores["recall_at_k_overall"]["@3"] == 0.0


def test_a_case_with_no_rag_annotations_reports_nothing_rather_than_zero():
    """cdc_001_smartstock is the 'zéro RAG' control: absent must not read as failed."""
    gt = _ground_truth(gaps=[
        {
            "id": "GT-GAP-001",
            "section_id": "functional",
            "category": "business_rule",
            "severity": "blocking",
            "description": "Le seuil n'est pas défini.",
            "keywords": ["seuil"],
        }
    ])

    scores = score_retrieval(_record(), gt, match_gaps(gt.gaps, []))

    assert scores["annotated"] == 0
    assert scores["mrr"] is None
    assert scores["recall_at_k"] == {"@1": None, "@3": None, "@5": None}


# ---------------------------------------------------------------------------
# 6. Question quality (§11)
# ---------------------------------------------------------------------------

GOOD_QUESTION = (
    "Vous écrivez « le seuil d'alerte dépend du produit » : quelle règle de calcul "
    "faut-il appliquer, et qui paramètre cette valeur pour les 18 magasins ?"
)


def _question_record(*questions: dict, **overrides) -> dict:
    return _record(
        gaps=[_gap("g1", "Le seuil d'alerte n'est pas défini.")],
        context_items=[
            {
                "source": "initial_cdc",
                "section_ids": ["functional"],
                "content": "Le seuil d'alerte dépend du produit et concerne les magasins.",
            }
        ],
        asked_questions=list(questions),
        **overrides,
    )


def test_a_precise_contextual_question_scores_well_on_every_dimension():
    gt = _ground_truth()
    record = _question_record(
        {"id": "q1", "gap_id": "g1", "text": GOOD_QUESTION, "turn": 1},
        transcript=[{"answers": [{"gap_id": "g1", "skip": False, "reason": "matched"}]}],
    )

    scores = score_questions(record, gt, match_gaps(gt.gaps, record["gaps"]))

    dims = scores["per_question"][0]["scores"]
    assert dims["addresses_gap"] == 2
    assert dims["specific"] == 2
    assert dims["understandable"] == 2
    assert dims["has_context"] == 2
    assert dims["not_duplicate"] == 2
    assert dims["answerable"] == 2
    assert scores["quality"] == 1.0


def test_a_generic_question_is_marked_down_for_being_unspecific():
    gt = _ground_truth()
    record = _question_record(
        {"id": "q1", "gap_id": "g1", "text": "Pouvez-vous préciser le périmètre ?", "turn": 1}
    )

    dims = score_questions(record, gt, match_gaps(gt.gaps, record["gaps"]))["per_question"][0]
    assert dims["scores"]["specific"] == 0


def test_asking_almost_the_same_question_twice_is_penalised():
    gt = _ground_truth()
    record = _question_record(
        {"id": "q1", "gap_id": "g1", "text": GOOD_QUESTION, "turn": 1},
        {"id": "q2", "gap_id": "g1", "text": GOOD_QUESTION, "turn": 2},
    )

    scores = score_questions(record, gt, match_gaps(gt.gaps, record["gaps"]))

    assert scores["per_question"][0]["scores"]["not_duplicate"] == 2
    assert scores["per_question"][1]["scores"]["not_duplicate"] == 0


def test_answerability_is_read_off_the_run_and_left_unmeasured_when_absent():
    gt = _ground_truth()
    base = {"id": "q1", "gap_id": "g1", "text": GOOD_QUESTION, "turn": 1}

    def answerable(transcript):
        record = _question_record(base, transcript=transcript)
        scores = score_questions(record, gt, match_gaps(gt.gaps, record["gaps"]))
        return scores["per_question"][0]["scores"]["answerable"]

    assert answerable([{"answers": [{"gap_id": "g1", "skip": False}]}]) == 2
    assert answerable([{"answers": [{"gap_id": "g1", "skip": True, "reason": "hedged"}]}]) == 1
    assert answerable([{"answers": [{"gap_id": "g1", "skip": True, "reason": "unknown"}]}]) == 0
    # Drafted but never asked (the run stopped first): unmeasured, not zero.
    assert answerable([]) is None


def test_question_efficiency_counts_questions_per_closed_gap():
    gt = _ground_truth()
    record = _question_record(
        {"id": "q1", "gap_id": "g1", "text": GOOD_QUESTION, "turn": 1},
        {"id": "q2", "gap_id": "g1", "text": "Et pour les magasins fermés ?", "turn": 2},
    )

    scores = score_questions(record, gt, match_gaps(gt.gaps, record["gaps"]))

    assert scores["resolved_gaps"] == 1
    assert scores["questions_per_resolved_gap"] == 2.0


def test_context_is_judged_against_untagged_cdc_text_too():
    """`ingest` only tags the chunks it can map to a section, so a CDC's title and
    intro prose carry none. Scoring a question only against its own section's
    slice marked every quote of that prose as contextless — and any section the
    splitter never populated scored a flat zero."""
    gt = _ground_truth()
    record = _record(
        gaps=[_gap("g1", "Le seuil d'alerte n'est pas défini.", section_ids=["technical"])],
        context_items=[
            # Real CDC text the splitter could not attribute to any section.
            {"source": "initial_cdc", "section_ids": [],
             "content": "Le seuil d'alerte dépend du produit et concerne les magasins."},
        ],
        asked_questions=[{"id": "q1", "gap_id": "g1", "text": GOOD_QUESTION, "turn": 1}],
    )

    scores = score_questions(record, gt, match_gaps(gt.gaps, record["gaps"]))

    assert scores["per_question"][0]["scores"]["has_context"] == 2


def test_content_tokens_drops_short_and_function_words():
    tokens = content_tokens("Quel est le délai de réponse pour les magasins ?")
    assert "delai" in tokens and "magasins" in tokens
    assert "pour" not in tokens and "est" not in tokens


# ---------------------------------------------------------------------------
# 7. Human effort (§26) and end-to-end improvement (§12)
# ---------------------------------------------------------------------------


def test_human_intervention_reduction_is_the_rag_share_of_all_resolutions():
    record = _record(
        context_items=[
            {"source": "initial_cdc"},  # not a resolution
            {"source": "rag"},
            {"source": "rag"},
            {"source": "user_answer"},
            {"source": "assumption"},
        ],
        transcript=[{"answers": [{"skip": True}, {"skip": False}]}],
    )

    effort = score_effort(record)

    assert (effort["resolved_by_rag"], effort["resolved_by_human"], effort["assumed"]) == (2, 1, 1)
    assert effort["human_intervention_reduction"] == 0.5
    assert effort["unknown_answers"] == 1


def test_completeness_proxy_credits_gaps_that_were_found_and_closed():
    gt = _ground_truth()
    record = _record(
        gaps=[
            _gap("g1", "Le seuil d'alerte n'est pas défini.", status="user_answered"),
            # Detected but left open at max_turns: found is not the same as closed.
            _gap("g3", "Les conflits de modification ne sont pas traités.",
                 category="edge_case", severity="nice_to_have", status="open"),
        ]
    )
    match = match_gaps(gt.gaps, record["gaps"])

    scores = score_completeness(record, gt, match)

    assert scores["gt_gap_detection"] == pytest.approx(2 / 3, abs=1e-4)
    assert scores["gt_gap_coverage"] == pytest.approx(1 / 3, abs=1e-4)
    assert scores["blocking_coverage"] == 1.0
    assert scores["gaps_still_open"] == 1
    # The section score improves as gaps are closed, but the open one still costs.
    assert scores["quality_score_final"] > scores["quality_score_initial"]
    assert scores["quality_score_delta"] > 0


# ---------------------------------------------------------------------------
# 8. Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_pools_counts_rather_than_averaging_ratios():
    """Micro-averaging: a 20-gap case must weigh more than the 3-gap control."""
    gt = _ground_truth()
    big = score_case(
        _record(case_id="big", gaps=[_gap("g1", "Le seuil d'alerte n'est pas défini.")]), gt
    )
    small = score_case(_record(case_id="small", gaps=[]), gt)

    totals = aggregate([big, small])

    assert totals["cases"] == 2
    assert totals["gaps"]["micro"]["tp"] == 1
    assert totals["gaps"]["micro"]["fn"] == 5  # 2 missed + 3 missed
    assert totals["gaps"]["blocking_total"] == 2
    assert totals["gaps"]["by_category"]["business_rule"]["tp"] == 1
    assert aggregate([]) == {}


def test_score_case_returns_every_metric_family():
    scores = score_case(_record(), _ground_truth())

    assert set(scores) >= {
        "gaps", "contradictions", "retrieval", "questions", "effort", "completeness"
    }
    # The match object is internal plumbing and must not reach scores.json.
    assert "_match" not in scores["gaps"]


# ---------------------------------------------------------------------------
# 9. The LLM judge — opt-in, and never allowed to break scoring
# ---------------------------------------------------------------------------


def test_the_judge_runs_only_when_asked_and_stays_beside_the_heuristics(monkeypatch):
    from evals.scoring import QuestionVerdict

    gt = _ground_truth()
    record = _question_record({"id": "q1", "gap_id": "g1", "text": GOOD_QUESTION, "turn": 1})

    seen: dict = {}

    def fake_call(prompt, model, **kwargs):
        seen["prompt"] = prompt
        seen["prompt_id"] = kwargs.get("prompt_id")
        return QuestionVerdict(
            addresses_gap=2, specific=1, understandable=2,
            has_context=1, not_duplicate=2, answerable=2, comment="ok",
        )

    monkeypatch.setattr("src.llm.call_structured", fake_call)

    assert "llm_judge" not in score_case(record, gt)["questions"]

    judged = score_case(record, gt, judge=True)["questions"]
    assert judged["llm_judge"]["judged"] == 1
    assert judged["llm_judge"]["mean_score"] == pytest.approx(10 / 6, abs=1e-3)
    # The deterministic scores are untouched by the judge's opinion.
    assert judged["mean_score"] == 2.0
    assert seen["prompt_id"] == "judge.question_quality"
    assert GOOD_QUESTION in seen["prompt"]


def test_a_judge_failure_costs_a_verdict_not_the_run(monkeypatch):
    gt = _ground_truth()
    record = _question_record({"id": "q1", "gap_id": "g1", "text": GOOD_QUESTION, "turn": 1})

    def boom(prompt, model, **kwargs):
        raise RuntimeError("backend down")

    monkeypatch.setattr("src.llm.call_structured", boom)

    judge = score_case(record, gt, judge=True)["questions"]["llm_judge"]

    assert (judge["judged"], judge["failed"]) == (0, 1)
    assert judge["mean_score"] is None


def test_the_judge_prompt_id_is_registered():
    """An unregistered id would raise mid-scoring instead of at import."""
    from src.prompts import version

    assert version("judge.question_quality")


# ---------------------------------------------------------------------------
# 10. The CLI, over a run directory on disk
# ---------------------------------------------------------------------------

CASE_ID = "cdc_009_reservation_salles"


def _write_run(root, *, decision_log_in_predictions: bool = True) -> None:
    """A minimal but real run directory: manifest + one finished case."""
    from evals.run_benchmark import _dumps, write_atomic

    root.mkdir(parents=True, exist_ok=True)
    write_atomic(
        root / "manifest.json",
        _dumps(
            {
                "run_id": root.name,
                "dataset_version": "v1",
                "case_ids": [CASE_ID],
                "simulator_mode": "oracle",
                "seed": 0,
                "git_commit": "abc123",
                "git_dirty": False,
                "llm": {"provider": "ollama", "model": "gpt-oss:20b"},
            }
        ),
    )

    decisions = [_decision("gap_detected", agent="gap_finder", output_ids=["g1"])]
    record = _record(
        case_id=CASE_ID,
        gaps=[_gap("g1", "Les règles d'annulation d'une réservation ne sont pas définies.",
                   category="business_rule")],
        context_items=[{"source": "user_answer"}],
        asked_questions=[{"id": "q1", "gap_id": "g1", "text": GOOD_QUESTION, "turn": 1}],
        transcript=[{"answers": [{"gap_id": "g1", "skip": False, "reason": "matched"}]}],
        decision_log=decisions if decision_log_in_predictions else [],
    )

    case_dir = root / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(case_dir / "predictions.json", _dumps(record))
    # Always written by CaseRecorder, and the fallback source for older runs.
    write_atomic(case_dir / "state.json", _dumps({"decision_log": decisions}))


def test_cli_writes_scores_json_csv_and_report(tmp_path, capsys):
    from evals import run_scoring

    _write_run(tmp_path / "unit_run")

    assert run_scoring.main(["--run", "unit_run", "--out", str(tmp_path)]) == 0

    root = tmp_path / "unit_run"
    payload = json.loads((root / "scores.json").read_text(encoding="utf-8"))
    assert payload["scorer_version"] == scoring.SCORER_VERSION
    assert payload["scored_cases"] == [CASE_ID]
    # The run's identity travels with the scores, so two files can be compared.
    assert payload["run_id"] == "unit_run"
    assert payload["git_commit"] == "abc123"
    assert payload["totals"]["cases"] == 1

    with open(root / "scores.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["case_id"] == CASE_ID
    assert set(run_scoring.SCORE_COLUMNS) == set(rows[0])

    report = (root / "scores.md").read_text(encoding="utf-8")
    assert "Benchmark scores" in report
    assert "Blocking-gap recall" in report
    assert "Unmatched predictions" in report
    assert "scored 1 case(s)" in capsys.readouterr().out


def test_cli_falls_back_to_state_json_for_an_older_run(tmp_path):
    """Runs written before predictions.json carried the decision log must still
    score their retrieval and contradictions, not silently report zero."""
    from evals import run_scoring

    _write_run(tmp_path / "old_run", decision_log_in_predictions=False)

    record = run_scoring.load_record(tmp_path / "old_run" / CASE_ID)

    assert record["decision_log"], "the fallback did not fire"
    assert run_scoring.main(["--run", "old_run", "--out", str(tmp_path)]) == 0


def test_cli_skips_a_case_that_never_finished(tmp_path, capsys):
    from evals import run_scoring

    root = tmp_path / "unit_run"
    _write_run(root)
    (root / CASE_ID / "predictions.json").unlink()

    assert run_scoring.main(["--run", "unit_run", "--out", str(tmp_path)]) == 1
    payload = json.loads((root / "scores.json").read_text(encoding="utf-8"))
    assert payload["skipped_cases"] == [CASE_ID]
    assert "no case has a predictions.json" in capsys.readouterr().err


def test_cli_reports_a_missing_run_instead_of_crashing(tmp_path, capsys):
    from evals import run_scoring

    assert run_scoring.main(["--run", "nope", "--out", str(tmp_path)]) == 1
    assert "No run at" in capsys.readouterr().err
    assert run_scoring.main(["--out", str(tmp_path / "empty")]) == 1


def test_scoring_never_writes_to_the_run_it_scores(tmp_path):
    """predictions.json and the transcripts are inputs — a scorer that mutated
    them would make the next scoring run mean something different."""
    from evals import run_scoring

    root = tmp_path / "unit_run"
    _write_run(root)
    before = (root / CASE_ID / "predictions.json").read_bytes()

    run_scoring.main(["--run", "unit_run", "--out", str(tmp_path)])

    assert (root / CASE_ID / "predictions.json").read_bytes() == before


def test_rerunning_the_scorer_is_idempotent(tmp_path):
    from evals import run_scoring

    _write_run(tmp_path / "unit_run")
    run_scoring.main(["--run", "unit_run", "--out", str(tmp_path)])
    first = (tmp_path / "unit_run" / "scores.csv").read_text(encoding="utf-8")

    run_scoring.main(["--run", "unit_run", "--out", str(tmp_path)])

    assert (tmp_path / "unit_run" / "scores.csv").read_text(encoding="utf-8") == first


def test_predictions_record_carries_the_decision_log():
    """Retrieval rank order lives nowhere else — if this key goes, Recall@K
    silently becomes zero rather than failing."""
    import inspect

    from evals.run_benchmark import run_case

    assert '"decision_log"' in inspect.getsource(run_case)
