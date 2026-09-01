"""Tests for the component-eval metric in evals/run_evals.py.

The point of the split reported there: a gap the finder never raised and a gap
it raised under the wrong category are both "not a strict hit", but they call
for opposite fixes — look harder vs. label better. A single blended F1 says
nothing about which one is happening, so `detection` (content match only) and
`strict` (content *and* category) are reported side by side and never merged.

Everything is offline: hand-built predictions shaped like `predicted_gaps()`.
"""

from __future__ import annotations

import json
from pathlib import Path

from evals.run_evals import (
    DATASETS_DIR,
    aggregate,
    category_ok,
    count_hits,
    hit_mark,
    match_gaps,
    matches,
)


def pred(description: str, category: str = "functional_ambiguity", **extra) -> dict:
    return {
        "id": f"gap_{abs(hash(description)) % 10**6}",
        "category": category,
        "severity": "blocking",
        "description": description,
        "section_ids": ["stock"],
        "follow_up_of_gap_id": None,
        **extra,
    }


EXPECTED = {
    "category": "integration",
    "severity": "blocking",
    "description": "Source of truth for ingredient stock is not identified",
    "keywords": ["m3", "source", "synchronisation", "maitre"],
}


# ---------------------------------------------------------------------------
# Matching is by content, not by category
# ---------------------------------------------------------------------------


def test_right_content_wrong_category_still_matches():
    found = pred("Le système maître du stock entre l'application et M3 n'est pas identifié")
    assert matches(EXPECTED, found, ignore_category=True)
    assert not matches(EXPECTED, found)  # strict reading disagrees, as it should
    assert not category_ok(EXPECTED, found)


def test_unrelated_gap_never_matches_however_lax():
    off_topic = pred("Aucun critère d'acceptation n'est défini", category="integration")
    # Same category as the annotation, but it is not the annotated finding.
    assert not matches(EXPECTED, off_topic, ignore_category=True)


def test_keyword_less_spec_keeps_the_category_gate():
    """`forbidden_gaps` and the critic dataset carry no keywords.

    With no content evidence the category is the only identifying signal, so
    ignore_category must not turn those specs into "matches anything".
    """
    forbidden = {"category": "nfr", "reason": "out of scope"}
    assert matches(forbidden, pred("Les temps de réponse ne sont pas chiffrés", category="nfr"),
                   ignore_category=True)
    assert not matches(forbidden, pred("Le stock maître n'est pas identifié", category="integration"),
                       ignore_category=True)


# ---------------------------------------------------------------------------
# The two recalls
# ---------------------------------------------------------------------------


def test_detection_and_strict_split_a_mislabelled_gap():
    pairs, unmatched = match_gaps(
        [EXPECTED], [pred("Le système maître du stock entre l'app et M3 n'est pas identifié")]
    )
    hits, strict_hits = count_hits(pairs)
    assert (hits, strict_hits) == (1, 0)
    # It was found: it is not a false positive either.
    assert unmatched == []
    assert hit_mark(*pairs[0]) == "CAT?"


def test_a_missed_gap_counts_in_neither():
    pairs, unmatched = match_gaps([EXPECTED], [pred("Aucun critère d'acceptation n'est défini")])
    hits, strict_hits = count_hits(pairs)
    assert (hits, strict_hits) == (0, 0)
    assert len(unmatched) == 1
    assert hit_mark(*pairs[0]) == "MISS"


def test_correctly_labelled_gap_counts_in_both():
    pairs, _ = match_gaps(
        [EXPECTED],
        [pred("La source de vérité du stock (app ou M3) n'est pas identifiée", category="integration")],
    )
    assert count_hits(pairs) == (1, 1)
    assert hit_mark(*pairs[0]) == "HIT "


def test_aggregate_reports_both_readings_and_the_mislabel_count():
    totals = {"hits": 6, "strict_hits": 4, "expected": 8, "actual": 12}
    metrics = aggregate(totals)

    assert metrics["detection"]["recall"] == 6 / 8
    assert metrics["strict"]["recall"] == 4 / 8
    assert metrics["detection"]["precision"] == 6 / 12
    assert metrics["mislabeled"] == 2
    assert metrics["category_agreement"] == 4 / 6
    # Never blended: the strict F1 alone is what the old single number was.
    assert metrics["strict"]["f1"] < metrics["detection"]["f1"]


def test_category_agreement_is_undefined_without_a_detection():
    assert aggregate({"hits": 0, "strict_hits": 0, "expected": 3, "actual": 0})["category_agreement"] is None


# ---------------------------------------------------------------------------
# The dataset has to carry the content annotation the metric relies on
# ---------------------------------------------------------------------------


def test_every_expected_gap_declares_keywords():
    path = Path(DATASETS_DIR) / "gap_finder.jsonl"
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for case in cases:
        expectations = (
            case.get("expected", {}).get("expected_new_gaps", [])
            if case.get("mode") == "fresh"
            else case.get("expected_gaps", [])
        )
        for exp in expectations:
            assert exp.get("keywords"), f"{case['case_id']}: an expectation without keywords matches anything"
