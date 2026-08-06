"""Ground-truth scoring of a benchmark run (roadmap Phase 4).

`run_benchmark.py` produces predictions; this turns them into scores. Every
function here is pure — it takes one case's `predictions.json` record plus its
`GroundTruth` and returns plain dicts. File I/O, the CLI and the report live in
`run_scoring.py`, so a scorer change can be replayed over an old run directory
without re-running the graph.

**Matching is by content, not by id.** Runtime gap ids are content hashes minted
during the run (`src/ids.py::stable_id`), so an annotation cannot name them in
advance. A predicted gap matches an annotated one when enough of the
annotation's `keywords` appear in the prediction — the same rule the oracle
simulator already uses to decide which question it is answering, reusing
`keyword_score` from evals/simulator.py so the two never drift apart.

**Precision is strict** (roadmap §7): a detected gap that matches no annotation
counts as a false positive, even when it is a perfectly real gap the annotator
did not write down. That makes precision a *lower bound*, which is why
`score_gaps` returns `unmatched_predictions` in full — the report prints them so
a low number can be checked against the annotations rather than believed.

Metric families, and the roadmap section each answers:

    score_gaps            §7  precision / recall / F1, by category
                          §8  severity confusion matrix, blocking-gap recall
    score_contradictions  §9  contradiction P/R/F1, critical recall
    score_retrieval       §10 Recall@K, MRR, retrieval-vs-reasoning split
    score_questions       §11 six-dimension question quality, questions per gap
    score_effort          §26 human intervention reduction
    score_completeness    §12 end-to-end improvement (deterministic proxy)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from evals.dataset import GroundTruth, GroundTruthGap
from evals.simulator import MATCH_THRESHOLD, keyword_score, normalize
from src.quality import score_section

# Bumped whenever the numbers a given run would produce change, so two
# scores.json files are only comparable when this agrees (roadmap §18/Phase 6).
SCORER_VERSION = "v1"

# Added to a candidate's ranking score when the annotation's section is among the
# prediction's. Ranking only — never a gate. A gap described in the right words
# but filed under the wrong section is still a detection, and calling it a miss
# would hide real recall behind a filing error.
SECTION_BONUS = 0.25

# Gap statuses that mean the loop actually closed the gap, rather than running
# out of turns with it still open.
CLOSED_STATUSES = {"rag_answered", "user_answered", "assumed", "resolved"}

SEVERITIES = ("blocking", "important", "nice_to_have")

RECALL_AT_K = (1, 3, 5)

# Short words and French function words carry no signal for overlap measures.
_STOPWORDS = {
    "avec", "cette", "comme", "dans", "des", "doit", "doivent", "elle", "est",
    "etre", "faut", "leur", "leurs", "mais", "meme", "nest", "nous", "ont",
    "ou", "par", "pas", "peut", "peuvent", "plus", "pour", "quand", "que",
    "quel", "quelle", "quelles", "quels", "qui", "sans", "ses", "son", "sont",
    "sur", "tout", "toute", "toutes", "tous", "une", "vous", "etes", "cela",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def content_tokens(text: str) -> set[str]:
    """Accent-folded words of 4+ characters, minus French function words.

    Used by the overlap-based question-quality dimensions, where the question
    being French means "de", "que" and "pour" would otherwise dominate.
    """
    return {
        tok
        for tok in _WORD_RE.findall(normalize(text))
        if len(tok) > 3 and tok not in _STOPWORDS
    }


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def covered(needle: set[str], haystack: set[str]) -> float:
    """Share of `needle` present in `haystack` — asymmetric, unlike jaccard."""
    return len(needle & haystack) / len(needle) if needle else 0.0


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@dataclass
class MatchResult:
    """One-to-one assignment of annotations to predictions."""

    pairs: dict[str, str] = field(default_factory=dict)  # gt id -> prediction id
    scores: dict[str, float] = field(default_factory=dict)  # gt id -> match score
    missed: list[str] = field(default_factory=list)  # annotated, never detected
    spurious: list[str] = field(default_factory=list)  # detected, not annotated

    @property
    def tp(self) -> int:
        return len(self.pairs)

    def prediction_for(self, gt_id: str) -> str | None:
        return self.pairs.get(gt_id)


def greedy_assign(
    candidates: list[tuple[float, str, str]], gt_ids: list[str], pred_ids: list[str]
) -> MatchResult:
    """Best-first one-to-one assignment over (score, gt_id, pred_id) candidates.

    Greedy rather than optimal (Hungarian): with a keyword threshold already
    applied the candidate sets are tiny and near-disjoint, and greedy keeps the
    result explainable — "this annotation matched that gap because it scored
    highest". Ties break on the ids, so a rerun assigns identically.
    """
    result = MatchResult()
    taken_pred: set[str] = set()

    for score, gt_id, pred_id in sorted(candidates, key=lambda c: (-c[0], c[1], c[2])):
        if gt_id in result.pairs or pred_id in taken_pred:
            continue
        result.pairs[gt_id] = pred_id
        result.scores[gt_id] = score
        taken_pred.add(pred_id)

    result.missed = [gid for gid in gt_ids if gid not in result.pairs]
    result.spurious = [pid for pid in pred_ids if pid not in taken_pred]
    return result


def gap_haystack(pred: dict) -> str:
    """Everything a prediction says about itself, for keyword matching."""
    return " ".join(
        [
            pred.get("description", ""),
            pred.get("category", ""),
            *(pred.get("section_ids") or []),
            pred.get("question_text") or "",
        ]
    )


def match_gaps(
    gt_gaps: list[GroundTruthGap], predictions: list[dict], threshold: float = MATCH_THRESHOLD
) -> MatchResult:
    candidates: list[tuple[float, str, str]] = []
    for gt in gt_gaps:
        for pred in predictions:
            score = keyword_score(gt.keywords, gap_haystack(pred))
            if score < threshold:
                continue
            if gt.section_id in (pred.get("section_ids") or []):
                score += SECTION_BONUS
            candidates.append((score, gt.id, pred["id"]))

    return greedy_assign(candidates, [g.id for g in gt_gaps], [p["id"] for p in predictions])


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


def prf(tp: int, fp: int, fn: int) -> dict:
    """Precision / recall / F1, with the convention run_evals.py already uses.

    An empty denominator scores 1.0: a case with no contradictions to find, on
    which the system reports none, is right — not undefined and not zero. This
    matters for `cdc_009_reservation_salles`, the negative control.
    """
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _ratio(numerator: float, denominator: float) -> float | None:
    """None, not 0.0, when there is nothing to divide — an absent measurement
    must not be averaged in as a bad one."""
    return round(numerator / denominator, 4) if denominator else None


# ---------------------------------------------------------------------------
# §7 / §8 — gap detection
# ---------------------------------------------------------------------------


def _counts_by(key: str, gt_gaps: list[GroundTruthGap], preds_by_id: dict[str, dict],
               match: MatchResult) -> dict[str, dict]:
    """Per-class TP/FP/FN, the standard multi-class way.

    A matched pair is a true positive for a class only when the annotation *and*
    the prediction both carry it. A pair that matched on content but disagrees on
    the label is a false negative for the annotated class and a false positive
    for the predicted one — counting it as a true positive for the annotated
    class while its label counts as a false positive elsewhere would mix two
    populations into one precision figure.

    Whether the gap was found at all, independent of how it was labelled, is
    `detection_recall` below and — for severity — the confusion matrix.
    """
    gt_by_id = {g.id: g for g in gt_gaps}
    buckets: dict[str, dict[str, int]] = {}

    def slot(name: str) -> dict[str, int]:
        return buckets.setdefault(name, {"tp": 0, "fp": 0, "fn": 0, "detected": 0, "annotated": 0})

    for gt in gt_gaps:
        slot(getattr(gt, key))["annotated"] += 1

    for gt_id, pred_id in match.pairs.items():
        annotated = getattr(gt_by_id[gt_id], key)
        predicted = preds_by_id[pred_id].get(key) or "?"
        slot(annotated)["detected"] += 1
        if predicted == annotated:
            slot(annotated)["tp"] += 1
        else:
            slot(annotated)["fn"] += 1
            slot(predicted)["fp"] += 1
    for gt_id in match.missed:
        slot(getattr(gt_by_id[gt_id], key))["fn"] += 1
    for pred_id in match.spurious:
        slot(preds_by_id[pred_id].get(key) or "?")["fp"] += 1

    return {
        name: {
            **prf(counts["tp"], counts["fp"], counts["fn"]),
            "annotated": counts["annotated"],
            # Found at all, whatever the system labelled it.
            "detection_recall": _ratio(counts["detected"], counts["annotated"]),
        }
        for name, counts in sorted(buckets.items())
    }


def score_gaps(record: dict, gt: GroundTruth, threshold: float = MATCH_THRESHOLD) -> dict:
    """Gap detection as a classification problem (roadmap §7 and §8).

    Contradictions are excluded on both sides: they are annotated in their own
    list and scored by `score_contradictions`, so counting them here too would
    double-count them.
    """
    predictions = [g for g in record.get("gaps", []) if g.get("category") != "contradiction"]
    preds_by_id = {p["id"]: p for p in predictions}
    match = match_gaps(gt.gaps, predictions, threshold)
    gt_by_id = {g.id: g for g in gt.gaps}

    confusion = {actual: {predicted: 0 for predicted in SEVERITIES} for actual in SEVERITIES}
    agreed_category = 0
    for gt_id, pred_id in match.pairs.items():
        annotation, prediction = gt_by_id[gt_id], preds_by_id[pred_id]
        predicted_severity = prediction.get("severity")
        if predicted_severity in confusion[annotation.severity]:
            confusion[annotation.severity][predicted_severity] += 1
        if prediction.get("category") == annotation.category:
            agreed_category += 1

    blocking = [g.id for g in gt.gaps if g.severity == "blocking"]
    blocking_found = sum(1 for gid in blocking if gid in match.pairs)

    return {
        "overall": prf(match.tp, len(match.spurious), len(match.missed)),
        "by_category": _counts_by("category", gt.gaps, preds_by_id, match),
        "by_severity": _counts_by("severity", gt.gaps, preds_by_id, match),
        # The safety metric of roadmap §8, and deliberately a *detection* recall:
        # what matters is that the blocking gap was surfaced to the author at
        # all. Whether the system also called it blocking is the confusion
        # matrix's job, and under-calling severity is a different, milder failure
        # than never mentioning the gap.
        "blocking_recall": _ratio(blocking_found, len(blocking)),
        "blocking_found": blocking_found,
        "blocking_total": len(blocking),
        "severity_confusion": confusion,
        "category_agreement": _ratio(agreed_category, match.tp),
        "annotated": len(gt.gaps),
        "predicted": len(predictions),
        "missed": [
            {
                "id": gid,
                "section_id": gt_by_id[gid].section_id,
                "category": gt_by_id[gid].category,
                "severity": gt_by_id[gid].severity,
                "description": gt_by_id[gid].description,
            }
            for gid in match.missed
        ],
        # Printed in full by the report: strict precision makes these the numbers
        # to eyeball before trusting the precision figure.
        "unmatched_predictions": [
            {
                "id": pid,
                "section_ids": preds_by_id[pid].get("section_ids") or [],
                "category": preds_by_id[pid].get("category"),
                "severity": preds_by_id[pid].get("severity"),
                "description": preds_by_id[pid].get("description", ""),
            }
            for pid in match.spurious
        ],
        "_match": match,  # consumed by score_retrieval / score_completeness
    }


# ---------------------------------------------------------------------------
# §9 — contradiction detection
# ---------------------------------------------------------------------------

# Above this token overlap, a contradiction reported by the final validator is
# taken to be a restatement of one already raised during the loop. Without the
# merge the same finding would be counted — and penalised — twice.
FINAL_DUPLICATE_THRESHOLD = 0.6

# Which agent produced a contradiction, and therefore what it is *about*. The
# benchmark annotates contradictions inside the initial CDC, so only the origins
# that read the CDC can be judged against it (see score_contradictions).
_CONTRADICTION_ORIGINS = {"gap_detected": "cdc", "contradiction_found": "answer"}
SCORED_ORIGINS = ("cdc", "final")


def collect_contradictions(record: dict) -> list[dict]:
    """Every contradiction the run reported, tagged with where it came from.

    Three producers, and they do not look at the same thing:

    * `cdc` — the gap finder reading the CDC text, which is the population the
      benchmark annotates.
    * `final` — the final validator's whole-document pass. It never becomes a
      gap at all (it lives in the `final_check` decision's details and in
      qa_report.md), so a scorer reading only `gaps` would credit none of them.
    * `answer` — the critic, comparing freshly integrated answers and assumptions
      against earlier context. Real work, but about material the initial CDC
      never contained.

    Attribution is by decision log: both producers stamp `output_ids=[gap.id]`.
    A contradiction gap with no matching decision (an old run with no decision
    log) is treated as `cdc`, the strict reading — better to over-count a false
    positive than to quietly inflate precision.
    """
    origin_by_gap: dict[str, str] = {}
    for entry in record.get("decision_log") or []:
        origin = _CONTRADICTION_ORIGINS.get(entry.get("decision_type"))
        if origin:
            for gap_id in entry.get("output_ids") or []:
                origin_by_gap.setdefault(gap_id, origin)

    found = [
        {
            "id": g["id"],
            "text": g.get("description", ""),
            "severity": g.get("severity"),
            "origin": origin_by_gap.get(g["id"], "cdc"),
        }
        for g in record.get("gaps", [])
        if g.get("category") == "contradiction"
    ]

    seen_tokens = [content_tokens(c["text"]) for c in found]
    for entry in record.get("decision_log") or []:
        if entry.get("decision_type") != "final_check":
            continue
        for i, text in enumerate((entry.get("details") or {}).get("contradictions") or []):
            tokens = content_tokens(text)
            if any(jaccard(tokens, seen) >= FINAL_DUPLICATE_THRESHOLD for seen in seen_tokens):
                continue
            found.append(
                {"id": f"final::{i}", "text": text, "severity": None, "origin": "final"}
            )
    return found


def score_contradictions(record: dict, gt: GroundTruth,
                         threshold: float = MATCH_THRESHOLD) -> dict:
    """Roadmap §9, with one asymmetry that the numbers depend on.

    `ground_truth.json` annotates contradictions **inside the initial CDC**:
    `statement_a` and `statement_b` are both quotes from it. The critic's
    findings are a different population — "this new answer contradicts what you
    said three turns ago" — which the benchmark says nothing about.

    So a critic finding may *match* an annotation (it counts as a true positive
    like any other), but an unmatched one is **not** a false positive: there is
    no annotation that could have made it right. Counting it as one would report
    near-zero precision on any case where the synthetic stakeholder contradicted
    itself, which is exactly what `--mode realistic` is built to make happen.
    Those findings are still reported, under `answer_level`.
    """
    predictions = collect_contradictions(record)
    preds_by_id = {p["id"]: p for p in predictions}

    candidates: list[tuple[float, str, str]] = []
    for annotation in gt.contradictions:
        statement_tokens = content_tokens(f"{annotation.statement_a} {annotation.statement_b}")
        for pred in predictions:
            score = keyword_score(annotation.keywords, pred["text"])
            if score < threshold:
                continue
            # Both statements of the contradiction should echo in a real match;
            # ranking only, so a paraphrase that keeps the keywords still counts.
            score += SECTION_BONUS * covered(statement_tokens, content_tokens(pred["text"]))
            candidates.append((score, annotation.id, pred["id"]))

    match = greedy_assign(
        candidates, [c.id for c in gt.contradictions], [p["id"] for p in predictions]
    )
    scored_spurious = [
        pid for pid in match.spurious if preds_by_id[pid]["origin"] in SCORED_ORIGINS
    ]
    answer_level = [
        pid for pid in match.spurious if preds_by_id[pid]["origin"] == "answer"
    ]

    critical = [c.id for c in gt.contradictions if c.severity == "blocking"]
    critical_found = sum(1 for cid in critical if cid in match.pairs)
    gt_by_id = {c.id: c for c in gt.contradictions}

    return {
        "overall": prf(match.tp, len(scored_spurious), len(match.missed)),
        "critical_recall": _ratio(critical_found, len(critical)),
        "critical_found": critical_found,
        "critical_total": len(critical),
        "annotated": len(gt.contradictions),
        "predicted": len(predictions),
        "by_origin": dict(Counter(p["origin"] for p in predictions)),
        # Detected between the run's own answers, so outside what the benchmark
        # annotates: counted and shown, never scored.
        "answer_level": len(answer_level),
        "missed": [
            {
                "id": cid,
                "statement_a": gt_by_id[cid].statement_a,
                "statement_b": gt_by_id[cid].statement_b,
                "severity": gt_by_id[cid].severity,
            }
            for cid in match.missed
        ],
        "unmatched_predictions": [
            {"id": pid, "origin": preds_by_id[pid]["origin"], "description": preds_by_id[pid]["text"]}
            for pid in scored_spurious
        ],
    }


# ---------------------------------------------------------------------------
# §10 — RAG as a retrieval system
# ---------------------------------------------------------------------------


def retrievals_by_gap(record: dict) -> dict[str, dict]:
    """gap id -> {"verdict", "chunk_ids"} for every gap RAG was tried on.

    The rank order is `evidence_ids`, which is the order `src/rag.py::retrieve`
    returned the hits in (see gap_filler). A *rejected* retrieval leaves no
    ContextItem behind, so the decision log is the only place its hits survive —
    and dropping those cases would measure retrieval only where it happened to
    work.
    """
    retrievals: dict[str, dict] = {}
    for entry in record.get("decision_log") or []:
        verdict = entry.get("decision_type")
        if verdict not in ("rag_answer", "rag_rejected"):
            continue
        for gap_id in entry.get("input_ids") or []:
            # First attempt wins: a gap is only ever RAG'd once (gap.rag_attempted).
            retrievals.setdefault(
                gap_id, {"verdict": verdict, "chunk_ids": list(entry.get("evidence_ids") or [])}
            )
    return retrievals


def chunk_matches(chunk_id: str, document: str, page: int | None) -> bool:
    """Is this retrieved chunk the annotated evidence?

    Chunk ids are `document::index` or `document::p{page}::index`
    (src/rag.py::_chunk_id), so the document is a prefix test and the page, when
    the annotation pins one, an infix test.
    """
    if not chunk_id.startswith(f"{document}::"):
        return False
    return page is None or f"::p{page}::" in chunk_id


def first_relevant_rank(chunk_ids: list[str], document: str, page: int | None) -> int | None:
    """1-based rank of the first annotated chunk, or None if it never came back."""
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        if chunk_matches(chunk_id, document, page):
            return rank
    return None


def score_retrieval(record: dict, gt: GroundTruth, match: MatchResult) -> dict:
    """Recall@K and MRR over the gaps the documents are supposed to answer.

    Two denominators, because they answer different questions:

    * `attempted` — of the retrievals that actually ran, how good is retrieval?
      This is the number that says whether the index and the embeddings work.
    * `annotated` — of every gap the documents *could* have closed, how many did
      the system pull the right chunk for? A gap it never detected is counted as
      a miss here, because from the user's point of view the document went unread.

    `sufficiency_judgment` then splits roadmap §10's two failure modes: retrieval
    that never surfaced the right chunk (a retrieval problem) from retrieval that
    did, only for the grader to throw it away (a reasoning problem).
    """
    annotated = [g for g in gt.gaps if g.resolvable_by == "rag" and g.expected_evidence]
    retrievals = retrievals_by_gap(record)

    per_gap: list[dict] = []
    for annotation in annotated:
        evidence = annotation.expected_evidence
        pred_id = match.prediction_for(annotation.id)
        retrieval = retrievals.get(pred_id) if pred_id else None
        rank = (
            first_relevant_rank(retrieval["chunk_ids"], evidence.document, evidence.page)
            if retrieval
            else None
        )
        per_gap.append(
            {
                "gt_id": annotation.id,
                "gap_id": pred_id,
                "document": evidence.document,
                "page": evidence.page,
                "detected": pred_id is not None,
                "attempted": retrieval is not None,
                "verdict": retrieval["verdict"] if retrieval else None,
                "retrieved": len(retrieval["chunk_ids"]) if retrieval else 0,
                "rank": rank,
            }
        )

    attempted = [g for g in per_gap if g["attempted"]]

    def at_k(rows: list[dict], k: int) -> float | None:
        return _ratio(sum(1 for g in rows if g["rank"] is not None and g["rank"] <= k), len(rows))

    def mrr(rows: list[dict]) -> float | None:
        return _ratio(sum(1 / g["rank"] for g in rows if g["rank"]), len(rows))

    found = [g for g in attempted if g["rank"] is not None]
    absent = [g for g in attempted if g["rank"] is None]
    judgment = {
        # Retrieval worked and the grader kept it — the whole chain did its job.
        "correct_accept": sum(1 for g in found if g["verdict"] == "rag_answer"),
        # Retrieval worked, the grader threw the evidence away: a reasoning problem.
        "wrong_reject": sum(1 for g in found if g["verdict"] == "rag_rejected"),
        # The annotated document never came back and the grader said so.
        "correct_reject": sum(1 for g in absent if g["verdict"] == "rag_rejected"),
        # Answered from something other than the annotated source. Not necessarily
        # wrong — another document may say the same thing — so it is reported, not
        # scored.
        "accepted_other_source": sum(1 for g in absent if g["verdict"] == "rag_answer"),
    }
    judgment["accuracy"] = _ratio(
        judgment["correct_accept"] + judgment["correct_reject"], len(attempted)
    )

    return {
        "annotated": len(annotated),
        "detected": sum(1 for g in per_gap if g["detected"]),
        "attempted": len(attempted),
        "recall_at_k": {f"@{k}": at_k(attempted, k) for k in RECALL_AT_K},
        "mrr": mrr(attempted),
        "recall_at_k_overall": {f"@{k}": at_k(per_gap, k) for k in RECALL_AT_K},
        "mrr_overall": mrr(per_gap),
        "sufficiency_judgment": judgment,
        "per_gap": per_gap,
    }


# ---------------------------------------------------------------------------
# §11 — question quality
# ---------------------------------------------------------------------------

QUESTION_DIMENSIONS = (
    "addresses_gap",
    "specific",
    "understandable",
    "has_context",
    "not_duplicate",
    "answerable",
)

_QUOTE_CHARS = ("«", "»", '"', "“", "”")


def _score_addresses_gap(question: str, annotation: GroundTruthGap | None, gap: dict | None) -> int:
    """Does the question actually ask about the gap it was drafted for?"""
    if annotation is not None:
        score = keyword_score(annotation.keywords, question)
        return 2 if score >= 0.5 else (1 if score > 0 else 0)
    if gap is None:
        return 0
    # No annotation to check against: fall back to the system's own description.
    share = covered(content_tokens(gap.get("description", "")), content_tokens(question))
    return 2 if share >= 0.3 else (1 if share > 0 else 0)


def _score_specific(question: str) -> int:
    """A question that quotes the ambiguous text or names a figure is specific;
    "pouvez-vous préciser le périmètre ?" is not — the drafting prompt forbids
    exactly that, so this is the check that it was obeyed."""
    quotes = any(ch in question for ch in _QUOTE_CHARS)
    digits = any(ch.isdigit() for ch in question)
    if (quotes or digits) and len(question) >= 80:
        return 2
    return 1 if len(question) >= 60 else 0


def _score_understandable(question: str) -> int:
    words = len(question.split())
    if words <= 45 and question.count("?") == 1:
        return 2
    return 1 if words <= 70 else 0


def _score_has_context(question: str, gap: dict | None, initial: "_InitialCdc") -> int:
    """Does it reference the CDC, or could it have been asked about any project?"""
    if gap is None:
        return 0
    shared = len(content_tokens(question) & initial.pool_for(gap.get("section_ids") or []))
    return 2 if shared >= 3 else (1 if shared >= 1 else 0)


@dataclass
class _InitialCdc:
    """The initial CDC's wording, indexed by section, for the context check.

    `untagged` matters more than it looks: `ingest` only tags the chunks it can
    map to a section, so the CDC's title and its intro prose — often the most
    quotable text in a one-page CDC — carry no section at all. Scoring a
    question only against its own section's slice marked those quotes as
    contextless, and any section the splitter never populated scored a flat zero
    for every question asked about it.
    """

    by_section: dict[str, set[str]] = field(default_factory=dict)
    untagged: set[str] = field(default_factory=set)
    everything: set[str] = field(default_factory=set)

    @classmethod
    def from_record(cls, record: dict) -> "_InitialCdc":
        index = cls()
        for item in record.get("context_items") or []:
            if item.get("source") != "initial_cdc":
                continue
            tokens = content_tokens(item.get("content", ""))
            index.everything |= tokens
            sections = item.get("section_ids") or []
            if not sections:
                index.untagged |= tokens
            for section_id in sections:
                index.by_section.setdefault(section_id, set()).update(tokens)
        return index

    def pool_for(self, section_ids: list[str]) -> set[str]:
        pool = set(self.untagged)
        for section_id in section_ids:
            pool |= self.by_section.get(section_id, set())
        # The gap's sections hold nothing quotable: judge against the whole CDC
        # rather than scoring every question about them as contextless.
        return pool or self.everything


def _score_not_duplicate(question: str, earlier: list[set[str]]) -> int:
    tokens = content_tokens(question)
    worst = max((jaccard(tokens, seen) for seen in earlier), default=0.0)
    return 2 if worst < 0.4 else (1 if worst < 0.6 else 0)


def _score_answerable(reply: dict | None) -> int | None:
    """Read off the run: could the stakeholder answer it at all?

    None when the question was drafted but never reached the simulator (the run
    stopped first) — an unmeasured dimension must not be averaged in as a zero.
    """
    if reply is None:
        return None
    if not reply.get("skip"):
        return 2
    # Hedging is the stakeholder being evasive; the other reasons mean the
    # question was genuinely unanswerable by this stakeholder.
    return 1 if reply.get("reason") == "hedged" else 0


def score_questions(record: dict, gt: GroundTruth, match: MatchResult) -> dict:
    """Roadmap §11: six dimensions at 0 / 1 / 2, plus question efficiency.

    Deterministic on purpose — scoring a run costs nothing and repeats exactly.
    `run_scoring.py --judge llm` adds an LLM rubric alongside these numbers
    without replacing them (roadmap §23: keep AI judgment separate from
    deterministic validation).
    """
    gaps_by_id = {g["id"]: g for g in record.get("gaps", [])}
    gt_by_pred = {pred_id: gt_id for gt_id, pred_id in match.pairs.items()}
    gt_gaps_by_id = {g.id: g for g in gt.gaps}

    # The stakeholder's reply per gap; a re-asked gap keeps the latest one.
    replies: dict[str, dict] = {}
    for round_ in record.get("transcript") or []:
        for answer in round_.get("answers") or []:
            replies[answer["gap_id"]] = answer

    initial = _InitialCdc.from_record(record)

    per_question: list[dict] = []
    seen_tokens: list[set[str]] = []
    for asked in sorted(record.get("asked_questions") or [], key=lambda q: q.get("turn", 0)):
        text = asked.get("text", "")
        gap = gaps_by_id.get(asked.get("gap_id"))
        annotation = gt_gaps_by_id.get(gt_by_pred.get(asked.get("gap_id"), ""))

        scores = {
            "addresses_gap": _score_addresses_gap(text, annotation, gap),
            "specific": _score_specific(text),
            "understandable": _score_understandable(text),
            "has_context": _score_has_context(text, gap, initial),
            "not_duplicate": _score_not_duplicate(text, seen_tokens),
            "answerable": _score_answerable(replies.get(asked.get("gap_id"))),
        }
        seen_tokens.append(content_tokens(text))
        measured = [v for v in scores.values() if v is not None]
        per_question.append(
            {
                "id": asked.get("id"),
                "gap_id": asked.get("gap_id"),
                "turn": asked.get("turn"),
                "matched_gt": gt_by_pred.get(asked.get("gap_id")),
                "scores": scores,
                "mean": _mean([float(v) for v in measured]),
            }
        )

    by_dimension = {
        dim: _mean([float(q["scores"][dim]) for q in per_question if q["scores"][dim] is not None])
        for dim in QUESTION_DIMENSIONS
    }
    overall = _mean([q["mean"] for q in per_question if q["mean"] is not None])

    closed = [g for g in record.get("gaps", []) if g.get("status") in CLOSED_STATUSES]
    answered = [g for g in record.get("gaps", []) if g.get("status") == "user_answered"]

    return {
        "asked": len(per_question),
        "by_dimension": by_dimension,
        "mean_score": overall,  # 0–2
        "quality": round(overall / 2, 4) if overall is not None else None,  # 0–1
        # Roadmap §11's efficiency measure: the goal is to minimise this without
        # losing completeness.
        "questions_per_resolved_gap": _ratio(len(per_question), len(closed)),
        "questions_per_answered_gap": _ratio(len(per_question), len(answered)),
        "resolved_gaps": len(closed),
        "per_question": per_question,
    }


# ---------------------------------------------------------------------------
# §11 (second opinion) — the optional LLM judge
# ---------------------------------------------------------------------------


class QuestionVerdict(BaseModel):
    """One judged question, on the same 0/1/2 scale as the heuristics."""

    addresses_gap: int = Field(ge=0, le=2)
    specific: int = Field(ge=0, le=2)
    understandable: int = Field(ge=0, le=2)
    has_context: int = Field(ge=0, le=2)
    not_duplicate: int = Field(ge=0, le=2)
    answerable: int = Field(ge=0, le=2)
    comment: str = ""


def build_judge_prompt(question: str, gap: dict | None, earlier: list[str]) -> str:
    gap_block = "(lacune inconnue)"
    if gap is not None:
        gap_block = (
            f"- catégorie : {gap.get('category')}\n"
            f"- sévérité : {gap.get('severity')}\n"
            f"- sections : {gap.get('section_ids')}\n"
            f"- description : {gap.get('description')}"
        )
    previous = "\n".join(f"- {q}" for q in earlier[-10:]) or "- (aucune)"

    return f"""Tu évalues la QUALITÉ d'une question posée par un agent d'affinage de cahier des
charges à la partie prenante. Tu ne réponds pas à la question : tu la notes.

Lacune que la question est censée lever :
{gap_block}

Questions déjà posées avant celle-ci :
{previous}

Question à évaluer :
"{question}"

Note chaque dimension sur 0 (mauvais), 1 (acceptable) ou 2 (bon) :
- addresses_gap : la question porte bien sur la lacune ci-dessus, et pas sur autre chose.
- specific : elle cite ou paraphrase l'élément ambigu concret, au lieu de rester générique
  (« pouvez-vous préciser le périmètre ? » vaut 0).
- understandable : formulation claire, une seule question, sans jargon inutile.
- has_context : elle rappelle assez de contexte du CDC pour être comprise seule.
- not_duplicate : elle ne redemande pas ce qui a déjà été posé ci-dessus.
- answerable : une partie prenante métier peut y répondre sans faire d'étude technique.

Sois sévère et cohérent : une note de complaisance rend la mesure inutile. Ajoute un
`comment` d'une phrase justifiant la note la plus basse."""


def judge_questions(record: dict, per_question: list[dict]) -> dict:
    """LLM second opinion on question quality (roadmap §11), never the default.

    Reported beside the deterministic scores, never merged into them: roadmap
    §23 wants AI judgment kept separate from deterministic validation, and an
    LLM grading the system that shares its model is weak evidence on its own.
    A judge failure degrades to a missing verdict rather than losing the run's
    scores.
    """
    from src.llm import call_structured  # local: keeps the pure path import-free

    gaps_by_id = {g["id"]: g for g in record.get("gaps", [])}
    asked_by_id = {q["id"]: q for q in record.get("asked_questions") or []}

    verdicts: list[dict] = []
    earlier: list[str] = []
    for entry in per_question:
        asked = asked_by_id.get(entry["id"])
        if asked is None:
            continue
        text = asked.get("text", "")
        prompt = build_judge_prompt(text, gaps_by_id.get(entry["gap_id"]), earlier)
        earlier.append(text)
        try:
            verdict = call_structured(prompt, QuestionVerdict, prompt_id="judge.question_quality")
        except Exception as exc:  # a judge failure must not cost the whole score
            verdicts.append({"id": entry["id"], "error": str(exc)})
            continue
        scores = verdict.model_dump(exclude={"comment"})
        verdicts.append(
            {
                "id": entry["id"],
                "gap_id": entry["gap_id"],
                "scores": scores,
                "mean": _mean([float(v) for v in scores.values()]),
                "comment": verdict.comment,
            }
        )

    scored = [v for v in verdicts if "scores" in v]
    by_dimension = {
        dim: _mean([float(v["scores"][dim]) for v in scored]) for dim in QUESTION_DIMENSIONS
    }
    mean_score = _mean([v["mean"] for v in scored if v["mean"] is not None])

    return {
        "judged": len(scored),
        "failed": len(verdicts) - len(scored),
        "by_dimension": by_dimension,
        "mean_score": mean_score,
        "quality": round(mean_score / 2, 4) if mean_score is not None else None,
        "per_question": verdicts,
    }


# ---------------------------------------------------------------------------
# §26 — human effort saved
# ---------------------------------------------------------------------------


def score_effort(record: dict) -> dict:
    """How much of the answering RAG did instead of the human.

    The denominator is every gap that got closed by *some* piece of information,
    matching roadmap §26's worked example (35 RAG / 100 total = 35%).
    """
    sources = Counter(c.get("source") for c in record.get("context_items") or [])
    by_rag = sources["rag"]
    by_human = sources["user_answer"]
    assumed = sources["assumption"]
    total = by_rag + by_human + assumed

    unknown = sum(
        1
        for round_ in record.get("transcript") or []
        for answer in round_.get("answers") or []
        if answer.get("skip")
    )

    return {
        "resolved_by_rag": by_rag,
        "resolved_by_human": by_human,
        "assumed": assumed,
        "resolutions": total,
        "human_intervention_reduction": _ratio(by_rag, total),
        "questions_asked": len(record.get("asked_questions") or []),
        "unknown_answers": unknown,
        "turns": record.get("turns", 0),
        "rounds": record.get("rounds", 0),
        "wall_s": record.get("wall_s", 0.0),
    }


# ---------------------------------------------------------------------------
# §12 — end-to-end improvement (deterministic proxy)
# ---------------------------------------------------------------------------


def score_completeness(record: dict, gt: GroundTruth, match: MatchResult) -> dict:
    """Did the CDC actually get better?

    Roadmap §12 asks for a human expert to score the initial and the final
    document on eight dimensions. The benchmark deliberately carries no
    expert-scored reference document (Phase 2's note: it would be one
    annotator's prose, not ground truth), so this is a **proxy** computed from
    the run itself, and must be read as one:

    * `gt_gap_coverage` — the honest end-to-end number: of the gaps a human
      annotated, how many did the system both find *and* close?
    * `quality_score_*` — `src/quality.py::score_section` applied twice per
      section: once treating every detected gap as outstanding (the initial CDC,
      as the system saw it) and once counting only the gaps still open at the
      end. The delta is how much defect weight the loop removed.
    """
    gaps = record.get("gaps", [])
    gaps_by_id = {g["id"]: g for g in gaps}

    def closed(gt_id: str) -> bool:
        pred = gaps_by_id.get(match.pairs.get(gt_id, ""))
        return bool(pred) and pred.get("status") in CLOSED_STATUSES

    blocking = [g.id for g in gt.gaps if g.severity == "blocking"]
    still_open = [g for g in gaps if g.get("status") == "open"]
    deferred = [g for g in gaps if g.get("status") == "deferred"]

    sections = record.get("section_statuses") or {}
    initial_scores, final_scores = [], []
    for section_id in sections:
        in_section = [g for g in gaps if section_id in (g.get("section_ids") or [])]
        initial_scores.append(score_section(in_section))
        final_scores.append(
            score_section([g for g in in_section if g.get("status") in ("open", "deferred")])
        )

    initial = _mean(initial_scores)
    final = _mean(final_scores)

    return {
        "gt_gap_coverage": _ratio(sum(1 for g in gt.gaps if closed(g.id)), len(gt.gaps)),
        "gt_gap_detection": _ratio(match.tp, len(gt.gaps)),
        "blocking_coverage": _ratio(sum(1 for gid in blocking if closed(gid)), len(blocking)),
        "open_gap_reduction": _ratio(len(gaps) - len(still_open), len(gaps)),
        "gaps_still_open": len(still_open),
        "gaps_deferred": len(deferred),
        "sections_complete": sum(1 for s in sections.values() if s.get("status") == "complete"),
        "sections_total": len(sections),
        "sections_complete_ratio": _ratio(
            sum(1 for s in sections.values() if s.get("status") == "complete"), len(sections)
        ),
        "quality_score_initial": initial,
        "quality_score_final": final,
        "quality_score_delta": (
            round(final - initial, 4) if initial is not None and final is not None else None
        ),
    }


# ---------------------------------------------------------------------------
# One case, then the whole run
# ---------------------------------------------------------------------------


def score_case(
    record: dict, gt: GroundTruth, threshold: float = MATCH_THRESHOLD, judge: bool = False
) -> dict:
    """Every metric family for one case.

    Pure and offline by default. `judge=True` is the one exception: it spends an
    LLM call per asked question and files the result under
    `questions["llm_judge"]`, next to the deterministic scores rather than on
    top of them.
    """
    gaps = score_gaps(record, gt, threshold)
    match: MatchResult = gaps.pop("_match")
    questions = score_questions(record, gt, match)
    if judge:
        questions["llm_judge"] = judge_questions(record, questions["per_question"])

    return {
        "case_id": record.get("case_id"),
        "title": record.get("title"),
        "status": record.get("status"),
        "finished": record.get("finished"),
        "gaps": gaps,
        "contradictions": score_contradictions(record, gt, threshold),
        "retrieval": score_retrieval(record, gt, match),
        "questions": questions,
        "effort": score_effort(record),
        "completeness": score_completeness(record, gt, match),
    }


def _pool(cases: list[dict], *path: str) -> dict:
    """Micro-average: sum TP/FP/FN across cases, then compute P/R/F1 once.

    Micro is the headline because it weights every annotated gap equally; a case
    with 20 gaps should count for more than the negative control's three. The
    macro mean sits beside it in `aggregate`.
    """
    totals = Counter()
    for case in cases:
        node = case
        for key in path:
            node = node.get(key, {})
        for key in ("tp", "fp", "fn"):
            totals[key] += node.get(key, 0)
    return prf(totals["tp"], totals["fp"], totals["fn"])


def _pooled_by_class(cases: list[dict], family: str, group: str) -> dict:
    classes: dict[str, Counter] = {}
    for case in cases:
        for name, slot in (case[family].get(group) or {}).items():
            bucket = classes.setdefault(name, Counter())
            for key in ("tp", "fp", "fn", "annotated"):
                bucket[key] += slot.get(key, 0)
            # detection_recall is a ratio, so re-derive it from its numerator
            # rather than averaging ten cases' percentages.
            bucket["detected"] += round((slot.get("detection_recall") or 0.0)
                                        * slot.get("annotated", 0))
    return {
        name: {
            **prf(c["tp"], c["fp"], c["fn"]),
            "annotated": c["annotated"],
            "detection_recall": _ratio(c["detected"], c["annotated"]),
        }
        for name, c in sorted(classes.items())
    }


def aggregate(cases: list[dict]) -> dict:
    """Run-level numbers. Micro-averaged, with the macro mean alongside."""
    if not cases:
        return {}

    def total(family: str, key: str) -> int:
        return sum(case[family].get(key, 0) for case in cases)

    def macro(family: str) -> float | None:
        return _mean([case[family]["overall"]["f1"] for case in cases])

    confusion = {actual: {pred: 0 for pred in SEVERITIES} for actual in SEVERITIES}
    for case in cases:
        for actual, row in case["gaps"]["severity_confusion"].items():
            for predicted, count in row.items():
                confusion[actual][predicted] += count

    blocking_found = total("gaps", "blocking_found")
    blocking_total = total("gaps", "blocking_total")
    critical_found = total("contradictions", "critical_found")
    critical_total = total("contradictions", "critical_total")

    retrieval_rows = [row for case in cases for row in case["retrieval"]["per_gap"]]
    attempted = [row for row in retrieval_rows if row["attempted"]]

    def at_k(rows: list[dict], k: int) -> float | None:
        return _ratio(sum(1 for r in rows if r["rank"] is not None and r["rank"] <= k), len(rows))

    def mrr(rows: list[dict]) -> float | None:
        return _ratio(sum(1 / r["rank"] for r in rows if r["rank"]), len(rows))

    judgment = Counter()
    for case in cases:
        for key, value in case["retrieval"]["sufficiency_judgment"].items():
            if key != "accuracy":
                judgment[key] += value
    judgment_total = len(attempted)

    quality_initial = _mean(
        [c["completeness"]["quality_score_initial"] for c in cases
         if c["completeness"]["quality_score_initial"] is not None]
    )
    quality_final = _mean(
        [c["completeness"]["quality_score_final"] for c in cases
         if c["completeness"]["quality_score_final"] is not None]
    )

    questions = [q for case in cases for q in case["questions"]["per_question"]]
    by_dimension = {
        dim: _mean([float(q["scores"][dim]) for q in questions if q["scores"][dim] is not None])
        for dim in QUESTION_DIMENSIONS
    }
    question_mean = _mean([q["mean"] for q in questions if q["mean"] is not None])

    by_rag = total("effort", "resolved_by_rag")
    by_human = total("effort", "resolved_by_human")
    assumed = total("effort", "assumed")

    closed_gaps = total("questions", "resolved_gaps")

    return {
        "cases": len(cases),
        "gaps": {
            "micro": _pool(cases, "gaps", "overall"),
            "macro_f1": macro("gaps"),
            "by_category": _pooled_by_class(cases, "gaps", "by_category"),
            "by_severity": _pooled_by_class(cases, "gaps", "by_severity"),
            "blocking_recall": _ratio(blocking_found, blocking_total),
            "blocking_found": blocking_found,
            "blocking_total": blocking_total,
            "severity_confusion": confusion,
            "annotated": total("gaps", "annotated"),
            "predicted": total("gaps", "predicted"),
        },
        "contradictions": {
            "micro": _pool(cases, "contradictions", "overall"),
            "macro_f1": macro("contradictions"),
            "critical_recall": _ratio(critical_found, critical_total),
            "critical_found": critical_found,
            "critical_total": critical_total,
            "annotated": total("contradictions", "annotated"),
            "predicted": total("contradictions", "predicted"),
            "answer_level": total("contradictions", "answer_level"),
        },
        "retrieval": {
            "annotated": len(retrieval_rows),
            "detected": sum(1 for r in retrieval_rows if r["detected"]),
            "attempted": len(attempted),
            "recall_at_k": {f"@{k}": at_k(attempted, k) for k in RECALL_AT_K},
            "mrr": mrr(attempted),
            "recall_at_k_overall": {f"@{k}": at_k(retrieval_rows, k) for k in RECALL_AT_K},
            "mrr_overall": mrr(retrieval_rows),
            "sufficiency_judgment": {
                **judgment,
                "accuracy": _ratio(
                    judgment["correct_accept"] + judgment["correct_reject"], judgment_total
                ),
            },
        },
        "questions": {
            "asked": len(questions),
            "by_dimension": by_dimension,
            "mean_score": question_mean,
            "quality": round(question_mean / 2, 4) if question_mean is not None else None,
            "questions_per_resolved_gap": _ratio(len(questions), closed_gaps),
        },
        "effort": {
            "resolved_by_rag": by_rag,
            "resolved_by_human": by_human,
            "assumed": assumed,
            "resolutions": by_rag + by_human + assumed,
            "human_intervention_reduction": _ratio(by_rag, by_rag + by_human + assumed),
            "questions_per_case": _ratio(total("effort", "questions_asked"), len(cases)),
            "turns_per_case": _ratio(total("effort", "turns"), len(cases)),
            "wall_s_per_case": _ratio(
                sum(case["effort"].get("wall_s", 0.0) for case in cases), len(cases)
            ),
        },
        "completeness": {
            "gt_gap_coverage": _mean(
                [c["completeness"]["gt_gap_coverage"] for c in cases
                 if c["completeness"]["gt_gap_coverage"] is not None]
            ),
            "blocking_coverage": _mean(
                [c["completeness"]["blocking_coverage"] for c in cases
                 if c["completeness"]["blocking_coverage"] is not None]
            ),
            "sections_complete": total("completeness", "sections_complete"),
            "sections_total": total("completeness", "sections_total"),
            "quality_score_initial": quality_initial,
            "quality_score_final": quality_final,
            "quality_score_delta": (
                round(quality_final - quality_initial, 4)
                if quality_initial is not None and quality_final is not None
                else None
            ),
        },
    }
