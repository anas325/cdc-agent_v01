"""Score a benchmark run against ground truth (roadmap Phase 4).

`run_benchmark.py` writes what the system *found*; this says how good that was.
It reads a run directory's `predictions.json` files plus the matching
`ground_truth.json` annotations and writes `scores.json`, `scores.csv` and
`scores.md` back into the same directory.

    uv run python evals/run_scoring.py                          # the most recent run
    uv run python evals/run_scoring.py --run bench_20260805_130607
    uv run python evals/run_scoring.py --cases cdc_003_ecommerce
    uv run python evals/run_scoring.py --judge llm              # add the LLM judge

Deliberately a **separate pass**, not something folded into the run: scoring is
cheap and offline, the scorer will keep improving, and re-scoring a two-hour run
with a better scorer must not mean running it again. It never writes anything a
run produced — `predictions.json`, the transcripts and the checkpoints are
inputs only.

All the arithmetic lives in evals/scoring.py; this module is I/O, the CLI, and
the report.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from evals import scoring  # noqa: E402
from evals.dataset import BenchmarkCase, load_benchmark  # noqa: E402
from evals.run_benchmark import (  # noqa: E402
    RESULTS_DIR,
    _dumps,
    _read_json,
    latest_run,
    write_atomic,
)

# One flat row per case. Keep in step with `case_row` — DictWriter blanks a
# column it is not given rather than complaining.
SCORE_COLUMNS = [
    "case_id",
    "status",
    "gap_precision",
    "gap_recall",
    "gap_f1",
    "gaps_annotated",
    "gaps_predicted",
    "blocking_recall",
    "con_precision",
    "con_recall",
    "con_f1",
    "con_annotated",
    "rag_annotated",
    "rag_attempted",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "mrr",
    "questions_asked",
    "question_quality",
    "questions_per_resolved_gap",
    "human_intervention_reduction",
    "gt_gap_coverage",
    "quality_score_initial",
    "quality_score_final",
    "quality_score_delta",
]


def _fmt(value, digits: int = 2, dash: str = "—") -> str:
    """Render a metric for the report. None means *not measured* — print the
    dash rather than a 0.00 that reads like a failure."""
    if value is None:
        return dash
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


# ---------------------------------------------------------------------------
# Reading a run back
# ---------------------------------------------------------------------------


def load_record(case_dir: Path) -> dict | None:
    """One case's predictions, or None if the case never finished.

    Runs written before `decision_log` joined the record fall back to
    `state.json`, which `CaseRecorder.record_state` has always written after
    every graph turn. Without it retrieval and the final validator's
    contradictions would silently score as zero on an older run — a wrong number
    is worse than a missing one.
    """
    record = _read_json(case_dir / "predictions.json")
    if not isinstance(record, dict) or "case_id" not in record:
        return None
    if not record.get("decision_log"):
        state = _read_json(case_dir / "state.json") or {}
        record["decision_log"] = state.get("decision_log") or []
    return record


def collect(results_root: Path, cases: list[BenchmarkCase]) -> tuple[list[dict], list[str]]:
    """(records paired with their case, ids that had nothing to score)."""
    pairs, skipped = [], []
    for case in cases:
        record = load_record(results_root / case.case_id)
        if record is None:
            skipped.append(case.case_id)
            continue
        pairs.append({"case": case, "record": record})
    return pairs, skipped


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def case_row(case_score: dict) -> dict:
    gaps = case_score["gaps"]
    con = case_score["contradictions"]
    rag = case_score["retrieval"]
    questions = case_score["questions"]
    effort = case_score["effort"]
    completeness = case_score["completeness"]

    return {
        "case_id": case_score["case_id"],
        "status": case_score["status"],
        "gap_precision": gaps["overall"]["precision"],
        "gap_recall": gaps["overall"]["recall"],
        "gap_f1": gaps["overall"]["f1"],
        "gaps_annotated": gaps["annotated"],
        "gaps_predicted": gaps["predicted"],
        "blocking_recall": gaps["blocking_recall"],
        "con_precision": con["overall"]["precision"],
        "con_recall": con["overall"]["recall"],
        "con_f1": con["overall"]["f1"],
        "con_annotated": con["annotated"],
        "rag_annotated": rag["annotated"],
        "rag_attempted": rag["attempted"],
        "recall_at_1": rag["recall_at_k"]["@1"],
        "recall_at_3": rag["recall_at_k"]["@3"],
        "recall_at_5": rag["recall_at_k"]["@5"],
        "mrr": rag["mrr"],
        "questions_asked": questions["asked"],
        "question_quality": questions["quality"],
        "questions_per_resolved_gap": questions["questions_per_resolved_gap"],
        "human_intervention_reduction": effort["human_intervention_reduction"],
        "gt_gap_coverage": completeness["gt_gap_coverage"],
        "quality_score_initial": completeness["quality_score_initial"],
        "quality_score_final": completeness["quality_score_final"],
        "quality_score_delta": completeness["quality_score_delta"],
    }


def render_scores_csv(rows: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=SCORE_COLUMNS, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _prf_table(title: str, by_class: dict, note: str = "") -> list[str]:
    """Per-class P/R/F1, with detection recall beside it.

    The two differ when the system finds a gap but labels it something else:
    strict P/R/F1 charges that as a miss *and* a false positive, while
    "found" only asks whether the author was told about the gap at all.
    """
    lines = [
        f"### {title}",
        "",
        "| classe | P | R | F1 | found | TP | FP | FN | annotated |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for name, slot in by_class.items():
        # prf() scores an empty denominator as 1.0 — right for a whole case that
        # correctly reports nothing, misleading in a per-class row where it would
        # print "P 1.00" beside "TP 0 / FP 0". Not applicable is not a good score.
        precision = _fmt(slot["precision"] if slot["tp"] + slot["fp"] else None)
        recall = _fmt(slot["recall"] if slot["tp"] + slot["fn"] else None)
        f1 = _fmt(slot["f1"] if slot["tp"] + slot["fp"] + slot["fn"] else None)
        lines.append(
            f"| `{name}` | {precision} | {recall} | {f1} | "
            f"{_pct(slot['detection_recall'])} | {slot['tp']} | "
            f"{slot['fp']} | {slot['fn']} | {slot['annotated']} |"
        )
    if note:
        lines += ["", note]
    return [*lines, ""]


def render_report(manifest: dict, totals: dict, case_scores: list[dict],
                  skipped: list[str]) -> str:
    if not totals:
        # An interrupted run scored before any case finished. Say so in the file
        # rather than leaving the previous report standing or writing nothing.
        return "\n".join(
            [
                "# Benchmark scores",
                "",
                f"- run: `{manifest.get('run_id')}`",
                "",
                "No case in this run has a `predictions.json`, so there is nothing to score.",
                "",
                f"Not scored: {', '.join(skipped) if skipped else '(no case selected)'}",
                "",
            ]
        )

    gaps, con = totals["gaps"], totals["contradictions"]
    rag, questions = totals["retrieval"], totals["questions"]
    effort, completeness = totals["effort"], totals["completeness"]

    lines = [
        "# Benchmark scores",
        "",
        f"- run: `{manifest.get('run_id')}` ({totals['cases']} case(s) scored)",
        f"- dataset: `{manifest.get('dataset_version')}` · scorer `{scoring.SCORER_VERSION}`",
        f"- simulator: `{manifest.get('simulator_mode')}` (seed {manifest.get('seed')})",
        f"- model: `{(manifest.get('llm') or {}).get('provider')}/"
        f"{(manifest.get('llm') or {}).get('model')}`",
        f"- git: `{manifest.get('git_commit')}`"
        + (" (dirty)" if manifest.get("git_dirty") else ""),
    ]
    if skipped:
        lines.append(f"- **not scored** (no predictions.json): {', '.join(skipped)}")

    lines += [
        "",
        "## Headline",
        "",
        "| metric | value |",
        "|---|--:|",
        f"| Gap F1 (micro) | **{_fmt(gaps['micro']['f1'])}** |",
        f"| Gap precision / recall | {_fmt(gaps['micro']['precision'])} / "
        f"{_fmt(gaps['micro']['recall'])} |",
        f"| **Blocking-gap recall** | **{_pct(gaps['blocking_recall'])}** "
        f"({gaps['blocking_found']}/{gaps['blocking_total']}) |",
        f"| Contradiction F1 (micro) | **{_fmt(con['micro']['f1'])}** |",
        f"| Critical contradiction recall | {_pct(con['critical_recall'])} "
        f"({con['critical_found']}/{con['critical_total']}) |",
        f"| Answer-level contradictions (not scored) | {con['answer_level']} |",
        f"| RAG Recall@1 / @3 / @5 | {_pct(rag['recall_at_k']['@1'])} / "
        f"{_pct(rag['recall_at_k']['@3'])} / {_pct(rag['recall_at_k']['@5'])} |",
        f"| RAG MRR | {_fmt(rag['mrr'])} |",
        f"| Question quality | {_fmt(questions['mean_score'])} / 2 |",
        f"| Questions per resolved gap | {_fmt(questions['questions_per_resolved_gap'])} |",
        f"| **Human intervention reduction** | **{_pct(effort['human_intervention_reduction'])}** |",
        f"| Ground-truth gap coverage | {_pct(completeness['gt_gap_coverage'])} |",
        f"| Section quality score (initial → final) | "
        f"{_fmt(completeness['quality_score_initial'], 1)} → "
        f"{_fmt(completeness['quality_score_final'], 1)} "
        f"({_fmt(completeness['quality_score_delta'], 1)}) |",
        "",
        "## Per-case",
        "",
        "| case | gap P/R/F1 | blocking R | contra F1 | R@3 | MRR | Q quality | HIR |",
        "|---|---|--:|--:|--:|--:|--:|--:|",
    ]
    for case in case_scores:
        g, c = case["gaps"], case["contradictions"]
        lines.append(
            f"| {case['case_id']} | {_fmt(g['overall']['precision'])}/"
            f"{_fmt(g['overall']['recall'])}/{_fmt(g['overall']['f1'])} | "
            f"{_pct(g['blocking_recall'])} | {_fmt(c['overall']['f1'])} | "
            f"{_pct(case['retrieval']['recall_at_k']['@3'])} | "
            f"{_fmt(case['retrieval']['mrr'])} | "
            f"{_fmt(case['questions']['mean_score'])} | "
            f"{_pct(case['effort']['human_intervention_reduction'])} |"
        )

    lines += [
        "",
        "## Gap detection",
        "",
        "`P`/`R`/`F1` are strict: a gap matched on content but given the wrong label",
        "counts as a miss for the annotated class and a false positive for the class the",
        "system chose. `found` ignores the label and asks only whether the gap was",
        "surfaced at all — the gap between the two columns is a labelling problem, not a",
        "detection one.",
        "",
    ]
    lines += _prf_table("By category", gaps["by_category"])
    lines += _prf_table("By severity", gaps["by_severity"])

    lines += [
        "### Severity confusion (matched gaps only)",
        "",
        "| actual \\ predicted | blocking | important | nice_to_have |",
        "|---|--:|--:|--:|",
    ]
    for actual in scoring.SEVERITIES:
        row = gaps["severity_confusion"][actual]
        lines.append(
            f"| **{actual}** | {row['blocking']} | {row['important']} | {row['nice_to_have']} |"
        )

    lines += [
        "",
        "## Contradictions",
        "",
        f"- annotated (inside the initial CDC): **{con['annotated']}** · "
        f"reported by the run: {con['predicted']}",
        f"- micro P/R/F1: {_fmt(con['micro']['precision'])} / {_fmt(con['micro']['recall'])} / "
        f"**{_fmt(con['micro']['f1'])}** · critical recall {_pct(con['critical_recall'])}",
        f"- of which **{con['answer_level']}** were found between the run's own answers",
        "  rather than inside the CDC. The benchmark annotates no such contradictions, so",
        "  they can only raise recall — they are never counted as false positives.",
    ]

    judgment = rag["sufficiency_judgment"]
    lines += [
        "",
        "## Retrieval (RAG)",
        "",
        f"- annotated RAG-resolvable gaps: **{rag['annotated']}** — "
        f"detected {rag['detected']}, retrieval attempted {rag['attempted']}",
        f"- Recall@1/@3/@5 over attempted: {_pct(rag['recall_at_k']['@1'])} / "
        f"{_pct(rag['recall_at_k']['@3'])} / {_pct(rag['recall_at_k']['@5'])} · MRR {_fmt(rag['mrr'])}",
        f"- over every annotated gap (an undetected gap counts as a miss): "
        f"{_pct(rag['recall_at_k_overall']['@1'])} / {_pct(rag['recall_at_k_overall']['@3'])} / "
        f"{_pct(rag['recall_at_k_overall']['@5'])} · MRR {_fmt(rag['mrr_overall'])}",
        "",
        "Retrieval problem vs reasoning problem (roadmap §10):",
        "",
        f"- evidence retrieved **and** accepted: {judgment.get('correct_accept', 0)}",
        f"- evidence retrieved but the grader rejected it — a reasoning problem: "
        f"**{judgment.get('wrong_reject', 0)}**",
        f"- evidence never retrieved, correctly rejected: {judgment.get('correct_reject', 0)}",
        f"- answered from another source than the annotated one: "
        f"{judgment.get('accepted_other_source', 0)}",
        f"- judgment accuracy: {_pct(judgment.get('accuracy'))}",
        "",
        "## Question quality",
        "",
        f"{questions['asked']} question(s), each dimension scored 0 / 1 / 2.",
        "",
        "| dimension | mean |",
        "|---|--:|",
    ]
    for dim, value in questions["by_dimension"].items():
        lines.append(f"| `{dim}` | {_fmt(value)} |")
    lines.append(f"| **overall** | **{_fmt(questions['mean_score'])}** |")

    judged = [c for c in case_scores if c["questions"].get("llm_judge")]
    if judged:
        lines += ["", "### LLM judge (second opinion)", ""]
        for case in judged:
            judge = case["questions"]["llm_judge"]
            lines.append(
                f"- {case['case_id']}: {_fmt(judge['mean_score'])} / 2 over "
                f"{judge['judged']} question(s)"
                + (f", {judge['failed']} failed" if judge["failed"] else "")
            )

    lines += [
        "",
        "## Human effort (roadmap §26)",
        "",
        f"- resolved by RAG / human / assumption: **{effort['resolved_by_rag']} / "
        f"{effort['resolved_by_human']} / {effort['assumed']}**",
        f"- human intervention reduction: **{_pct(effort['human_intervention_reduction'])}**",
        f"- per case: {_fmt(effort['questions_per_case'], 1)} question(s), "
        f"{_fmt(effort['turns_per_case'], 1)} turn(s), {_fmt(effort['wall_s_per_case'], 1)}s",
        "",
        "## Misses and unmatched predictions",
        "",
        "Precision is strict: a detected gap matching no annotation counts as a false",
        "positive even when it is real. Read the unmatched list before believing a low",
        "precision figure — and if these are genuinely valid gaps, the precision above is",
        "a lower bound, not a verdict.",
        "",
    ]
    for case in case_scores:
        missed = case["gaps"]["missed"]
        unmatched = case["gaps"]["unmatched_predictions"]
        lines += [
            f"### {case['case_id']}",
            "",
            f"**Missed ({len(missed)})**",
            "",
        ]
        lines += [f"- `{m['id']}` [{m['severity']}/{m['category']}] {m['description']}"
                  for m in missed] or ["- (aucune)"]
        lines += ["", f"**Unmatched predictions ({len(unmatched)})**", ""]
        lines += [f"- [{u['severity']}/{u['category']}] {u['description']}"
                  for u in unmatched] or ["- (aucune)"]
        for miss in case["contradictions"]["missed"]:
            lines.append(f"- contradiction manquée `{miss['id']}` : "
                         f"« {miss['statement_a']} » vs « {miss['statement_b']} »")
        lines.append("")

    lines += [
        "---",
        "",
        "Gap and contradiction matching is by content (annotation keywords against the",
        f"detected gap), threshold {scoring.MATCH_THRESHOLD}. `quality_score_*` and",
        "`gt_gap_coverage` are the deterministic proxy for roadmap §12: the benchmark",
        "carries no expert-scored reference document, so this measures defect weight",
        "removed, not a human's judgment of the final CDC.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", help="run id under --out (default: the most recent one)")
    p.add_argument("--cases", help="comma-separated case ids (default: every case in the run)")
    p.add_argument("--out", type=Path, help=f"results root (default: {RESULTS_DIR})")
    p.add_argument(
        "--threshold",
        type=float,
        default=scoring.MATCH_THRESHOLD,
        help="keyword share needed to match a prediction to an annotation "
             f"(default: {scoring.MATCH_THRESHOLD})",
    )
    p.add_argument(
        "--judge",
        choices=["none", "llm"],
        default="none",
        help="add an LLM rubric judge for question quality, beside the deterministic scores "
             "(costs one call per asked question)",
    )
    return p


def progress(line: str) -> None:
    """Scoring is fast, but `--judge llm` spends a call per question — and even
    offline, seeing each case's numbers land beats waiting for the totals."""
    print(line, file=sys.stderr, flush=True)


def _case_line(case_score: dict) -> str:
    gaps = case_score["gaps"]
    rag = case_score["retrieval"]
    return (
        f"      gaps {gaps['overall']['tp']}/{gaps['annotated']} found "
        f"(P {gaps['overall']['precision']:.2f} R {gaps['overall']['recall']:.2f} "
        f"F1 {gaps['overall']['f1']:.2f}) · blocking {_pct(gaps['blocking_recall'])} · "
        f"contra F1 {case_score['contradictions']['overall']['f1']:.2f} · "
        f"RAG {rag['attempted']}/{rag['annotated']} tried R@3 "
        f"{_pct(rag['recall_at_k']['@3'])} · Q {_fmt(case_score['questions']['mean_score'])}/2"
    )


def score_run(results_root: Path, *, case_ids: list[str] | None = None,
              threshold: float = scoring.MATCH_THRESHOLD, judge: bool = False) -> dict:
    """Score one run directory and write its three artifacts. Returns scores.json."""
    manifest = _read_json(results_root / "manifest.json") or {}
    selected = case_ids or list(manifest.get("case_ids") or [])
    cases = load_benchmark(case_ids=selected or None)

    pairs, skipped = collect(results_root, cases)
    for case_id in skipped:
        progress(f"  - {case_id} ... pas de predictions.json, ignoré")

    case_scores = []
    for index, pair in enumerate(pairs, start=1):
        case_id = pair["case"].case_id
        progress(f"  - [{index}/{len(pairs)}] {case_id} ...")
        case_scores.append(
            scoring.score_case(pair["record"], pair["case"].ground_truth, threshold, judge=judge)
        )
        progress(_case_line(case_scores[-1]))

    totals = scoring.aggregate(case_scores)

    payload = {
        "scorer_version": scoring.SCORER_VERSION,
        "match_threshold": threshold,
        "judge": "llm" if judge else "none",
        # Copied from the run so two scores.json files can be compared without
        # going back for their manifests (roadmap Phase 6's regression reports).
        "run_id": manifest.get("run_id", results_root.name),
        "dataset_version": manifest.get("dataset_version"),
        "git_commit": manifest.get("git_commit"),
        "simulator_mode": manifest.get("simulator_mode"),
        "seed": manifest.get("seed"),
        "llm": manifest.get("llm"),
        "prompt_versions": manifest.get("prompt_versions"),
        "scored_cases": [c["case_id"] for c in case_scores],
        "skipped_cases": skipped,
        "totals": totals,
        "cases": case_scores,
    }

    write_atomic(results_root / "scores.json", _dumps(payload))
    write_atomic(results_root / "scores.csv", render_scores_csv([case_row(c) for c in case_scores]))
    write_atomic(
        results_root / "scores.md", render_report(manifest, totals, case_scores, skipped)
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir = args.out or RESULTS_DIR

    if args.run:
        results_root = results_dir / args.run
    else:
        results_root = latest_run(results_dir)
        if results_root is None:
            print(f"No run to score under {results_dir}.", file=sys.stderr)
            return 1
    if not (results_root / "manifest.json").exists():
        print(f"No run at {results_root}.", file=sys.stderr)
        return 1

    manifest = _read_json(results_root / "manifest.json") or {}
    llm = manifest.get("llm") or {}
    print(
        f"scoring {results_root.name}: dataset {manifest.get('dataset_version')}, "
        f"simulator {manifest.get('simulator_mode')} (seed {manifest.get('seed')}), "
        f"model {llm.get('provider')}/{llm.get('model')}, "
        f"scorer {scoring.SCORER_VERSION} @ threshold {args.threshold}"
        + (", + LLM judge" if args.judge == "llm" else ""),
        # The per-case trace goes to stderr, which is unbuffered; without this the
        # banner would surface after it whenever the two are piped together.
        flush=True,
    )

    case_ids = [c.strip() for c in args.cases.split(",")] if args.cases else None
    payload = score_run(
        results_root,
        case_ids=case_ids,
        threshold=args.threshold,
        judge=args.judge == "llm",
    )

    if not payload["scored_cases"]:
        print(f"Nothing to score in {results_root} — no case has a predictions.json.",
              file=sys.stderr)
        return 1

    totals = payload["totals"]
    print(f"scored {len(payload['scored_cases'])} case(s) in {results_root.name}")
    print(
        f"  gap F1 {totals['gaps']['micro']['f1']:.2f} "
        f"(P {totals['gaps']['micro']['precision']:.2f} / "
        f"R {totals['gaps']['micro']['recall']:.2f}) · "
        f"blocking recall {_pct(totals['gaps']['blocking_recall'])}"
    )
    print(
        f"  contradiction F1 {totals['contradictions']['micro']['f1']:.2f} · "
        f"RAG R@3 {_pct(totals['retrieval']['recall_at_k']['@3'])} · "
        f"MRR {_fmt(totals['retrieval']['mrr'])}"
    )
    print(
        f"  question quality {_fmt(totals['questions']['mean_score'])}/2 · "
        f"human intervention reduction "
        f"{_pct(totals['effort']['human_intervention_reduction'])}"
    )
    if payload["skipped_cases"]:
        print(f"  not scored: {', '.join(payload['skipped_cases'])}", file=sys.stderr)
    print(f"\nwrote {results_root / 'scores.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
