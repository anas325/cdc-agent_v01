"""Put two scored runs side by side — usually the baseline against the graph.

    uv run python evals/compare_runs.py base_20260914_101500 bench_20260914_120000

Reads each run's `scores.json` (so both must have been through
`run_scoring.py`) and writes `comparison.md` beside the second one, plus the
same table on stdout. Nothing is recomputed here: a delta is only ever the
difference between two numbers the scorer already produced, which keeps the
comparison honest when the scorer changes — rescore both runs and it moves for
both at once.

It refuses a pair that is not comparable. Two runs scored against different
dataset versions, answered by different stakeholder modes, driven by different
models, or scored by different scorer versions do not measure the same thing,
and a delta between them would be a number with no meaning. `--force` prints it
anyway, with the mismatches listed above the table.

The direction convention: the *first* run is the reference (the baseline), the
second is the candidate (the system under test). A positive delta therefore
means the candidate is ahead — except on the handful of metrics where less is
better, which are marked and inverted in the verdict line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from evals.run_benchmark import RESULTS_DIR, _read_json, write_atomic  # noqa: E402

# (label, path into `totals`, "ratio" | "score2" | "count" | "seconds", lower_is_better)
#
# Only headline numbers. The per-category and per-severity breakdowns stay in
# each run's own scores.md — a comparison that prints everything gets read like
# a log rather than like an answer.
METRICS: list[tuple[str, tuple[str, ...], str, bool]] = [
    ("Gap F1 (micro)", ("gaps", "micro", "f1"), "ratio", False),
    ("Gap precision", ("gaps", "micro", "precision"), "ratio", False),
    ("Gap recall", ("gaps", "micro", "recall"), "ratio", False),
    ("Blocking-gap recall", ("gaps", "blocking_recall"), "ratio", False),
    ("Gaps predicted", ("gaps", "predicted"), "count", False),
    ("Contradiction F1", ("contradictions", "micro", "f1"), "ratio", False),
    ("Contradiction recall", ("contradictions", "micro", "recall"), "ratio", False),
    ("RAG Recall@3", ("retrieval", "recall_at_k", "@3"), "ratio", False),
    ("RAG MRR", ("retrieval", "mrr"), "ratio", False),
    ("Retrieval judgment accuracy", ("retrieval", "sufficiency_judgment", "accuracy"), "ratio", False),
    ("Question quality", ("questions", "mean_score"), "score2", False),
    ("Questions asked", ("questions", "asked"), "count", True),
    ("Questions per resolved gap", ("questions", "questions_per_resolved_gap"), "score2", True),
    ("Human intervention reduction", ("effort", "human_intervention_reduction"), "ratio", False),
    ("Ground-truth gap coverage", ("completeness", "gt_gap_coverage"), "ratio", False),
    ("Quality score delta", ("completeness", "quality_score_delta"), "score2", False),
    ("Wall-clock per case", ("effort", "wall_s_per_case"), "seconds", True),
]

# The subset the verdict line counts. Coverage and question quality are what the
# whole loop exists to buy; cost is what it spends. Keep this short — a verdict
# averaged over seventeen metrics says nothing.
HEADLINE = ("Gap F1 (micro)", "Blocking-gap recall", "Ground-truth gap coverage", "Question quality")

# Fields that must agree for a delta to mean anything.
COMPARABLE = [
    ("dataset_version", "dataset version"),
    ("simulator_mode", "simulator mode"),
    ("scorer_version", "scorer version"),
    ("match_threshold", "match threshold"),
]


def dig(payload: dict, path: tuple[str, ...]):
    node = payload
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def fmt(value, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "ratio":
        return f"{value * 100:.1f}%"
    if kind == "count":
        return f"{value:.0f}" if isinstance(value, float) else str(value)
    if kind == "seconds":
        return f"{value:.0f}s"
    return f"{value:.2f}"


def fmt_delta(before, after, kind: str) -> str:
    """The delta in the metric's own units, signed, or a dash if either side is
    missing — a metric nothing measured must not read as "no change"."""
    if before is None or after is None:
        return "—"
    delta = after - before
    if kind == "ratio":
        return f"{delta * 100:+.1f} pts"
    if kind == "count":
        return f"{delta:+.0f}"
    if kind == "seconds":
        return f"{delta:+.0f}s"
    return f"{delta:+.2f}"


def incomparable(a: dict, b: dict) -> list[str]:
    """Which of COMPARABLE the two runs disagree on, in words."""
    problems = []
    for key, label in COMPARABLE:
        left, right = a.get(key), b.get(key)
        if left != right:
            problems.append(f"{label}: {left!r} vs {right!r}")
    # The biggest confound of all: "is the architecture worth it?" is not a
    # question two different models can answer between them.
    left, right = (a.get("llm") or {}).get("model"), (b.get("llm") or {}).get("model")
    if left != right:
        problems.append(f"model: {left!r} vs {right!r}")
    if a.get("run_id") == b.get("run_id"):
        problems.append("both sides are the same run")
    return problems


def rows(reference: dict, candidate: dict) -> list[dict]:
    out = []
    for label, path, kind, lower_better in METRICS:
        before, after = dig(reference["totals"], path), dig(candidate["totals"], path)
        out.append(
            {
                "label": label,
                "kind": kind,
                "lower_is_better": lower_better,
                "before": before,
                "after": after,
                "delta": (after - before) if (before is not None and after is not None) else None,
            }
        )
    return out


def verdict(table: list[dict]) -> str:
    """One line, over HEADLINE only, counting wins rather than averaging.

    Averaging percentages and 0–2 scores together would produce a number that
    looks precise and means nothing, so this counts instead.
    """
    headline = [r for r in table if r["label"] in HEADLINE and r["delta"] is not None]
    if not headline:
        return "No headline metric was measured on both runs."
    better = sum(
        1 for r in headline if (r["delta"] < 0 if r["lower_is_better"] else r["delta"] > 0)
    )
    worse = sum(
        1 for r in headline if (r["delta"] > 0 if r["lower_is_better"] else r["delta"] < 0)
    )
    return (
        f"Candidate ahead on {better}/{len(headline)} headline metric(s), "
        f"behind on {worse}, level on {len(headline) - better - worse}."
    )


def render(reference: dict, candidate: dict, problems: list[str]) -> str:
    table = rows(reference, candidate)

    def name(payload: dict) -> str:
        return f"{payload.get('run_id')} ({payload.get('system', 'graph')})"

    lines = [
        "# Run comparison",
        "",
        f"- reference: `{name(reference)}` — {len(reference.get('scored_cases') or [])} case(s)",
        f"- candidate: `{name(candidate)}` — {len(candidate.get('scored_cases') or [])} case(s)",
        f"- dataset `{candidate.get('dataset_version')}`, simulator "
        f"`{candidate.get('simulator_mode')}`, scorer `{candidate.get('scorer_version')}`",
        f"- model: `{(reference.get('llm') or {}).get('model')}` vs "
        f"`{(candidate.get('llm') or {}).get('model')}`",
        "",
    ]
    if problems:
        lines += [
            "> **Not comparable** — compared anyway with `--force`:",
            *(f"> - {p}" for p in problems),
            "",
        ]
    lines += [
        f"**{verdict(table)}**",
        "",
        "| metric | reference | candidate | delta |",
        "|---|---:|---:|---:|",
    ]
    for row in table:
        label = row["label"] + (" ↓" if row["lower_is_better"] else "")
        lines.append(
            f"| {label} | {fmt(row['before'], row['kind'])} | {fmt(row['after'], row['kind'])} | "
            f"{fmt_delta(row['before'], row['after'], row['kind'])} |"
        )
    lines += [
        "",
        "`↓` marks a metric where lower is better.",
        "",
        "Per-case numbers and the category/severity breakdowns stay in each run's own",
        "`scores.md`; nothing here is recomputed from predictions.",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("reference", help="run id of the reference (usually the baseline)")
    p.add_argument("candidate", help="run id of the system under test")
    p.add_argument("--out", type=Path, help=f"results root (default: {RESULTS_DIR})")
    p.add_argument("--write", type=Path,
                   help="where to write comparison.md (default: beside the candidate run)")
    p.add_argument("--force", action="store_true",
                   help="compare runs that disagree on dataset, simulator or scorer version")
    return p


def load_scores(root: Path, run_id: str) -> dict | None:
    payload = _read_json(root / run_id / "scores.json")
    return payload if isinstance(payload, dict) and "totals" in payload else None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.out or RESULTS_DIR

    reference = load_scores(root, args.reference)
    candidate = load_scores(root, args.candidate)
    for run_id, payload in ((args.reference, reference), (args.candidate, candidate)):
        if payload is None:
            print(
                f"No scores.json in {root / run_id} — score it first:\n"
                f"    uv run python evals/run_scoring.py --run {run_id}",
                file=sys.stderr,
            )
            return 1

    problems = incomparable(reference, candidate)
    if problems and not args.force:
        print("Refusing to compare these runs:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("Pass --force to compare them anyway.", file=sys.stderr)
        return 1

    report = render(reference, candidate, problems if args.force else [])
    path = args.write or (root / args.candidate / "comparison.md")
    write_atomic(path, report)
    print(report)
    print(f"-> {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
