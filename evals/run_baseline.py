"""Run the simple baseline over the benchmark, for comparison with the graph.

`run_benchmark.py` measures the swarm. This runs `evals/baseline.py` — one LLM
call, naive retrieval, one question batch — over the *same* cases, with the same
synthetic stakeholder, and writes the same artifacts to the same layout. The
point is the pair: a number from `run_scoring.py` only means something next to
what a prompt-only solution scores on the same ten CDCs.

    uv run python evals/run_baseline.py                          # all cases, oracle mode
    uv run python evals/run_baseline.py --cases cdc_003_ecommerce --cache
    uv run python evals/run_baseline.py --no-rag                 # skip retrieval entirely
    uv run python evals/run_baseline.py --rag-closes-gaps        # the ungraded-RAG variant
    uv run python evals/run_baseline.py --score                  # then score it
    uv run python evals/compare_runs.py <baseline_run> <bench_run>

Because the output directory is shaped exactly like a benchmark run — a
`manifest.json`, one `predictions.json` per case — the scorer needs no changes:

    uv run python evals/run_scoring.py --run base_20260914_101500

Run ids are prefixed `base_` rather than `bench_`, and the manifest carries
`system: "baseline"`, so the two can share `evals/results/` without a comparison
ever silently pairing a run with itself.

It is much cheaper than the benchmark — one LLM call per case plus the
stakeholder's, against the graph's hundreds — so there is no checkpointing and
no mid-case resume here. `--resume` only skips cases already on disk.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from evals import baseline, harness  # noqa: E402
from evals.dataset import BenchmarkCase, dataset_version, load_benchmark  # noqa: E402
from evals.run_benchmark import (  # noqa: E402
    RESULTS_DIR,
    CaseRecorder,
    _dumps,
    _read_json,
    flush_run_outputs,
    latest_run,
    load_case_row,
    summarize,
    telemetry_snapshot,
    write_atomic,
)
from evals.simulator import MODES, make_simulator  # noqa: E402
from src import telemetry  # noqa: E402
from src.config import load_sections, load_settings  # noqa: E402
from src.ids import stable_id  # noqa: E402
from src.state import ContextItem  # noqa: E402

REPORT_TITLE = "Baseline run report"


def initial_context(cdc_text: str) -> ContextItem:
    """The CDC as the baseline sees it: one block, no section splitting.

    The graph's `ingest` slices the document per section so each prompt carries
    only its own; not doing that is part of what makes this a baseline. The item
    is deliberately left untagged, which is also how `scoring._InitialCdc` reads
    a CDC whose chunks mapped to no section — the question-context check then
    judges against the whole document, a slightly *generous* reading that errs
    against the system under comparison rather than for it.
    """
    return ContextItem(
        id=stable_id("ctx", "initial_cdc", cdc_text[:200]),
        content=cdc_text,
        source="initial_cdc",
        section_ids=[],
        turn_added=0,
        created_by="system",
    )


def run_case(
    case: BenchmarkCase, *, simulator, use_rag: bool = True, rag_closes_gaps: bool = False
) -> dict:
    """One case end to end. Returns the record written to predictions.json.

    Shaped exactly like `run_benchmark.run_case`'s return value, because the
    scorer, `summarize` and the report all read that shape and must not be able
    to tell which system produced a given run directory.
    """
    from src.rag import ingest_source_docs, retrieve

    thread_id = f"base-{case.case_id}"
    telemetry.reset(thread_id)

    sections = load_sections()
    started = time.perf_counter()
    status, error, stop_reason = "ok", None, "baseline_complete"

    gaps, questions, decisions = [], [], []
    context_items = [initial_context(case.initial_cdc_text)]
    transcript: list[dict] = []

    try:
        with telemetry.record_node("baseline_analyze"):
            gaps, questions, decisions = baseline.analyze(case.initial_cdc_text, sections)

        # A case with no reference documents has nothing to retrieve from, and
        # querying an empty index once per gap would fill the decision log with
        # rejections that mean nothing.
        if use_rag and case.source_doc_names:
            try:
                with telemetry.record_node("baseline_rag"):
                    ingest_source_docs()
                    rag_items, rag_decisions = baseline.fill_from_rag(
                        gaps, retrieve, close_gaps=rag_closes_gaps
                    )
            except Exception as exc:
                # An unusable index (a stale Chroma collection built with another
                # embedding function, most often) is an environment problem, not a
                # result. Degrade to no retrieval and say so, rather than record
                # the case as a failure of the baseline.
                print(f"    (RAG indisponible, ignoré : {exc})", file=sys.stderr)
                rag_items, rag_decisions = [], []
            context_items += rag_items
            decisions += rag_decisions

        # One batch, every open gap, no dedup and no budget — the naive behaviour
        # the orchestrator's question gate exists to avoid.
        open_gaps = {gap.id for gap in gaps if gap.status == "open"}
        pending = [
            {"gap_id": q.gap_id, "text": q.text} for q in questions if q.gap_id in open_gaps
        ]
        questions = [q for q in questions if q.gap_id in open_gaps]

        if pending:
            with telemetry.record_node("baseline_questions"):
                batch = simulator.answer(pending, {"gaps": gaps})
            transcript.append(
                {
                    "round": 0,
                    "turn": 1,
                    "questions": pending,
                    "answers": batch.as_records(),
                }
            )
            items, answer_decisions = baseline.integrate(gaps, batch.as_records())
            context_items += items
            decisions += answer_decisions
    except Exception as exc:  # one bad case must not abort the batch
        status, error, stop_reason = "error", "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ), "error"

    statuses = baseline.section_statuses(gaps, sections)

    return {
        "case_id": case.case_id,
        "title": case.title,
        "thread_id": thread_id,
        "system": "baseline",
        "status": status,
        "error": error,
        "finished": status == "ok",
        "done": status == "ok",
        "stop_reason": stop_reason,
        # The baseline has exactly one turn and one question round, by construction.
        "turns": 1,
        "rounds": len(transcript),
        "segments": 1,
        "wall_s": round(time.perf_counter() - started, 2),
        "gaps": [g.model_dump() for g in gaps],
        "context_items": [c.model_dump() for c in context_items],
        "asked_questions": [q.model_dump() for q in questions],
        "decision_log": [d.model_dump() for d in decisions],
        "section_statuses": {sid: ss.model_dump() for sid, ss in statuses.items()},
        "transcript": transcript,
        "telemetry": telemetry_snapshot(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cases", help="comma-separated case ids (default: the whole dataset)")
    p.add_argument("--mode", choices=MODES, default="oracle",
                   help="synthetic stakeholder mode — match the benchmark run you compare against")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the realistic simulator")
    p.add_argument("--provider", choices=["ollama", "anthropic"], help="override llm.provider")
    p.add_argument("--model", help="override llm.model")
    p.add_argument("--temperature", type=float, help="override llm.temperature")
    p.add_argument("--top-k", type=int, help="override rag.top_k")
    p.add_argument("--no-rag", action="store_true",
                   help="skip retrieval entirely (by default it runs and is measured, "
                        "but never closes a gap)")
    p.add_argument("--rag-closes-gaps", action="store_true",
                   help="also let the retriever's similarity score close a gap, ungraded — "
                        f"the naive variant, floor {baseline.RAG_SCORE_FLOOR}")
    p.add_argument("--cache", action="store_true", help="enable the LLM disk cache for this run")
    p.add_argument("--score", action="store_true",
                   help="score the run against ground truth once it finishes")
    p.add_argument("--out", type=Path, help=f"results root (default: {RESULTS_DIR})")

    naming = p.add_mutually_exclusive_group()
    naming.add_argument("--run-id", help="name this run (default: base_<UTC timestamp>)")
    naming.add_argument("--resume", nargs="?", const="@latest", metavar="RUN_ID",
                        help="fill in the cases a previous baseline run never wrote")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cache:
        import os

        os.environ["CDC_LLM_CACHE"] = "1"

    root = args.out or RESULTS_DIR
    root.mkdir(parents=True, exist_ok=True)

    if args.resume:
        previous = latest_run(root) if args.resume == "@latest" else root / args.resume
        if previous is None or not previous.is_dir():
            print(f"No baseline run to resume under {root}", file=sys.stderr)
            return 1
        results_root = previous
        # A resume that narrows with --cases must not shrink the run: the manifest
        # is what run_scoring.py reads to know which cases belong to it, and
        # rewriting it to the subset would drop the others from the score.
        prior_case_ids = list((_read_json(results_root / "manifest.json") or {}).get("case_ids") or [])
    else:
        prior_case_ids = []
        results_root = root / (args.run_id or f"base_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}")
        results_root.mkdir(parents=True, exist_ok=True)
    run_id = results_root.name

    selected = [c.strip() for c in args.cases.split(",")] if args.cases else (prior_case_ids or None)
    cases = load_benchmark(case_ids=selected)
    if not cases:
        print("No cases selected.")
        return 1
    run_case_ids = prior_case_ids + [
        c.case_id for c in cases if c.case_id not in prior_case_ids
    ]

    base = load_settings()
    effective = harness.case_settings(
        base,
        source_dir=base.rag.source_dir,
        persist_dir=base.rag.persist_dir,
        output_dir=base.quarto.output_dir,
        top_k=args.top_k,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
    )

    manifest = harness.run_manifest(
        run_id=run_id,
        dataset_version=dataset_version(),
        case_ids=run_case_ids,
        simulator_mode=args.mode,
        seed=args.seed,
        settings=effective,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    # What makes this run comparable — and what makes it *not* a benchmark run.
    # compare_runs.py refuses to pair two runs that disagree on the first field.
    manifest["system"] = "baseline"
    manifest["baseline"] = {
        "strategy": "oneshot",
        "rag": "off" if args.no_rag else ("closes_gaps" if args.rag_closes_gaps else "measured"),
        "rag_score_floor": baseline.RAG_SCORE_FLOOR if args.rag_closes_gaps else None,
        "prompt_id": baseline.PROMPT_ID,
    }
    write_atomic(results_root / "manifest.json", _dumps(manifest))

    print(
        f"baseline {run_id}: {len(cases)} case(s), mode={args.mode} (seed {args.seed}), "
        f"model {effective.llm.provider}/{effective.llm.model}, "
        f"RAG {'off' if args.no_rag else f'top_k {effective.rag.top_k}'}"
        + (" (ferme les lacunes)" if args.rag_closes_gaps and not args.no_rag else "")
        + (", cache LLM" if args.cache else "")
        + f" -> {results_root}",
        flush=True,
    )

    # Keyed by case id and emitted in the run's own order, so a resume narrowed
    # with --cases still carries the other cases' rows into summary.csv.
    rows_by_case: dict[str, dict] = {}
    if args.resume:
        for case_id in run_case_ids:
            done = load_case_row(results_root / case_id)
            if done is not None:
                rows_by_case[case_id] = done

    def flush() -> None:
        flush_run_outputs(
            results_root,
            manifest,
            [rows_by_case[cid] for cid in run_case_ids if cid in rows_by_case],
            REPORT_TITLE,
        )

    flush()

    for index, case in enumerate(cases, start=1):
        if case.case_id in rows_by_case:
            print(f"  - [{index}/{len(cases)}] {case.case_id} ... déjà terminé, ignoré")
            continue

        case_out = results_root / case.case_id
        settings = harness.case_settings(
            effective,
            source_dir=case.source_docs_dir,
            # The benchmark's per-case index, reused on purpose: the baseline and
            # the graph must retrieve from the same chunks for Recall@K to compare.
            persist_dir=ROOT / ".cache" / "bench" / case.case_id / "chroma",
            output_dir=case_out / "artifacts",
        )

        print(f"  - [{index}/{len(cases)}] {case.case_id} ... ", end="", flush=True)
        with CaseRecorder(case_out) as recorder:
            recorder.record_status("running", started_at=datetime.now(timezone.utc).isoformat())
            try:
                with harness.isolate(settings):
                    simulator = make_simulator(
                        args.mode,
                        ground_truth=case.ground_truth,
                        profile=case.stakeholder,
                        seed=args.seed,
                    )
                    record = run_case(
                        case,
                        simulator=simulator,
                        use_rag=not args.no_rag,
                        rag_closes_gaps=args.rag_closes_gaps,
                    )
            except KeyboardInterrupt:
                recorder.record_status("interrupted")
                print("interrompu")
                break
            # record_state reads a graph state, where the turn counter is "turn".
            recorder.record_state({**record, "turn": record["turns"]})
            recorder.finish(record)

        row = summarize(record)
        rows_by_case[case.case_id] = row
        print(
            f"{row['status']} — {row['gaps_total']} gap(s), {row['questions_asked']} question(s), "
            f"RAG/humain/hypothèse {row['resolved_by_rag']}/{row['resolved_by_user']}/"
            f"{row['assumed']}, {row['sections_complete']}/{row['sections_total']} section(s), "
            f"{row['wall_s']}s"
        )
        if record.get("error"):
            print(record["error"], file=sys.stderr)
        flush()

    print(f"\n-> {results_root}")
    if args.score:
        from evals.run_scoring import main as score_main

        return score_main(["--run", run_id, "--out", str(root)])
    print(f"Score it with:  uv run python evals/run_scoring.py --run {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
