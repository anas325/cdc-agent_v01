"""Deterministic batch runner over the CDC benchmark (roadmap Phase 3, §16).

Drives the *real* compiled graph end-to-end for every benchmark case, with a
synthetic stakeholder standing in for the human at each `interrupt()`. Unlike
run_evals.py — which calls two agents directly — this exercises the whole
machine: ingest, initial scan, orchestrator loop, RAG, question dedup, the
human-in-the-loop cycle, the critic, synthesis and final validation.

    uv run python evals/run_benchmark.py                                # all cases, oracle mode
    uv run python evals/run_benchmark.py --cases cdc_003_ecommerce
    uv run python evals/run_benchmark.py --mode realistic --seed 7
    uv run python evals/run_benchmark.py --provider anthropic --max-turns 8 --cache

It emits **predictions and descriptive statistics, not scores**: comparing them
against ground truth (precision/recall/F1, Recall@K, question quality) is
roadmap Phase 4, and reads the artifacts written here.

Each case runs in isolation — its own RAG corpus, its own Chroma index, its own
output directory — via evals/harness.py::isolate. Nothing lands in the repo-root
`output/` that the Streamlit app uses.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langgraph.types import Command  # noqa: E402

from evals import harness  # noqa: E402
from evals.dataset import BenchmarkCase, dataset_version, load_benchmark  # noqa: E402
from evals.simulator import MODES, AnswerBatch, make_simulator  # noqa: E402
from src import telemetry  # noqa: E402
from src.config import load_sections, load_settings  # noqa: E402
from src.graph import build_graph  # noqa: E402
from src.state import LoopSettings, SectionStatus  # noqa: E402

RESULTS_DIR = ROOT / "evals" / "results"

# Hard ceiling on interrupt/resume rounds per case. The graph's own max_turns is
# the real bound; this only stops a pathological loop from running forever.
MAX_ROUNDS = 60


# ---------------------------------------------------------------------------
# Driving the graph
# ---------------------------------------------------------------------------


def _drain(stream) -> list[dict]:
    """Consume a graph.stream() and collect any interrupts it emitted."""
    interrupts: list = []
    for update in stream:
        if "__interrupt__" in update:
            interrupts.extend(update["__interrupt__"])
    return interrupts


def pending_questions(graph, config) -> list[dict] | None:
    """The question batch the graph is currently blocked on, if any.

    Reads the checkpoint rather than trusting what the stream emitted: the
    Streamlit app uses the same fallback, and it also covers resuming a thread
    that was interrupted by a previous process.
    """
    snapshot = graph.get_state(config)
    for task in snapshot.tasks:
        for itr in getattr(task, "interrupts", ()):  # LangGraph Interrupt objects
            value = getattr(itr, "value", None)
            if isinstance(value, dict) and value.get("questions"):
                return list(value["questions"])
    return None


def run_case(case: BenchmarkCase, *, simulator, loop_settings: LoopSettings) -> dict:
    """One full graph run. Returns the record written to predictions.json."""
    thread_id = f"bench-{case.case_id}-{uuid.uuid4().hex[:8]}"
    telemetry.reset(thread_id)

    graph = build_graph()  # MemorySaver: no Postgres needed for a benchmark run
    config = {"configurable": {"thread_id": thread_id}}

    # Exactly what src/app.py::start_run sends — the graph builds everything else.
    input_state = {
        "initial_cdc_text": case.initial_cdc_text,
        "loop_settings": loop_settings,
        "section_statuses": {
            sec.id: SectionStatus(section_id=sec.id, status="empty")
            for sec in load_sections()
            if sec.required
        },
    }

    transcript: list[dict] = []
    started = time.perf_counter()
    status = "ok"
    error = None

    try:
        _drain(graph.stream(input_state, config, stream_mode="updates"))

        rounds = 0
        while (questions := pending_questions(graph, config)) is not None:
            if rounds >= MAX_ROUNDS:
                status = "round_limit"
                break
            values = graph.get_state(config).values
            batch: AnswerBatch = simulator.answer(questions, values)
            transcript.append(
                {
                    "round": rounds,
                    "turn": values.get("turn"),
                    "questions": questions,
                    "answers": batch.as_records(),
                }
            )
            _drain(graph.stream(Command(resume=batch.resume_payload), config, stream_mode="updates"))
            rounds += 1
    except Exception as exc:  # one bad case must not abort the batch
        status = "error"
        error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    wall_s = time.perf_counter() - started
    values = graph.get_state(config).values
    finished = graph.get_state(config).next == ()

    return {
        "case_id": case.case_id,
        "title": case.title,
        "thread_id": thread_id,
        "status": status,
        "error": error,
        "finished": finished,
        "done": bool(values.get("done")),
        "stop_reason": values.get("stop_reason"),
        "turns": values.get("turn", 0),
        "rounds": len(transcript),
        "wall_s": round(wall_s, 2),
        "gaps": [g.model_dump() for g in values.get("gaps", [])],
        "context_items": [c.model_dump() for c in values.get("context_items", [])],
        "asked_questions": [q.model_dump() for q in values.get("asked_questions", [])],
        "section_statuses": {
            sid: ss.model_dump() for sid, ss in (values.get("section_statuses") or {}).items()
        },
        "transcript": transcript,
        "telemetry": telemetry.summary(),
    }


# ---------------------------------------------------------------------------
# Descriptive statistics (no ground-truth comparison — that is Phase 4)
# ---------------------------------------------------------------------------

SUMMARY_COLUMNS = [
    "case_id",
    "status",
    "finished",
    "turns",
    "rounds",
    "stop_reason",
    "gaps_total",
    "gaps_blocking",
    "gaps_important",
    "gaps_nice_to_have",
    "resolved_by_rag",
    "resolved_by_user",
    "assumed",
    "deferred",
    "open",
    "questions_asked",
    "unknown_answers",
    "sections_complete",
    "sections_total",
    "wall_s",
    "llm_calls",
    "llm_s",
    "retries",
    "cache_hits",
]


def summarize(record: dict) -> dict:
    gaps = record["gaps"]
    severities = Counter(g["severity"] for g in gaps)
    statuses = Counter(g["status"] for g in gaps)
    categories = Counter(g["category"] for g in gaps)
    sources = Counter(c["source"] for c in record["context_items"])
    section_statuses = record["section_statuses"]
    tele = record.get("telemetry") or {}

    unknown = sum(
        1 for round_ in record["transcript"] for a in round_["answers"] if a["skip"]
    )

    row = {
        "case_id": record["case_id"],
        "status": record["status"],
        "finished": record["finished"],
        "turns": record["turns"],
        "rounds": record["rounds"],
        "stop_reason": record["stop_reason"] or "",
        "gaps_total": len(gaps),
        "gaps_blocking": severities["blocking"],
        "gaps_important": severities["important"],
        "gaps_nice_to_have": severities["nice_to_have"],
        # The RAG-vs-human-vs-assumption split is the raw material for roadmap
        # §26's "human intervention reduction"; the ratio itself is Phase 4's.
        "resolved_by_rag": sources["rag"],
        "resolved_by_user": sources["user_answer"],
        "assumed": sources["assumption"],
        "deferred": statuses["deferred"],
        "open": statuses["open"],
        "questions_asked": len(record["asked_questions"]),
        "unknown_answers": unknown,
        "sections_complete": sum(1 for s in section_statuses.values() if s["status"] == "complete"),
        "sections_total": len(section_statuses),
        "wall_s": record["wall_s"],
        "llm_calls": tele.get("llm_count", 0),
        "llm_s": round(tele.get("llm_s", 0.0), 2),
        "retries": tele.get("retry_count", 0),
        "cache_hits": tele.get("cache_hit_count", 0),
    }
    row["_categories"] = dict(categories)  # kept out of the CSV, used by report.md
    return row


def write_report(path: Path, manifest: dict, rows: list[dict]) -> None:
    lines = [
        "# Benchmark run report",
        "",
        f"- run_id: `{manifest['run_id']}`",
        f"- dataset: `{manifest['dataset_version']}` ({len(rows)} case(s))",
        f"- simulator: `{manifest['simulator_mode']}` (seed {manifest['seed']})",
        f"- model: `{manifest['llm']['provider']}/{manifest['llm']['model']}` "
        f"@ temperature {manifest['llm']['temperature']}",
        f"- git: `{manifest['git_commit']}`" + (" (dirty)" if manifest["git_dirty"] else ""),
        "",
        "## Per-case",
        "",
        "| case | status | turns | gaps (B/I/N) | RAG | human | assumed | questions | "
        "«je ne sais pas» | sections | wall |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['case_id']} | {r['status']} | {r['turns']} | "
            f"{r['gaps_blocking']}/{r['gaps_important']}/{r['gaps_nice_to_have']} | "
            f"{r['resolved_by_rag']} | {r['resolved_by_user']} | {r['assumed']} | "
            f"{r['questions_asked']} | {r['unknown_answers']} | "
            f"{r['sections_complete']}/{r['sections_total']} | {r['wall_s']}s |"
        )

    totals = Counter()
    for r in rows:
        for key in ("resolved_by_rag", "resolved_by_user", "assumed", "questions_asked", "gaps_total"):
            totals[key] += r[key]
    categories = Counter()
    for r in rows:
        categories.update(r["_categories"])

    lines += [
        "",
        "## Totals",
        "",
        f"- gaps detected: **{totals['gaps_total']}**",
        f"- resolved by RAG / human / assumption: "
        f"**{totals['resolved_by_rag']} / {totals['resolved_by_user']} / {totals['assumed']}**",
        f"- questions asked: **{totals['questions_asked']}**",
        "",
        "### Gaps by category",
        "",
    ]
    for cat, count in sorted(categories.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{cat}`: {count}")

    lines += [
        "",
        "---",
        "",
        "Descriptive only — no ground-truth comparison. Precision/recall/F1, blocking-gap",
        "recall, RAG Recall@K and question-quality scoring are roadmap Phase 4 and consume",
        "`predictions.json` + `ground_truth.json`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cases", help="comma-separated case ids (default: every case in the manifest)")
    p.add_argument("--mode", choices=MODES, default="oracle", help="synthetic stakeholder mode")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the realistic simulator")
    p.add_argument("--provider", choices=["ollama", "anthropic"], help="override llm.provider")
    p.add_argument("--model", help="override llm.model")
    p.add_argument("--temperature", type=float, help="override llm.temperature")
    p.add_argument("--top-k", type=int, help="override rag.top_k")
    p.add_argument("--max-turns", type=int, help="override loop.max_turns")
    p.add_argument("--max-questions-per-batch", type=int, help="override loop.max_questions_per_batch")
    p.add_argument("--cache", action="store_true", help="enable the LLM disk cache for this run")
    p.add_argument("--run-id", help="name this run (default: bench_<UTC timestamp>)")
    p.add_argument("--out", type=Path, help=f"results root (default: {RESULTS_DIR})")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    run_id = args.run_id or f"bench_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    results_root = (args.out or RESULTS_DIR) / run_id
    results_root.mkdir(parents=True, exist_ok=True)

    if args.cache:
        # Read at call time by src/llm_cache.py, so setting it here is enough.
        os.environ["CDC_LLM_CACHE"] = "1"
        os.environ.setdefault("CDC_LLM_CACHE_DIR", str(ROOT / ".cache" / "llm"))

    case_ids = [c.strip() for c in args.cases.split(",")] if args.cases else None
    cases = load_benchmark(case_ids=case_ids)
    if not cases:
        print("No cases selected.")
        return 1

    base = load_settings()
    loop_update = {}
    if args.max_turns is not None:
        loop_update["max_turns"] = args.max_turns
    if args.max_questions_per_batch is not None:
        loop_update["max_questions_per_batch"] = args.max_questions_per_batch
    loop_settings = base.loop.model_copy(update=loop_update) if loop_update else base.loop

    started_at = datetime.now(timezone.utc).isoformat()
    print(f"run {run_id}: {len(cases)} case(s), mode={args.mode} -> {results_root}")

    rows: list[dict] = []
    effective_settings = base
    for case in cases:
        case_out = results_root / case.case_id
        case_out.mkdir(parents=True, exist_ok=True)

        settings = harness.case_settings(
            base,
            source_dir=case.source_docs_dir,
            # Per-case Chroma index: otherwise every case retrieves from every
            # other case's reference documents.
            persist_dir=ROOT / ".cache" / "bench" / case.case_id / "chroma",
            output_dir=case_out / "artifacts",
            top_k=args.top_k,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
        )
        settings = settings.model_copy(update={"loop": loop_settings})
        effective_settings = settings

        print(f"  - {case.case_id} ... ", end="", flush=True)
        # isolate() drops the memoized chat models, so a provider/model override
        # carried in `settings` is enough — no need to rebind the agent modules
        # the way run_evals.py has to.
        with harness.isolate(settings):
            simulator = make_simulator(
                args.mode,
                ground_truth=case.ground_truth,
                profile=case.stakeholder,
                seed=args.seed,
            )
            record = run_case(case, simulator=simulator, loop_settings=loop_settings)

        (case_out / "predictions.json").write_text(
            json.dumps({k: v for k, v in record.items() if k != "telemetry"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (case_out / "telemetry.json").write_text(
            json.dumps(record["telemetry"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with open(case_out / "transcript.jsonl", "w", encoding="utf-8") as f:
            for entry in record["transcript"]:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        row = summarize(record)
        rows.append(row)
        print(
            f"{row['status']} — {row['turns']} turn(s), {row['gaps_total']} gap(s), "
            f"{row['questions_asked']} question(s), {row['wall_s']}s"
        )
        if record["error"]:
            print(record["error"], file=sys.stderr)

    manifest = harness.run_manifest(
        run_id=run_id,
        dataset_version=dataset_version(),
        case_ids=[c.case_id for c in cases],
        simulator_mode=args.mode,
        seed=args.seed,
        settings=effective_settings,
        started_at=started_at,
    )
    (results_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with open(results_root / "summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    write_report(results_root / "report.md", manifest, rows)

    failed = [r["case_id"] for r in rows if r["status"] == "error"]
    print(f"\nwrote {results_root}")
    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
