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
    uv run python evals/run_benchmark.py --resume                       # continue the last run

It emits **predictions and descriptive statistics, not scores**: comparing them
against ground truth (precision/recall/F1, Recall@K, question quality) is
roadmap Phase 4, and reads the artifacts written here.

Each case runs in isolation — its own RAG corpus, its own Chroma index, its own
output directory — via evals/harness.py::isolate. Nothing lands in the repo-root
`output/` that the Streamlit app uses.

**Interrupting is cheap.** A run is ten cases of several minutes each, so nothing
is held in memory until the end: the graph checkpoint is flushed to disk after
*every node*, the transcript after every question round, and summary.csv/report.md
after every case. `--resume` then picks the run back up — a finished case is
skipped, and an unfinished one continues from its last completed graph turn
rather than starting over. See evals/checkpoints.py for how the checkpointer is
made durable without adding a dependency.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langgraph.types import Command  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from evals import checkpoints, harness  # noqa: E402
from evals.dataset import BenchmarkCase, dataset_version, load_benchmark  # noqa: E402
from evals.simulator import MODES, AnswerBatch, make_simulator  # noqa: E402
from src import telemetry  # noqa: E402
from src.config import load_sections, load_settings  # noqa: E402
from src.graph import build_graph  # noqa: E402
from src.state import LoopSettings, SectionStatus  # noqa: E402

RESULTS_DIR = ROOT / "evals" / "results"

# Hard ceiling on interrupt/resume rounds per case. The graph's own max_turns is
# the real bound; this only stops a pathological loop from running forever.
MAX_ROUNDS = 15

# Manifest fields that must agree before a run may be resumed: mixing two of them
# into one summary.csv would make the numbers mean nothing. --force overrides.
RESUME_MUST_MATCH = [
    ("dataset_version",),
    ("simulator_mode",),
    ("seed",),
    ("llm", "provider"),
    ("llm", "model"),
]


# ---------------------------------------------------------------------------
# Writing to disk
# ---------------------------------------------------------------------------


def _json_default(obj):
    """CDCState carries pydantic models; dump them rather than fail on a snapshot."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, (datetime, Path)):
        return str(obj)
    return repr(obj)


def _dumps(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)


def write_atomic(path: Path, text: str) -> None:
    """Write via a sibling temp file + os.replace.

    Every artifact goes through this. Several are rewritten after each graph turn,
    and a crash caught mid-write must leave the previous good version in place
    rather than a truncated one — predictions.json in particular is the marker
    that says a case is finished.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # a torn last line: everything before it is still good
    except OSError:
        return []
    return rows


class CaseRecorder:
    """Everything one case writes to disk, written as the case happens.

    Two jobs. `checkpoint/` is what makes an interrupted case *resumable* — it is
    the LangGraph checkpointer's own state, flushed after every graph turn. The
    rest makes a partial case *readable*, and lets a resumed process pick the
    transcript and the simulator back up where the previous one left off.

    `predictions.json` is written last and is the one true completion marker;
    directory existence is not, since the directory is created before the case
    starts.
    """

    ARTIFACTS = (
        "steps.jsonl", "transcript.jsonl", "state.json", "segments.json",
        "simulator.json", "predictions.json", "telemetry.json", "status.json",
    )

    def __init__(self, case_out: Path, *, resume: bool = False):
        self.dir = case_out
        self.dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.dir / "checkpoint"
        if not resume:
            self.reset()
        self._steps = None
        self._transcript = None
        self._step_no = len(_read_jsonl(self.dir / "steps.jsonl"))

    # -- lifecycle -----------------------------------------------------------

    def reset(self) -> None:
        """Drop a previous attempt's state so this one really starts over.

        Without it, reusing a --run-id *without* --resume would quietly continue
        from the old checkpoint instead of re-running the case.
        """
        checkpoints.discard(self.checkpoint_dir)
        for name in self.ARTIFACTS:
            (self.dir / name).unlink(missing_ok=True)

    def __enter__(self) -> "CaseRecorder":
        self._steps = open(self.dir / "steps.jsonl", "a", encoding="utf-8")
        self._transcript = open(self.dir / "transcript.jsonl", "a", encoding="utf-8")
        return self

    def __exit__(self, *exc_info) -> None:
        for handle in (self._steps, self._transcript):
            if handle is not None:
                handle.close()
        self._steps = self._transcript = None

    def _append(self, handle, payload: dict) -> None:
        handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    # -- reading back what a previous process left ---------------------------

    def resumable(self) -> bool:
        return checkpoints.has_state(self.checkpoint_dir)

    def load_transcript(self) -> list[dict]:
        return _read_jsonl(self.dir / "transcript.jsonl")

    def load_simulator_state(self) -> dict | None:
        return _read_json(self.dir / "simulator.json")

    def load_segments(self) -> list[dict]:
        return _read_json(self.dir / "segments.json") or []

    # -- writing -------------------------------------------------------------

    def record_step(self, *, node: str, values: dict, node_s: float | None,
                    sync_s: float, keys: list[str]) -> None:
        """One graph turn. The checkpoint is already flushed; this is the trail.

        `sync_s` is here to be read: the whole checkpoint history is re-pickled
        every turn, so this is the number that says whether that is still cheap.
        """
        self._step_no += 1
        self._append(
            self._steps,
            {
                "step": self._step_no,
                "node": node,
                "turn": values.get("turn"),
                "at": datetime.now(timezone.utc).isoformat(),
                "node_s": round(node_s, 3) if node_s is not None else None,
                "sync_s": round(sync_s, 3),
                "keys": keys,
            },
        )

    def record_state(self, values: dict) -> None:
        """A readable snapshot of the findings so far, for a run that never ends."""
        write_atomic(
            self.dir / "state.json",
            _dumps(
                {
                    "turn": values.get("turn"),
                    "done": values.get("done"),
                    "stop_reason": values.get("stop_reason"),
                    "section_statuses": values.get("section_statuses") or {},
                    "gaps": values.get("gaps") or [],
                    "context_items": values.get("context_items") or [],
                    "asked_questions": values.get("asked_questions") or [],
                    "decision_log": values.get("decision_log") or [],
                }
            ),
        )

    def record_telemetry(self, summary: dict) -> None:
        write_atomic(self.dir / "telemetry.json", _dumps(summary))

    def record_round(self, entry: dict, simulator_state: dict, rounds: int) -> None:
        """A question round, durable before the graph is allowed to consume it.

        The transcript line goes first and `rounds` pairs the simulator state with
        it, so a crash landing between the two writes is *detectable* on resume
        (see run_case) instead of silently re-spending dice already drawn.
        """
        self._append(self._transcript, entry)
        write_atomic(
            self.dir / "simulator.json", _dumps({"rounds": rounds, "state": simulator_state})
        )

    def record_segment(self, segment: dict) -> None:
        """Append one process's worth of wall-clock and telemetry."""
        write_atomic(self.dir / "segments.json", _dumps(self.load_segments() + [segment]))

    def record_status(self, state: str, **extra) -> None:
        """running | done | error | interrupted — an empty case dir says nothing."""
        payload = {"case_id": self.dir.name, "state": state,
                   "at": datetime.now(timezone.utc).isoformat(), **extra}
        write_atomic(self.dir / "status.json", _dumps(payload))

    def finish(self, record: dict, *, keep_checkpoints: bool = False) -> None:
        """Close the case out. predictions.json is written last, on purpose."""
        # A monkeypatched run_case never calls record_round, and a resumed case
        # already has its rounds on disk; reconcile rather than assume.
        transcript = record.get("transcript") or []
        if len(self.load_transcript()) != len(transcript):
            with open(self.dir / "transcript.jsonl", "w", encoding="utf-8") as f:
                for entry in transcript:
                    f.write(json.dumps(entry, ensure_ascii=False, default=_json_default) + "\n")

        self.record_telemetry(record.get("telemetry") or {})
        self.record_status(
            "error" if record.get("status") == "error" else "done",
            status=record.get("status"),
            finished=record.get("finished"),
            turns=record.get("turns"),
            rounds=record.get("rounds"),
            wall_s=record.get("wall_s"),
            segments=record.get("segments"),
        )
        write_atomic(
            self.dir / "predictions.json",
            _dumps({k: v for k, v in record.items() if k != "telemetry"}),
        )
        if not keep_checkpoints:
            # Pickled checkpoint history that nothing needs once predictions.json
            # exists, and that would otherwise dominate the run directory's size.
            checkpoints.discard(self.checkpoint_dir)


def load_case_row(case_out: Path) -> dict | None:
    """Rebuild a summary row from a previous run's artifacts, or None if unfinished.

    Reuses `summarize` so a resumed run and a straight-through one produce
    identical CSV rows. predictions.json holds every key it needs except
    telemetry, which lives beside it.
    """
    record = _read_json(case_out / "predictions.json")
    if not isinstance(record, dict) or "case_id" not in record:
        return None
    record["telemetry"] = _read_json(case_out / "telemetry.json") or {}
    try:
        return summarize(record)
    except (KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Driving the graph
# ---------------------------------------------------------------------------


def _drain(stream, on_update=None) -> list[dict]:
    """Consume a graph.stream() and collect any interrupts it emitted.

    `on_update` fires once per completed node — the hook the runner uses to flush
    the checkpoint after every graph turn.
    """
    interrupts: list = []
    for update in stream:
        if "__interrupt__" in update:
            interrupts.extend(update["__interrupt__"])
            continue
        if on_update is not None:
            on_update(update)
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


def merge_telemetry(summaries: list[dict]) -> dict:
    """Sum telemetry.summary() across the processes a case ran in.

    src/telemetry.py's collectors are process-global and reset at each case start,
    so a case resumed after a crash would otherwise report only its last segment,
    and summary.csv's llm_calls / llm_s / retries / cache_hits would undercount.
    Shares and means are ratios, so they are recomputed rather than averaged.
    """
    summaries = [s for s in summaries if s]
    if not summaries:
        return {}
    if len(summaries) == 1:
        return summaries[0]

    scalars = ("compute_s", "wait_s", "llm_s", "node_count", "llm_count",
               "retry_count", "failure_count", "cache_hit_count")
    merged: dict = {key: sum(s.get(key, 0) for s in summaries) for key in scalars}

    for group in ("by_node", "by_schema"):
        agg: dict[str, dict] = {}
        for summary in summaries:
            for name, slot in (summary.get(group) or {}).items():
                cur = agg.setdefault(name, {"count": 0, "total_s": 0.0, "max_s": 0.0})
                cur["count"] += slot.get("count", 0)
                cur["total_s"] += slot.get("total_s", 0.0)
                cur["max_s"] = max(cur["max_s"], slot.get("max_s", 0.0))
                for extra in ("llm_calls", "retried", "failed"):
                    if extra in slot:
                        cur[extra] = cur.get(extra, 0) + slot[extra]
        merged[group] = agg

    for group, total in (("by_node", merged["compute_s"]), ("by_schema", merged["llm_s"])):
        for slot in merged[group].values():
            slot["mean_s"] = slot["total_s"] / slot["count"] if slot["count"] else 0.0
            slot["share"] = slot["total_s"] / total if total else 0.0

    merged["llm_share_of_compute"] = (
        merged["llm_s"] / merged["compute_s"] if merged["compute_s"] else 0.0
    )
    return merged


def run_case(case: BenchmarkCase, *, simulator, loop_settings: LoopSettings,
             recorder: CaseRecorder | None = None) -> dict:
    """One full graph run. Returns the record written to predictions.json.

    With a `recorder` the run is durable: the checkpointer persists to disk and is
    flushed after every node, so calling this again with the same recorder
    continues from the last completed graph turn instead of starting the case
    over. Without one (the default, used by unit tests) it behaves as it always
    did — an in-memory checkpointer, and nothing written.
    """
    # Deterministic, unlike the uuid this used to carry: a resumed process has to
    # address the same thread. Checkpoints are per case directory, so the plain
    # case id cannot collide.
    thread_id = f"bench-{case.case_id}"
    telemetry.reset(thread_id)
    config = {"configurable": {"thread_id": thread_id}}

    resuming = recorder is not None and recorder.resumable()
    if recorder is None:
        graph = build_graph()  # MemorySaver: no Postgres needed for a benchmark run
        sync = None
        transcript: list[dict] = []
        prior_segments: list[dict] = []
    else:
        saver, sync = checkpoints.open_saver(recorder.checkpoint_dir)
        graph = build_graph(checkpointer=saver)
        # has_state() only proves the *file* holds something; this proves it holds
        # something for this thread. Otherwise a stale file would send the case
        # down the resume path and straight to a bogus "already finished".
        resuming = resuming and saver.get_tuple(config) is not None
        transcript = recorder.load_transcript() if resuming else []
        prior_segments = recorder.load_segments() if resuming else []
        saved = recorder.load_simulator_state() if resuming else None
        if saved:
            simulator.set_state(saved.get("state") or {})
            if saved.get("rounds") != len(transcript):
                print(
                    "    (état du simulateur décalé d'un tour par rapport au "
                    "transcript — le tirage aléatoire peut diverger)",
                    file=sys.stderr,
                )

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

    def on_update(update: dict) -> None:
        """One graph turn completed: make it durable before starting the next."""
        started_sync = time.perf_counter()
        if sync is not None:
            sync()  # the checkpoint first: it is the only part that is resumable
        sync_s = time.perf_counter() - started_sync
        if recorder is None:
            return
        try:
            values = graph.get_state(config).values
            runs = telemetry.node_runs()
            for node, node_values in update.items():
                recorder.record_step(
                    node=node,
                    values=values,
                    node_s=runs[-1].duration_s if runs else None,
                    sync_s=sync_s,
                    keys=sorted(node_values.keys()) if isinstance(node_values, dict) else [],
                )
            recorder.record_state(values)
            recorder.record_telemetry(telemetry.summary())
        except OSError as exc:  # a full disk must not kill a run that can still finish
            print(f"    (écriture de progression échouée : {exc})", file=sys.stderr)

    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    status = "ok"
    error = None

    try:
        if not resuming:
            _drain(graph.stream(input_state, config, stream_mode="updates"), on_update)
        elif graph.get_state(config).next and pending_questions(graph, config) is None:
            # A previous process died inside a node: LangGraph replays the pending
            # tasks, and every superstep before them stays done.
            _drain(graph.stream(None, config, stream_mode="updates"), on_update)

        # A round recorded but never consumed — the crash landed between the two.
        # Replay it rather than ask the simulator again, which would both spend
        # its RNG twice and duplicate the transcript entry.
        replayable = transcript[-1] if (resuming and transcript) else None

        while (questions := pending_questions(graph, config)) is not None:
            if len(transcript) >= MAX_ROUNDS:
                status = "round_limit"
                break
            values = graph.get_state(config).values

            if replayable is not None and replayable.get("questions") == questions:
                payload = {
                    a["gap_id"]: {"text": a["text"], "skip": a["skip"]}
                    for a in replayable["answers"]
                }
            else:
                batch: AnswerBatch = simulator.answer(questions, values)
                entry = {
                    "round": len(transcript),
                    "turn": values.get("turn"),
                    "questions": questions,
                    "answers": batch.as_records(),
                }
                transcript.append(entry)
                if recorder is not None:
                    recorder.record_round(entry, simulator.get_state(), len(transcript))
                payload = batch.resume_payload
            replayable = None

            _drain(graph.stream(Command(resume=payload), config, stream_mode="updates"), on_update)
    except Exception as exc:  # one bad case must not abort the batch
        status = "error"
        error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    finally:
        # Runs on KeyboardInterrupt too, so an interrupted case keeps its numbers.
        segment = {
            "started_at": started_at,
            "wall_s": round(time.perf_counter() - started, 2),
            "status": status,
            "telemetry": telemetry.summary(),
        }
        if recorder is not None:
            recorder.record_segment(segment)

    segments = prior_segments + [segment]
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
        # >1 means the case was interrupted and resumed; wall_s and telemetry are
        # sums across those processes.
        "segments": len(segments),
        "wall_s": round(sum(s["wall_s"] for s in segments), 2),
        "gaps": [g.model_dump() for g in values.get("gaps", [])],
        "context_items": [c.model_dump() for c in values.get("context_items", [])],
        "asked_questions": [q.model_dump() for q in values.get("asked_questions", [])],
        # The scorer's only source of retrieval rank order: which chunks came back
        # for a gap, in which order, lives in the rag_answer/rag_rejected entries'
        # evidence_ids and nowhere else (a rejected retrieval leaves no ContextItem
        # at all). The final validator's cross-section contradictions are likewise
        # only in its final_check entry's details.
        "decision_log": [d.model_dump() for d in values.get("decision_log", [])],
        "section_statuses": {
            sid: ss.model_dump() for sid, ss in (values.get("section_statuses") or {}).items()
        },
        "transcript": transcript,
        "telemetry": merge_telemetry([s["telemetry"] for s in segments]),
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
    "segments",
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
        "segments": record.get("segments", 1),
        "llm_calls": tele.get("llm_count", 0),
        "llm_s": round(tele.get("llm_s", 0.0), 2),
        "retries": tele.get("retry_count", 0),
        "cache_hits": tele.get("cache_hit_count", 0),
    }
    row["_categories"] = dict(categories)  # kept out of the CSV, used by report.md
    return row


def render_report(manifest: dict, rows: list[dict]) -> str:
    lines = [
        "# Benchmark run report",
        "",
        f"- run_id: `{manifest['run_id']}`",
        f"- dataset: `{manifest['dataset_version']}` ({len(rows)} case(s))",
        f"- simulator: `{manifest['simulator_mode']}` (seed {manifest['seed']})",
        f"- model: `{manifest['llm']['provider']}/{manifest['llm']['model']}` "
        f"@ temperature {manifest['llm']['temperature']}",
        f"- git: `{manifest['git_commit']}`" + (" (dirty)" if manifest["git_dirty"] else ""),
    ]
    if manifest.get("resumed_at"):
        lines.append(
            f"- reprises: {len(manifest['resumed_at'])} (dernière `{manifest['resumed_at'][-1]}`)"
        )
    lines += [
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
        "Descriptive only — nothing here is compared to ground truth. For precision/recall/F1,",
        "blocking-gap recall, RAG Recall@K and question quality, score the run:",
        "",
        f"    uv run python evals/run_scoring.py --run {manifest['run_id']}",
        "",
        "which reads these `predictions.json` files against `ground_truth.json` and writes",
        "`scores.md` beside this file.",
        "",
    ]
    return "\n".join(lines)


def write_report(path: Path, manifest: dict, rows: list[dict]) -> None:
    write_atomic(path, render_report(manifest, rows))


def flush_run_outputs(results_root: Path, manifest: dict, rows: list[dict]) -> None:
    """Rewrite the run-level artifacts. Called after *every* case, not just the last.

    Whole-file rewrites are fine because write_atomic never leaves a partial file
    and ten rows is nothing — so there is no need for an append-only ledger on the
    side that would then have to be reconciled with these.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    write_atomic(results_root / "summary.csv", buffer.getvalue())
    write_report(results_root / "report.md", manifest, rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cases",
                   help="comma-separated case ids (default: every case in the dataset manifest, "
                        "or — with --resume — every case the run being resumed covered)")
    p.add_argument("--mode", choices=MODES, default="oracle", help="synthetic stakeholder mode")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the realistic simulator")
    p.add_argument("--provider", choices=["ollama", "anthropic"], help="override llm.provider")
    p.add_argument("--model", help="override llm.model")
    p.add_argument("--temperature", type=float, help="override llm.temperature")
    p.add_argument("--top-k", type=int, help="override rag.top_k")
    p.add_argument("--max-turns", type=int, help="override loop.max_turns")
    p.add_argument("--max-questions-per-batch", type=int, help="override loop.max_questions_per_batch")
    p.add_argument("--cache", action="store_true", help="enable the LLM disk cache for this run")
    p.add_argument("--score", action="store_true",
                   help="score the run against ground truth once it finishes "
                        "(same as running evals/run_scoring.py afterwards)")
    p.add_argument("--out", type=Path, help=f"results root (default: {RESULTS_DIR})")

    naming = p.add_mutually_exclusive_group()
    naming.add_argument("--run-id", help="name this run (default: bench_<UTC timestamp>)")
    naming.add_argument(
        "--resume",
        nargs="?",
        const="@latest",
        metavar="RUN_ID",
        help="continue an interrupted run (default: the most recent one under --out)",
    )
    p.add_argument("--force", action="store_true",
                   help="resume even if the run was recorded with a different config")
    p.add_argument("--keep-checkpoints", action="store_true",
                   help="keep a case's checkpoint files after it completes")
    return p


def latest_run(root: Path) -> Path | None:
    """The most recently touched run directory that got as far as a manifest."""
    if not root.is_dir():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
    return max(candidates, key=lambda p: p.stat().st_mtime, default=None)


def _get(manifest: dict, path: tuple[str, ...]):
    node = manifest
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def manifest_conflicts(previous: dict, current: dict) -> list[str]:
    """Fields that differ badly enough to make one summary.csv meaningless."""
    return [
        ".".join(path)
        for path in RESUME_MUST_MATCH
        if _get(previous, path) != _get(current, path)
    ]


def resolve_run_dir(args, results_dir: Path) -> Path | None:
    """Where this invocation writes: a fresh directory, or the run being resumed."""
    if args.resume is None:
        run_id = args.run_id or f"bench_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
        return results_dir / run_id
    if args.resume == "@latest":
        found = latest_run(results_dir)
        if found is None:
            print(f"Nothing to resume under {results_dir}.", file=sys.stderr)
        return found
    candidate = results_dir / args.resume
    if not (candidate / "manifest.json").exists():
        print(f"No run to resume at {candidate}.", file=sys.stderr)
        return None
    return candidate


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir = args.out or RESULTS_DIR
    resuming = args.resume is not None

    results_root = resolve_run_dir(args, results_dir)
    if results_root is None:
        return 1
    run_id = results_root.name
    results_root.mkdir(parents=True, exist_ok=True)

    if args.cache:
        # Read at call time by src/llm_cache.py, so setting it here is enough.
        os.environ["CDC_LLM_CACHE"] = "1"
        os.environ.setdefault("CDC_LLM_CACHE_DIR", str(ROOT / ".cache" / "llm"))

    previous = (_read_json(results_root / "manifest.json") or {}) if resuming else {}

    # A resumed run covers exactly what the original one covered. Re-deriving this
    # from the dataset default would quietly turn `--cases one_case` + `--resume`
    # into a ten-case run, starting with cases the original never touched.
    # `--cases` on top of that narrows which of them this process picks up.
    run_case_ids = list(previous.get("case_ids") or [])
    selected = [c.strip() for c in args.cases.split(",")] if args.cases else (run_case_ids or None)

    cases = load_benchmark(case_ids=selected)
    if not cases:
        print("No cases selected.")
        return 1
    # Anything selected but not yet part of the run joins it, so its row is not
    # computed and then dropped on the floor by flush().
    run_case_ids += [c.case_id for c in cases if c.case_id not in run_case_ids]

    base = load_settings()
    loop_update = {}
    if args.max_turns is not None:
        loop_update["max_turns"] = args.max_turns
    if args.max_questions_per_batch is not None:
        loop_update["max_questions_per_batch"] = args.max_questions_per_batch
    loop_settings = base.loop.model_copy(update=loop_update) if loop_update else base.loop

    # Fold the CLI overrides in once, before the loop. The manifest is written up
    # front so an interrupted run still has one, which means it must not depend on
    # whichever case happened to run last — and run_manifest only reads
    # case-independent fields anyway. Handing the base dirs back in leaves them
    # untouched; the per-case call below is what redirects RAG and output.
    effective_settings = harness.case_settings(
        base,
        source_dir=base.rag.source_dir,
        persist_dir=base.rag.persist_dir,
        output_dir=base.quarto.output_dir,
        top_k=args.top_k,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
    ).model_copy(update={"loop": loop_settings})

    started_at = datetime.now(timezone.utc).isoformat()
    manifest = harness.run_manifest(
        run_id=run_id,
        dataset_version=dataset_version(),
        case_ids=run_case_ids,
        simulator_mode=args.mode,
        seed=args.seed,
        settings=effective_settings,
        started_at=started_at,
    )

    if resuming:
        conflicts = manifest_conflicts(previous, manifest)
        if conflicts and not args.force:
            print(
                f"Refusing to resume {run_id}: it was recorded with a different "
                f"{', '.join(conflicts)}. Pass --force to mix them anyway.",
                file=sys.stderr,
            )
            return 1
        # The original run's identity and provenance win; this only records that
        # another process picked it up.
        previous["resumed_at"] = list(previous.get("resumed_at") or []) + [started_at]
        previous["case_ids"] = run_case_ids
        manifest = previous
    write_atomic(results_root / "manifest.json", _dumps(manifest))

    print(f"run {run_id}: {len(cases)} case(s), mode={args.mode} -> {results_root}")

    # Keyed by case id and emitted in the run's own order, so a resume that fills
    # in the holes still writes the CSV in the dataset's order — and one that
    # narrows to a single case with --cases still carries the others' rows.
    rows_by_case: dict[str, dict] = {}
    if resuming:
        for case_id in run_case_ids:
            # predictions.json is the only completion marker; by this run's
            # contract a case that ended in error counts as done, not as a retry.
            done = load_case_row(results_root / case_id)
            if done is not None:
                rows_by_case[case_id] = done

    def flush() -> None:
        flush_run_outputs(
            results_root,
            manifest,
            [rows_by_case[cid] for cid in run_case_ids if cid in rows_by_case],
        )

    flush()  # an empty-but-valid summary.csv beats none at all
    interrupted: str | None = None

    for case in cases:
        case_out = results_root / case.case_id

        if case.case_id in rows_by_case:
            print(f"  - {case.case_id} ... déjà terminé, ignoré")
            continue

        settings = harness.case_settings(
            effective_settings,
            source_dir=case.source_docs_dir,
            # Per-case Chroma index: otherwise every case retrieves from every
            # other case's reference documents.
            persist_dir=ROOT / ".cache" / "bench" / case.case_id / "chroma",
            output_dir=case_out / "artifacts",
        )

        print(f"  - {case.case_id} ... ", end="", flush=True)
        with CaseRecorder(case_out, resume=resuming) as recorder:
            recorder.record_status("running", started_at=datetime.now(timezone.utc).isoformat())
            # isolate() drops the memoized chat models, so a provider/model override
            # carried in `settings` is enough — no need to rebind the agent modules
            # the way run_evals.py has to.
            try:
                with harness.isolate(settings):
                    simulator = make_simulator(
                        args.mode,
                        ground_truth=case.ground_truth,
                        profile=case.stakeholder,
                        seed=args.seed,
                    )
                    record = run_case(
                        case, simulator=simulator, loop_settings=loop_settings, recorder=recorder
                    )
            except KeyboardInterrupt:
                # The checkpoint is already on disk, flushed at the last graph turn.
                recorder.record_status("interrupted")
                interrupted = case.case_id
                print("interrompu")
                break
            recorder.finish(record, keep_checkpoints=args.keep_checkpoints)

        row = summarize(record)
        rows_by_case[case.case_id] = row
        print(
            f"{row['status']} — {row['turns']} turn(s), {row['gaps_total']} gap(s), "
            f"{row['questions_asked']} question(s), {row['wall_s']}s"
        )
        if record["error"]:
            print(record["error"], file=sys.stderr)
        flush()

    flush()
    print(f"\nwrote {results_root}")

    # Scoring is a separate, re-runnable pass by design (evals/run_scoring.py);
    # this is only the convenience of not having to type the second command. An
    # interrupted run is scored on the next --resume, not on a partial batch.
    if args.score and not interrupted:
        from evals import run_scoring

        run_scoring.main(["--run", run_id, "--out", str(results_dir)])

    if interrupted:
        print(
            f"interrompu pendant {interrupted} — reprendre avec :\n"
            f"  uv run python evals/run_benchmark.py --resume {run_id}",
            file=sys.stderr,
        )
        return 130

    failed = [r["case_id"] for r in rows_by_case.values() if r["status"] == "error"]
    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
