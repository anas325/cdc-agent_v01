"""Process-global timing collector for graph nodes and LLM calls.

Deliberately *not* part of CDCState: state is checkpointed and most of its
fields have no reducer, so writing telemetry there would either be clobbered
by concurrent updates or bloat every checkpoint. The Streamlit UI is a
single-session debug tool running in one process, so a module-level singleton
is the right scope.

Attribution works via a ContextVar holding the node currently executing, so
`record_llm` calls made deep inside an agent land on the enclosing NodeRun
without any plumbing through function signatures.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

# human_input pauses on interrupt() — its "duration" is the user thinking, not
# compute, so it is tracked separately and excluded from compute totals.
WAIT_NODES = {"human_input"}


@dataclass
class LLMCall:
    node: str
    schema: str
    model: str
    duration_s: float
    attempts: int
    ok: bool
    prompt_chars: int
    response_chars: int
    started_at: float
    error: str | None = None
    cache_hit: bool = False
    prompt_id: str | None = None
    prompt_version: str | None = None


@dataclass
class NodeRun:
    node: str
    duration_s: float
    started_at: float
    llm_calls: list[LLMCall] = field(default_factory=list)
    error: str | None = None

    @property
    def is_wait(self) -> bool:
        return self.node in WAIT_NODES

    @property
    def label(self) -> str:
        return f"{self.node} (attente)" if self.is_wait else self.node

    @property
    def llm_seconds(self) -> float:
        return sum(c.duration_s for c in self.llm_calls)


_node_runs: list[NodeRun] = []
_llm_calls: list[LLMCall] = []
_run_started_at: float | None = None
_thread_id: str | None = None

_current_node: ContextVar[NodeRun | None] = ContextVar("current_node", default=None)


def reset(thread_id: str | None = None) -> None:
    """Drop everything collected so far. Called when a new run starts."""
    global _run_started_at, _thread_id
    _node_runs.clear()
    _llm_calls.clear()
    _run_started_at = None
    _thread_id = thread_id
    _current_node.set(None)


def thread_id() -> str | None:
    return _thread_id


def run_started_at() -> float | None:
    return _run_started_at


@contextmanager
def record_node(name: str):
    """Time one graph node; LLM calls inside it attribute to this run."""
    global _run_started_at
    started = time.perf_counter()
    if _run_started_at is None:
        _run_started_at = started

    run = NodeRun(node=name, duration_s=0.0, started_at=started)
    token = _current_node.set(run)
    try:
        yield run
    except BaseException as exc:  # noqa: BLE001 - recorded then re-raised
        # GraphInterrupt from interrupt() lands here too; it is control flow,
        # not a failure, so only real exceptions get an error label.
        if type(exc).__name__ not in ("GraphInterrupt", "GraphBubbleUp"):
            run.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        run.duration_s = time.perf_counter() - started
        _current_node.reset(token)
        _node_runs.append(run)


def current_node_name() -> str:
    run = _current_node.get()
    return run.node if run else "(hors nœud)"


def current_run() -> NodeRun | None:
    """The NodeRun currently in scope, for re-binding across worker threads."""
    return _current_node.get()


@contextmanager
def bound_to(run: NodeRun | None):
    """Bind `run` as the current node inside a worker thread.

    A fanned-out LLM call runs in a fresh thread whose ContextVar starts empty;
    binding the parent's NodeRun here makes record_llm attribute the call to the
    enclosing node instead of "(hors nœud)".
    """
    token = _current_node.set(run)
    try:
        yield
    finally:
        _current_node.reset(token)


def record_llm(
    *,
    schema: str,
    model: str,
    duration_s: float,
    attempts: int,
    ok: bool,
    prompt_chars: int,
    response_chars: int,
    started_at: float,
    error: str | None = None,
    cache_hit: bool = False,
    prompt_id: str | None = None,
    prompt_version: str | None = None,
) -> None:
    call = LLMCall(
        node=current_node_name(),
        schema=schema,
        model=model,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        duration_s=duration_s,
        attempts=attempts,
        ok=ok,
        prompt_chars=prompt_chars,
        response_chars=response_chars,
        started_at=started_at,
        error=error,
        cache_hit=cache_hit,
    )
    _llm_calls.append(call)
    run = _current_node.get()
    if run is not None:
        run.llm_calls.append(call)


def current_node_elapsed() -> float | None:
    """Seconds spent so far in the enclosing node, for in-flight logging."""
    run = _current_node.get()
    return None if run is None else time.perf_counter() - run.started_at


def node_runs() -> list[NodeRun]:
    return list(_node_runs)


def llm_calls() -> list[LLMCall]:
    return list(_llm_calls)


def _aggregate(rows: list[tuple[str, float]]) -> dict[str, dict]:
    agg: dict[str, dict] = {}
    for key, seconds in rows:
        slot = agg.setdefault(key, {"count": 0, "total_s": 0.0, "max_s": 0.0})
        slot["count"] += 1
        slot["total_s"] += seconds
        slot["max_s"] = max(slot["max_s"], seconds)
    for slot in agg.values():
        slot["mean_s"] = slot["total_s"] / slot["count"]
    return agg


def summary() -> dict:
    """Per-node and per-LLM-schema aggregates, plus overall totals.

    `compute_s` excludes wait nodes so the shares answer "where did the
    machine time go", not "how long did the human take to answer".
    """
    compute_runs = [r for r in _node_runs if not r.is_wait]
    compute_s = sum(r.duration_s for r in compute_runs)
    wait_s = sum(r.duration_s for r in _node_runs if r.is_wait)
    llm_s = sum(c.duration_s for c in _llm_calls)

    by_node = _aggregate([(r.node, r.duration_s) for r in compute_runs])
    for name, slot in by_node.items():
        slot["share"] = slot["total_s"] / compute_s if compute_s else 0.0
        slot["llm_calls"] = sum(len(r.llm_calls) for r in compute_runs if r.node == name)

    by_schema = _aggregate([(c.schema, c.duration_s) for c in _llm_calls])
    for name, slot in by_schema.items():
        slot["share"] = slot["total_s"] / llm_s if llm_s else 0.0
        slot["retried"] = sum(1 for c in _llm_calls if c.schema == name and c.attempts > 1)
        slot["failed"] = sum(1 for c in _llm_calls if c.schema == name and not c.ok)

    return {
        "compute_s": compute_s,
        "wait_s": wait_s,
        "llm_s": llm_s,
        "llm_share_of_compute": llm_s / compute_s if compute_s else 0.0,
        "node_count": len(compute_runs),
        "llm_count": len(_llm_calls),
        "retry_count": sum(1 for c in _llm_calls if c.attempts > 1),
        "failure_count": sum(1 for c in _llm_calls if not c.ok),
        "cache_hit_count": sum(1 for c in _llm_calls if c.cache_hit),
        "by_node": by_node,
        "by_schema": by_schema,
    }


def format_duration(seconds: float) -> str:
    """Compact, consistently formatted durations for the UI."""
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)} min {rest:04.1f} s"
