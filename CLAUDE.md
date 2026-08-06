# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A multi-agent system ("swarm") that iteratively refines a vague **cahier des charges**
(CDC — a French-style requirements spec) into a document precise enough to hand to a
dev team. It detects gaps/ambiguities, resolves them via RAG when possible, asks the
author targeted questions otherwise, checks new info for contradictions, and finally
synthesizes a DOCX (via Quarto) plus a QA report. The UI language is French. The only
interface is a Streamlit app — there is no CLI (`main.py` is an unused `uv` stub).

Full architecture docs live in `docs/00-overview.md` through `docs/07-evaluation.md`.
`prompt.md` at the repo root is the original build spec and a useful intent reference.

## Commands

Uses [uv](https://docs.astral.sh/uv/) (Python 3.11+). Do not invoke `pip`/`python` directly.

```bash
uv sync                              # install deps (incl. dev group)
uv run streamlit run src/app.py      # run the app (the real entrypoint)
uv run pytest                        # run the full test suite
uv run pytest tests/test_routing.py::test_name   # single test
uv run python scripts/hash_password.py           # bcrypt-hash a password for secrets.toml
uv run langgraph dev                 # LangGraph dev server (graph exposed via langgraph.json)

uv run python evals/run_evals.py                 # component evals (gap_finder / critic, no graph)
uv run python evals/run_benchmark.py             # full-graph benchmark, synthetic stakeholder
uv run python evals/run_benchmark.py --cases cdc_003_ecommerce --cache   # one case, cached
uv run python evals/run_benchmark.py --resume                            # continue the last run
uv run python evals/run_scoring.py               # score the last run against ground truth
uv run python evals/run_scoring.py --run <run_id> --judge llm            # + LLM question judge
```

There is no linter/formatter configured. Match the surrounding style (`from __future__
import annotations`, type hints, French docstrings/log strings in graph/agent code).

## Architecture

The whole system is a **LangGraph state machine** (`src/graph.py`). One shared, typed
state object (`CDCState` in `src/state.py`) flows through fixed agent nodes, with a
human-in-the-loop `interrupt` for question/answer turns. Key invariants — respect these
when editing:

- **Only `graph.py` node wrappers mutate state.** Each agent in `src/agents/` is a pure
  function `(CDCState, ...) -> result`; it returns data, it does not touch the graph.
  Keep agent logic testable in isolation and keep all control flow (who runs next, and
  why) in `graph.py`'s routing functions.
- **Sections are config, not code.** `config/sections.yaml` defines what a CDC must
  cover. Never hard-code section names/ids; load them via `src/config.py` and reference
  by `id` (str). `state.py`'s comment reiterates this.
- **Every LLM call returns typed data.** All agents go through `call_structured(prompt,
  PydanticModel)` in `src/llm.py`, which asks for JSON matching the model's schema and
  parses it robustly (handles code fences, retries with the parse error fed back). Do
  not import a provider-specific chat class in agent code.
- **Provider is a config flip.** `src/llm.py`'s factory builds Ollama (default) or
  Anthropic from `config/settings.yaml`'s `llm.provider`. Adding a provider means
  extending `_build_llm`, not changing agents.
- **RAG answers and human answers are the same kind of thing.** Both become a
  `ContextItem` tagged with a `source` (`initial_cdc` | `rag` | `user_answer` |
  `assumption`) and `section_ids`. Downstream agents don't care where an answer came from.
- **Nothing is silently dropped.** User-doesn't-know → an `assumption` ContextItem,
  flagged in the final doc. Context mapped to no section, and contradictions found on a
  final read, land in `output/qa_report.md`.
- **Every decision is auditable.** Each `ContextItem` carries its provenance
  (`created_by`, `evidence` chunks with document/page, `confidence`,
  `validation_status`, `model`, `prompt_version`); each meaningful agent decision
  appends a `DecisionLogEntry` to `state["decision_log"]` (checkpointed, dumped to
  `output/decision_log.jsonl`). Agents build these and return them — only `graph.py`
  writes them into state. Every `call_structured` passes a `prompt_id` registered in
  `src/prompts.py`; **bump its version there when you edit a prompt's wording.**

### Graph flow

```
START → ingest → initial_scan → orchestrator → (route)
  gap_finder → gap_filler → (pending questions?)
     → human_input (interrupt) → integrate_answers → critic → orchestrator
     → critic → orchestrator
  → synthesizer → final_validator → END          (all required sections complete)
  → END                                          (blocking gaps still open at max_turns)
```

`ingest` splits the initial CDC per section (`src/utils/cdc_sections.py`) so prompts
carry only the slices they need; unmatched chunks become untagged items surfaced later
as "unmapped". `initial_scan` fans out one gap-finder pass per section in parallel.
The orchestrator (`src/agents/orchestrator.py`) owns section picking, loop-limit /
`max_turns` handling, and the question-dedup gate.

### Concurrency & telemetry

- `map_structured(jobs)` in `src/llm.py` runs independent `call_structured` callables on
  a thread pool (LLM calls are network-bound), preserving input order for `zip`-back.
  Capped at `_MAX_FANOUT_WORKERS = 8` to respect Ollama-cloud rate limits.
- `src/telemetry.py` records per-node wall-clock and nested LLM-call timings. Nodes are
  wrapped with `_timed(...)` at registration in `build_graph` (not decorated), so the
  raw node functions stay directly callable in tests. Fanned-out workers re-bind the
  enclosing node via `telemetry.bound_to`, so attribute new concurrency accordingly.

### Persistence

`src/db.py` — Supabase (Postgres). `PostgresSaver` is the LangGraph checkpointer (runs
survive restarts, keyed by `thread_id`); a `runs` table indexes threads per user for
"my runs" history. All durable run state lives in the checkpointer — the Streamlit app
re-reads it on every rerun rather than trusting in-memory globals. **Use the Supabase
SESSION pooler (port 5432) or a direct connection — NOT the transaction pooler (6543)**,
which breaks the prepared-statement config.

### Evaluation

Three harnesses in `evals/`. `run_evals.py` calls single agents directly (fast prompt
iteration). `run_benchmark.py` drives the **compiled graph** end to end over
`evals/datasets/benchmark/` — ten annotated CDC cases with ground-truth gaps,
contradictions and per-case reference documents — with a synthetic stakeholder
(`evals/simulator.py`) answering each `interrupt()`; it emits predictions and
descriptive stats, never scores. `run_scoring.py` does the scoring, as a **separate
pass over a finished run directory**: it is offline and takes seconds, so an improved
scorer can be replayed over a run that already cost two hours. The first two share
`evals/harness.py`.

Two things to respect when touching the dataset: a `resolvable_by: "rag"` ground-truth
gap must not also carry an `expected_answer` (the oracle would mask a retrieval
failure), and each case runs under `harness.isolate(...)` so its RAG corpus, Chroma
index and output dir stay private. Full reference: `docs/07-evaluation.md`.

**Scoring** (`evals/scoring.py` = pure functions, `run_scoring.py` = CLI/report/IO).
Predicted↔annotated matching is by *content* — runtime gap ids are content hashes —
reusing `keyword_score` from `simulator.py` so the scorer and the oracle can't drift.
Four invariants worth keeping: the scorer never writes to what it reads; precision is
strict (an unannotated detection is a false positive) so `unmatched_predictions` is
always printed for review; the critic's answer-vs-answer contradictions can raise
recall but are never false positives, because ground truth only annotates
CDC-internal ones; and `predictions.json` carries `decision_log` because RAG rank
order lives nowhere else (a *rejected* retrieval leaves no `ContextItem`, only a
`rag_rejected` entry with `evidence_ids`). Question quality is deterministic by
default; `--judge llm` adds an LLM rubric beside it, never merged into it.

**The benchmark is resumable, so nothing is buffered until the end.** Every artifact
goes through `write_atomic` and is written as it happens — the graph checkpoint after
each node, the transcript after each question round, `summary.csv`/`report.md` after
each case, `manifest.json` before the loop. Keep it that way when editing
`run_benchmark.py`. `predictions.json` is the completion marker `--resume` reads (and
a case that ended in error counts as done). Mid-case resume works because
`evals/checkpoints.py` backs LangGraph's `InMemorySaver` with `PersistentDict` on
disk — no extra dependency — and `thread_id` is deterministic (`bench-<case_id>`).
Anything stateful a case relies on must be persisted per turn too: that is why the
stakeholder simulator has `get_state`/`set_state`, and why per-process telemetry is
summed back together by `merge_telemetry`.

## Configuration & secrets

- `config/settings.yaml` — `llm` / `embeddings` / `rag` / `loop` / `quarto` settings,
  loaded into typed models by `src/config.py` (memoized via `lru_cache`; call
  `clear_config_cache()` after editing YAML on disk, as tests do). The eval harness
  swaps settings per benchmark case via `set_settings_override(...)`, which sits in
  front of the cache; outside the harness it is unset and nothing changes.
- `src/.env` — API keys (`OLLAMA_API_KEY` or `ANTHROPIC_API_KEY`), and optional
  `CDC_LLM_CACHE=1` to memoize structured LLM calls to `.cache/llm/` for fast dev
  iteration (off by default; see `src/llm_cache.py`). Loaded by `src/config.py`.
- `.streamlit/secrets.toml` — auth + DB (gitignored; copy from `secrets.toml.example`).
  `COOKIE_KEY` (JWT signing), `SUPABASE_DB_URL`, and a closed `[credentials.usernames.*]`
  list with **bcrypt** password hashes (never plaintext) via `scripts/hash_password.py`.
  Auth uses `streamlit-authenticator`. On Streamlit Cloud, paste the same content into
  the app's Secrets settings.

`.gitignore` excludes `output/*`, `evals/results/*`, `.chroma`, `.cache`, `.env`,
`.streamlit`, `prompt.md`. The benchmark **datasets** are committed — they are ground truth.

## Gotchas

- **`torch` is pinned CPU-only** via a dedicated `pyproject.toml` index, so Streamlit
  Cloud (Linux) doesn't pull multi-GB CUDA wheels. Keep that when touching deps.
- **Embeddings stay local** even when the chat LLM is cloud Ollama: `rag.py` probes the
  local Ollama embedding endpoint and silently falls back to a `sentence-transformers`
  model (`all-MiniLM-L6-v2`) if unavailable. The chat `base_url` is separate.
- **Stable IDs matter.** `src/ids.py::stable_id(...)` derives content-hashed ids for
  context items/answers so re-running the same CDC keeps hitting the LLM disk cache.
  Don't switch these to random uuids.
- A cold run spends 5+ minutes in LLM warm-up before the first question; `CDC_LLM_CACHE=1`
  replays it in well under a second.
