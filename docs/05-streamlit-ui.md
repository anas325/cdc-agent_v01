# Streamlit UI

[`src/app.py`](../src/app.py) is the only interface to the system — there is
no CLI beyond ad-hoc scripts. It's intentionally "functional, testing-focused
— not polished" (per `prompt.md`): single Streamlit session, single run at a
time, no auth, no multi-user support.

Run it with:

```
uv run streamlit run src/app.py
```

The page is split into two tabs:

- **Pilotage** — drive a run: start it, watch nodes complete, read section
  status and gaps, answer question batches, download the output.
- **Sous le capot** — everything needed to debug the swarm: where the wall
  clock went (per node, per LLM call), the raw gap table, the turn log, and
  the checkpointer snapshot.

## Authentication

Access is gated by [`streamlit-authenticator`](https://github.com/mkhorasani/Streamlit-Authenticator)
against a **closed, admin-managed** user list in `st.secrets`
(`[credentials.usernames.*]`, bcrypt-hashed passwords — no self-registration).
Login is persisted in a signed JWT **cookie** (30-day expiry), so a reload, a
new tab, or a server restart keep the user logged in — unlike the old
`st.session_state`-only password gate. The logged-in `username` scopes every
run to its owner. See `_get_authenticator()` / the gate at the top of
`src/app.py`, and add users with `scripts/hash_password.py`.

## The core design constraint: no in-memory state

The graph is compiled once per Streamlit process (`get_graph()`, decorated
with `@st.cache_resource`), but **all run state lives in the LangGraph
checkpointer** — a `PostgresSaver` backed by Supabase (`src/db.py`), keyed by a
`thread_id`. Every Streamlit rerun (which happens on every widget interaction)
re-reads state via `get_graph().get_state(config)` rather than trusting anything
held in Python globals between reruns. Because checkpoints are now durable in
Postgres, runs survive server restarts: a `runs` table maps each `thread_id` to
its owning `username`, the sidebar's **Mes runs** list lets a user reopen any
past run (`load_run_into_session()`), and a fresh session auto-reopens the most
recent one (`_autoload_last_run()`).

`current_config()` returns `{"configurable": {"thread_id": ...}}` — the
standard LangGraph checkpointer key, threaded through every `stream`/
`get_state` call.

**Timings are the one deliberate exception.** They live in
[`src/telemetry.py`](../src/telemetry.py), a process-global collector, *not*
in `CDCState` — see [Telemetry](#telemetry) below for why.

## Session state fields

```python
thread_id: str | None            # UUID, minted on "Démarrer" click
run_active: bool                 # True while the graph is streaming (blocks re-click)
pending_questions: list | None   # current interrupt's question batch, or None
finished: bool                   # True once the graph reached done=True
last_step_seconds: float | None  # wall clock of the most recent step
step_timeline: list[float]       # one entry per completed step, for the cumulative metric
```

Widget-keyed state uses two disjoint namespaces on purpose: `section_skip_*`
for the sidebar's per-section skip checkboxes, and `qa_answer_*` / `qa_skip_*`
for the answer form. They used to share a `skip_{id}` prefix, which would
collide if a section id ever equalled a gap id.

## Telemetry

`src/telemetry.py` is a module-level singleton holding `NodeRun` and `LLMCall`
records for the current run. It is deliberately *not* part of `CDCState`:
state is checkpointed and most of its fields have no reducer, so telemetry
written there would either be clobbered by concurrent updates or bloat every
checkpoint. Since the UI is a single-session debug tool in one process,
process scope is the right scope.

Two instrumentation points, both applied without touching agent code:

- **Nodes** — `_timed(name, fn)` in [`src/graph.py`](../src/graph.py) wraps
  each node *at registration* in `build_graph()`, so the ten node functions
  stay directly callable and testable without telemetry attached.
- **LLM calls** — `call_structured()` in [`src/llm.py`](../src/llm.py) records
  one `LLMCall` per invocation (covering all its repair retries) from a
  `finally` block, so failed and timed-out calls still show up. Attribution to
  the enclosing node happens through a `ContextVar`, so nothing has to be
  threaded through function signatures.

`_log()` in `graph.py` also stamps `details["_elapsed_s"]` onto every
`TurnLogEntry`, so the turn log carries timing too.

Two things the aggregates handle specially:

- `human_input` is a **wait node** — its "duration" is the human thinking, not
  compute. It's recorded and labelled `human_input (attente)`, but excluded
  from `compute_s` and from the per-node shares, so "where did the machine
  time go" isn't drowned out by "how long did the user take".
- `interrupt()` raises `GraphInterrupt` through the node wrapper. That's
  control flow, not failure, so `record_node` doesn't mark it as an error.

## Flow

### 1. Sidebar — `render_sidebar(settings)`

- CDC file uploader (`.md`/`.txt`/`.pdf`) → `.md`/`.txt` are read as text
  directly; a `.pdf` is first written into `data/source_docs/` and then
  extracted via `rag._extract_text_from_path()` (see
  [LLM & RAG](03-llm-and-rag.md#ingestion)). Either way the result becomes
  `initial_cdc_text` on run start.
- Source-doc uploader (multiple `.md`/`.txt`/`.pdf`) → written directly into
  `data/source_docs/`. Note: this does **not** trigger ingestion
  immediately — files are only (re-)indexed the next time a run starts,
  since `ingest_source_docs()` runs inside the graph's `ingest` node.
- Loop-setting number inputs (`max_turns`, `max_questions_per_batch`,
  `max_questions_per_gap`), defaulting to `settings.loop.*` from
  `config/settings.yaml` but overridable per run.
- "Sections à ignorer" — one checkbox per required section (from
  `load_sections()`), unchecked by default. Returned as `skip_map`
  (`section_id -> bool`) and consumed by `start_run` to pre-seed those
  sections as `skipped` so the orchestrator never picks them (see
  [Graph & Agents](02-graph-and-agents.md#orchestrator)).
- "Démarrer un nouveau run" button — disabled while `run_active` is true.
- Once a thread exists: the `thread_id` and a "Réinitialiser la télémétrie"
  button, for clearing timings without starting a new run.

### 2. Running a step — `run_graph(payload)`

Both starting a run and resuming from an answer batch go through the same
helper. It **streams** rather than blocking:

```python
for update in graph.stream(payload, current_config(), stream_mode="updates"):
    ...  # one st.write line per completed node, with its duration
```

This matters: the first step alone runs one LLM call per section inside
`initial_scan` (sequentially — see
[Graph & Agents](02-graph-and-agents.md)), and that used to sit behind a
single opaque `st.spinner` with no feedback. Now each node reports as it
finishes, inside an `st.status` block that collapses to
"Terminé en 1 min 12.4 s" when the step ends.

`start_run` mints a fresh `thread_id` (so every "Démarrer" click starts an
unrelated new run; resuming happens implicitly by *not* clicking it and just
reloading the page while `thread_id` is still set), resets telemetry, and
seeds `section_statuses` — every required section `"skipped"` if `skip_map`
marked it, otherwise `"empty"`.

Exceptions during the stream are caught and shown with `st.exception` and an
error-state status block, rather than blanking the page — this is a debug
tool, and a traceback is the point.

### 3. Interpreting a step result — `process_step_result(result)`

```python
if result.get("__interrupt__"):
    pending_questions = result["__interrupt__"][0].value["questions"]
    finished = False
else:
    pending_questions = None
    finished = bool(result.get("done"))
```

`__interrupt__` is LangGraph's convention for surfacing an active
`interrupt()` call; its `.value` is exactly the dict passed to
`interrupt({"questions": [...]})` in `human_input_node`. In streaming mode it
arrives as its own update; `run_graph` falls back to reading the
checkpointer's pending `tasks[*].interrupts` in case a future LangGraph
version stops emitting it.

### 4. Tab "Pilotage"

- **`render_run_metrics`** — a row of `st.metric`: last step duration,
  cumulative time, current turn, open-gap count (with a blocking-gap delta),
  and sections completed.
- **`render_status_table`** — one row per configured section: status
  (emoji + label — ⚪ empty, 🟡 in_progress, 🟢 complete, 🔴 reopened,
  ⚫ skipped), a `Score` column, and open-gap counts broken down by severity.
  Rebuilt fresh from `section_statuses` + `gaps` every rerun — no separate
  UI-side bookkeeping. `score_section(gaps)` starts at 100 and subtracts, per
  open gap, `SEVERITY_WEIGHTS[severity] * CATEGORY_WEIGHTS[category]`
  (severity 10/5/1 for blocking/important/nice_to_have; category multiplies
  that by 0.8–2.0, contradiction weighted highest), floored at 0 — a quick
  at-a-glance proxy for how "risky" a section still is, purely a UI
  computation with no effect on routing.
- **`render_question_form`** — one text input + "je ne sais pas" checkbox per
  pending question, inside a single `st.form` (so all answers submit together,
  matching the batching the orchestrator already enforces), each prefixed with
  its gap's severity/category badges. The form has two submit buttons:
  - **"Envoyer les réponses"** builds `{gap_id: {"text": ..., "skip": ...}}`
    and calls `run_graph(Command(resume=answers))`.
  - **"Passer la section « … »"** skips the whole section the batch belongs to
    — computed as the earliest section (in `sections.yaml` order) among the
    pending gaps — by resuming with the `{"__skip_section__": <section_id>}`
    signal that `integrate_answers_node` recognizes. This lets the user jump to
    the next section without answering the current one; that section's open
    gaps are deferred and it's marked `skipped`.

  `Command(resume=...)` is LangGraph's mechanism for feeding a value back into
  a paused `interrupt()` — it becomes `human_input_node`'s return value, which
  flows into `integrate_answers_node`.
- **`render_gaps`** — the readable view of `state["gaps"]`. One expander per
  section (auto-expanded when it holds a blocking gap), containing one
  bordered card per gap: severity badge, status badge, category, gap id, the
  description, a nested expander with the questions asked and the answers
  received (joined via `AskedQuestion.gap_id` and
  `ContextItem.linked_gap_id`), and a caption with the
  `questions_asked/max_questions_per_gap` budget. Gaps with no `section_ids`
  land in a final "Non rattachés" group. Sorted open-first, then
  blocking → important → nice_to_have, then by gap type (contradiction first,
  via `CATEGORY_ORDER`) — matching the pool's selection order. Resolved gaps
  are hidden behind a toggle by default.
- **`render_completion`** — shown once `finished`. Renders
  `output/qa_report.md` inline, and offers download buttons for
  `cdc_final.qmd` (always present) and `cdc_final.docx` (only if the file
  exists — absent when Quarto isn't installed or rendering failed, in which
  case an info message points at the `.qmd` fallback instead).

### 5. Tab "Sous le capot"

- **Timing overview** — compute time, LLM time (with its share of compute),
  human wait time, LLM call count (flagging retries), node execution count.
- **"Où passe le temps"** — per-node aggregate table (executions, total, mean,
  max, LLM calls, share of compute as a progress bar) plus a horizontal bar
  chart, and an expander with the full per-execution chronology (start offset,
  duration, LLM seconds, error).
- **"Appels LLM"** — every call: node, output schema, duration, attempts, ok,
  prompt/response character counts, error. Repair retries are called out with
  a warning, since they're a real cost that is otherwise invisible. A nested
  expander aggregates by output schema.
- **Gaps (table brute)** — flat sortable/filterable `st.dataframe` of every
  gap field, complementing the card view in Pilotage.
- **Journal des tours** — every `TurnLogEntry` in `state["turn_log"]`, newest
  first, each expander labelled with the agent, its one-line summary, and the
  node elapsed time, and containing a `st.json` dump of the structured details.
- **État brut du checkpointer** — the scalar state fields (`turn`,
  `current_mode`, `current_section_id`, `stop_reason`, …)
  plus expanders dumping `context_items`, `asked_questions`, and
  `sections_config`.

## What's explicitly out of scope (v1)

Per `prompt.md`'s non-goals: no auth, no multi-user, no async/parallel
section processing, no database beyond Chroma. A second browser tab reusing
the same `thread_id` would observe/resume the same run, but the UI has no
concept of concurrent runs or run history beyond "the current session's
thread." Telemetry being process-global is fine for the same reason — with
two concurrent runs in one process, their timings would interleave.
