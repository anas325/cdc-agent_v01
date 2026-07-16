# Streamlit UI

[`src/app.py`](../src/app.py) is the only interface to the system — there is
no CLI beyond ad-hoc scripts. It's intentionally "functional, testing-focused
— not polished" (per `prompt.md`): single Streamlit session, single run at a
time, no auth, no multi-user support.

Run it with:

```
uv run streamlit run src/app.py
```

## The core design constraint: no in-memory state

The graph is compiled once per Streamlit process (`get_graph()`, decorated
with `@st.cache_resource`), but **all run state lives in the LangGraph
checkpointer** (`MemorySaver`), keyed by a `thread_id` stored in
`st.session_state`. Every Streamlit rerun (which happens on every widget
interaction) re-reads state from the checkpointer via
`get_graph().get_state(config)` rather than trusting anything held in Python
globals between reruns. This is what makes closing and reopening the browser
tab mid-run recoverable — as long as `st.session_state.thread_id` survives
(it does, since it's part of the Streamlit session, not a Python global).

`current_config()` returns `{"configurable": {"thread_id": ...}}` — the
standard LangGraph checkpointer key, threaded through every `invoke`/
`get_state` call.

## Session state fields

```python
thread_id: str | None          # UUID, minted on "Démarrer" click
run_active: bool                # True while graph.invoke() is running (blocks re-click)
pending_questions: list | None  # current interrupt's question batch, or None
finished: bool                  # True once the graph reached done=True
```

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
- "Démarrer / Reprendre un nouveau run" button — disabled while
  `run_active` is true (i.e. mid-invoke) to prevent double-submission.

### 2. Starting a run — `start_run(cdc_text, loop_settings, skip_map)`

Mints a fresh `thread_id` (so every "Démarrer" click starts an unrelated new
run — this button does not resume an existing thread despite its label;
resuming happens implicitly by *not* clicking it and just reloading the
page while `thread_id` is still set), then builds an initial
`section_statuses` dict — every required section is seeded `"skipped"` if
`skip_map` marked it, otherwise `"empty"` — before invoking:

```python
input_state = {
    "initial_cdc_text": cdc_text,
    "loop_settings": loop_settings,
    "section_statuses": section_statuses,
}
result = get_graph().invoke(input_state, current_config())
```

`graph.invoke()` runs synchronously until the graph either hits an
`interrupt()` or reaches `END` — the whole first pass through however many
sections/turns happens inside this one blocking call, wrapped in
`st.spinner`.

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
`interrupt()` call in the invoke result; its `.value` is exactly the dict
passed to `interrupt({"questions": [...]})` in `human_input_node`.

### 4. Main area, every rerun

```python
values = get_state_values()      # graph.get_state(config).values, or {} if no thread yet
render_status_table(values)      # always, once a thread exists
if pending_questions: render_question_form(...)
elif finished: render_completion(...)
render_turn_log(values)          # always, appended-to across turns
```

- **`render_status_table`** — one row per configured section: status
  (emoji + label — ⚪ empty, 🟡 in_progress, 🟢 complete, 🔴 reopened,
  ⚫ skipped), a `Score` column, and open-gap counts broken down by severity
  (blocking / important / nice_to_have). Rebuilt fresh from
  `section_statuses` + `gaps` every rerun — no separate UI-side bookkeeping.
  `score_section(gaps)` starts at 100 and subtracts, per open gap,
  `SEVERITY_WEIGHTS[severity] * CATEGORY_WEIGHTS[category]` (severity ranges
  10/5/1 for blocking/important/nice_to_have; category multiplies that by
  0.8–2.0, contradiction weighted highest), floored at 0 — a quick at-a-glance
  proxy for how "risky" a section still is, purely a UI computation with no
  effect on routing.
- **`render_question_form`** — one text input + "je ne sais pas" checkbox
  per pending question, inside a single `st.form` (so all answers submit
  together, matching the batching the orchestrator already enforces). On
  submit:

  ```python
  answers = {gap_id: {"text": ..., "skip": ...}, ...}
  result = get_graph().invoke(Command(resume=answers), current_config())
  process_step_result(result)
  st.rerun()
  ```

  `Command(resume=answers)` is LangGraph's mechanism for feeding a value
  back into a paused `interrupt()` call — it becomes `human_input_node`'s
  return value, which flows into `integrate_answers_node`.
- **`render_completion`** — shown once `finished`. Renders
  `output/qa_report.md` inline (`st.markdown`), and offers download buttons
  for `cdc_final.qmd` (always present) and `cdc_final.docx` (only if the
  file exists — absent when Quarto isn't installed or rendering failed,
  in which case an info message points at the `.qmd` fallback instead).
- **`render_turn_log`** — every `TurnLogEntry` accumulated in
  `state["turn_log"]`, newest first, each in its own `st.expander` showing
  the agent name, a one-line summary, and a `st.json` dump of structured
  details when present. This is the primary debugging surface for watching
  what each agent decided on each turn without digging through server logs.

## What's explicitly out of scope (v1)

Per `prompt.md`'s non-goals: no auth, no multi-user, no async/parallel
section processing, no database beyond Chroma. A second browser tab reusing
the same `thread_id` would observe/resume the same run, but the UI has no
concept of concurrent runs or run history beyond "the current session's
thread."
