# Graph & Agents

The orchestration backbone is [`src/graph.py`](../src/graph.py): a LangGraph
`StateGraph(CDCState)` with nine nodes. Node functions in `graph.py` are thin
wrappers — they call into `src/agents/*.py` for actual logic, then translate
the result into a state-update dict and a `TurnLogEntry`. This split keeps
agent logic unit-testable without spinning up the graph.

## Graph topology

```
START -> ingest -> orchestrator -> (route_after_orchestrator)
    -> gap_finder -> gap_filler -> (route_after_gap_filler)
         -> human_input -> integrate_answers -> critic -> orchestrator
         -> critic -> orchestrator
    -> synthesizer -> final_validator -> END
    -> END  (blocking gaps hit max_turns)
```

Compiled with `MemorySaver()` as checkpointer — state is durable per
`thread_id`, which is what makes the Streamlit app's interrupt/resume cycle
(and surviving a tab close) work. See [`build_graph()`](../src/graph.py).

### `route_after_orchestrator`

```python
if state.done: -> END
elif pending_user_questions: -> human_input      # leftover batch from a prior turn
elif current_mode == "fresh": -> gap_finder       # re-evaluate fresh items first
elif current_section_id is None: -> synthesizer   # nothing left to work on
else: -> gap_finder
```

### `route_after_gap_filler`

```python
pending_user_questions ? -> human_input : critic
```

## Node-by-node

### `ingest`

Runs once at the start of a run. Loads `sections.yaml` and `settings.yaml`
(via `src/config.py`), calls `rag.ingest_source_docs()` to (re)index
`data/source_docs/` into Chroma, seeds `context_items` with the user-supplied
initial CDC text (if any, tagged `source="initial_cdc"`, attached to every
section since it's unstructured input), and initializes every section's
status to `empty` — unless the caller already passed in a `section_statuses`
entry for it (e.g. `status="skipped"`, set by the Streamlit sidebar before
`invoke()`), in which case the incoming status is kept as-is rather than
overwritten.

### `orchestrator` (agent: [`src/agents/orchestrator.py`](../src/agents/orchestrator.py))

Runs every turn; it's the only node with real branching logic. Priority
order each turn:

1. **Leftover pending questions** from a batch cap — if the previous turn's
   `gap_filler` queued more questions than `max_questions_per_batch` allowed,
   they were carried over rather than dropped; send those immediately (no
   re-planning).
2. **Fresh items** — if any `context_items` have `fresh=True` (just-added RAG
   answers, user answers, or assumptions), clear their `fresh` flag and route
   to `gap_finder` in `"fresh"` mode to check whether they actually close the
   gaps they're linked to, or introduce new contradictions. This pre-empts
   normal section work.
3. **Loop limits** (see below).
4. **Pick next section** — `pick_next_section()`: scans required sections in
   priority order `reopened > in_progress > empty` (config order within each
   bucket), skipping any section whose status is `skipped`. Returns `None`
   once everything required is `complete` (or `skipped`), which routes to
   `synthesizer`.

Note the asymmetry with `all_required_complete()` (a standalone helper, not
currently wired into `route_after_orchestrator` — routing to `synthesizer`
actually happens via `pick_next_section()` returning `None`): it treats
`skipped` the same as `empty` and returns `False` if any required section has
either status. So `pick_next_section` alone is what lets a run with skipped
sections reach synthesis.

#### Loop limits

`apply_loop_limits()` fires once `turn >= max_turns`:

- If any **open, blocking** gap remains → stop the whole run
  (`done=True`, `stop_reason` set). A blocking gap is never silently
  dropped or downgraded.
- Otherwise, every open (non-blocking) gap is downgraded to `deferred` and
  the run proceeds straight to synthesis with whatever's been gathered.

#### Question dedup gate

`dedup_gate(state, gap, candidate_question)` — one LLM call, invoked from
`gap_filler_node` (not from the orchestrator agent module directly, despite
living in `orchestrator.py`) for every question about to be queued. It's
given the full cross-section context and the entire question-history log,
and must decide:

- `already_resolved=True` → existing context already fully answers this;
  don't ask, mark the gap `resolved` using the existing item.
- `partially_resolved=True` → context covers part of it; rewrite the
  question to cover only the missing part.
- neither → ask the candidate question as drafted.

This runs *before* a question is added to a batch, so RAG answers and prior
user answers never produce a redundant prompt to the user.

#### Question batching

`batch_questions(pending, max_batch)` splits the queue into
`(this_turn_batch, remainder)`. Only `this_turn_batch` is sent as ONE
interrupt; `remainder` is carried into `pending_user_questions` for the
following turn (see priority #1 above) — the user is never shown more than
`max_questions_per_batch` questions in a single Streamlit form.

### `gap_finder` (agent: [`src/agents/gap_finder.py`](../src/agents/gap_finder.py))

One LLM call (`call_structured`) per invocation, run in one of two modes:

- **`section` mode** — focused on `state.current_section_id`. Prompted with
  the full cross-section context (contradictions can be transversal), the
  section's already-open gaps (so it doesn't duplicate them), and the
  section's `completion_hints` as an explicit checklist. Instructed to walk
  the fixed gap taxonomy category by category —
  `functional_ambiguity, nfr, data_model, business_rule, edge_case,
  integration, acceptance_criteria, contradiction, scope` — rather than free
  associate. Also returns a `section_complete` verdict: true only if the
  section has zero open blocking/important gaps after this pass.
- **`fresh` mode** — focused on a set of just-added `context_items`
  (`active_fresh_item_ids`). For each, judges whether it actually resolves
  the gap it's linked to (`resolved_gap_ids`) or needs a follow-up gap
  (`follow_up_of_gap_id` set, for a partial/ambiguous answer). Also checks
  whether an `ASSUMPTION:` item contradicts an already-`complete` section.

`GapFinderResult.new_gaps` are minted with fresh ids via `new_id("gap")`;
resolved ids are applied by the `gap_finder_node` wrapper in `graph.py`,
which also updates `section_statuses` when in `section` mode.

### `gap_filler` (agent: [`src/agents/gap_filler.py`](../src/agents/gap_filler.py))

For every currently-open gap targeted this turn (`active_gap_ids`, sorted
`blocking` → `important` → `nice_to_have`):

1. **RAG first**: `rag.retrieve(gap.description)` against the Chroma
   collection. If there are hits, one LLM call (`_grade_rag_hits`) judges
   whether they're *sufficient and unambiguous* — not just topically
   related. If sufficient, a new `ContextItem(source="rag", fresh=True)` is
   created and the gap moves to `rag_answered`.
2. **Otherwise, draft a question**: `_draft_question()` — one LLM call
   instructed to produce a question that quotes or paraphrases the specific
   ambiguous text (never a generic "can you clarify scope?"), naming both
   sides explicitly if it's a contradiction.

Back in `graph.py`'s `gap_filler_node`, each drafted question then goes
through the orchestrator's dedup gate (`orch.dedup_gate`) before being
queued. If a gap has already hit `max_questions_per_gap` rounds, instead of
asking again it calls `build_assumption()` — one more LLM call that proposes
a pragmatic, industry-standard default, prefixed `ASSUMPTION:`, and marks the
gap `assumed`. Assumptions built this way, or built from an explicit user
"I don't know" (see `integrate_answers` below), always carry
`source="assumption"` so the synthesizer can flag them visibly.

### `human_input`

```python
answers = interrupt({"questions": [{"gap_id": ..., "text": ...}, ...]})
return {"_raw_answers": answers}
```

A LangGraph `interrupt()` — execution pauses here, the checkpointer persists
state, and the Streamlit app renders the batch as a form. Resuming happens
via `graph.invoke(Command(resume=answers), config)`. See
[Streamlit UI](05-streamlit-ui.md).

### `integrate_answers`

Pure state transformation, no LLM call. For each queued question: records it
into `asked_questions` (permanent log), and either:

- the user skipped or left it blank → `gap_filler_agent.build_assumption()`
  is called to synthesize a default, gap → `assumed`; or
- the user answered → a `ContextItem(source="user_answer", fresh=True,
  linked_gap_id=gap.id)` is created, gap → `user_answered` with the new item
  id appended to `answer_item_ids`.

Kept as a separate node from `critic` deliberately (see `prompt.md`) so
"turning raw answers into context" and "checking those answers for
contradictions" stay independently testable.

### `critic` (agent: [`src/agents/critic.py`](../src/agents/critic.py))

Runs **globally** after every batch of new context is integrated — never
scoped to one section, because a new answer can contradict any other part of
the document. Given the fresh items (`active_fresh_item_ids`) plus the full
cross-section context and the list of sections currently marked `complete`,
one LLM call looks specifically for:

1. a fresh item contradicting an earlier context item (any section), or
2. a fresh item contradicting a section already marked `complete`.

Each finding becomes a `Gap(category="contradiction", ...)`; if it touches a
section that was `complete`, that section is flipped to `reopened` with
`reopen_reason` set to the finding's description — which routes the
orchestrator back to that section on a later turn. The critic never grades
its own findings; it only inspects state other agents produced.

### `synthesizer` / `final_validator`

Covered in depth in [Document Generation](04-document-generation.md). Short
version: `synthesizer` renders every section's accepted context into
professional French prose, fills `templates/cdc_template.qmd`, writes
`output/cdc_final.qmd`, and shells out to `quarto render` for DOCX (non-fatal
if Quarto isn't installed — the `.qmd` still gets written). `final_validator`
does one more LLM-based full-document contradiction sweep, lists any context
item that never made it into a slot, and writes `output/qa_report.md`, then
sets `done=True`.

## Where LLM calls happen (summary)

| Agent | LLM calls | Purpose |
|---|---|---|
| orchestrator | `dedup_gate` | Is this question already answered? |
| gap_finder | 1 per invocation | Find gaps/contradictions, section-complete verdict |
| gap_filler | `_grade_rag_hits`, `_draft_question`, `build_assumption` | Grade RAG sufficiency, draft targeted questions, propose defaults |
| critic | 1 per invocation (skipped if no fresh items) | Find contradictions in newly integrated context |
| synthesizer | 1 per section slot | Render section prose |
| final_validator | `find_final_contradictions` | Whole-document contradiction sweep |

All go through `call_structured()` — see [LLM & RAG](03-llm-and-rag.md).
