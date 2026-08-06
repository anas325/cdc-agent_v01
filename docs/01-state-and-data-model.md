# State & Data Model

All models live in [`src/state.py`](../src/state.py). Everything is Pydantic
v2 (`BaseModel`), except the top-level graph state, which is a `TypedDict`
(LangGraph requirement).

## `CDCState` — the single shared state object

Every node in the graph receives the full `CDCState` and returns a partial
dict of updates that LangGraph merges in. There's no per-agent private state;
if an agent needs to know something, it's in here.

```python
class CDCState(TypedDict, total=False):
    sections_config: list[SectionConfig]       # loaded once from config/sections.yaml
    context_items: list[ContextItem]            # every accepted piece of information
    gaps: list[Gap]                              # every detected gap, at any status
    section_statuses: dict[str, SectionStatus]   # section_id -> status
    asked_questions: list[AskedQuestion]         # full history, used by the dedup gate
    current_section_id: str | None
    turn: int
    pending_user_questions: list[PendingQuestion]
    done: bool
    loop_settings: LoopSettings
    turn_log: Annotated[list[TurnLogEntry], operator.add]   # append-only, for the UI
    decision_log: Annotated[list[DecisionLogEntry], operator.add]  # append-only audit trail
    initial_cdc_text: str
    stop_reason: str | None
    current_mode: Literal["fresh", "section"]
    active_fresh_item_ids: list[str]
    _raw_answers: dict            # transient, human_input -> integrate_answers only
    _qmd_text: str                # transient, synthesizer -> final_validator only
    _mapped_items: dict[str, list[str]]  # transient, synthesizer -> final_validator only
```

`turn_log` and `decision_log` use LangGraph's reducer pattern
(`Annotated[..., operator.add]`): every node that logs appends to the list
rather than replacing it, so the Streamlit UI can render a full history across
turns. Because they live in the state, they are checkpointed — the audit trail
survives a restart and a resumed run.

Fields prefixed `_` are not really part of the durable domain state — they're
scratch space for passing a value from one specific node to the very next one
(`synthesizer` → `final_validator`, `human_input` → `integrate_answers`)
without threading it through the type system elsewhere.

## Core domain models

### `ContextItem` — an accepted piece of information

```python
class ContextItem(BaseModel):
    id: str
    content: str
    source: Literal["initial_cdc", "user_answer", "rag", "assumption"]
    section_ids: list[str]        # can belong to multiple sections
    linked_gap_id: str | None     # which gap this answers, if any
    turn_added: int
    fresh: bool = False           # True the turn it's added -> forces re-evaluation
    # --- provenance ---
    created_by: Literal["user", "rag", "llm", "system"] = "system"
    timestamp: str | None = None                  # ISO-8601 UTC
    evidence: list[Evidence] = []                 # source chunks, for RAG answers
    evidence_grade: Literal["sufficient", "partial", "insufficient"] | None = None
    confidence: float | None = None               # operational score 0–1
    validation_status: Literal[
        "unreviewed", "accepted", "rejected", "needs_review"
    ] = "unreviewed"
    model: str | None = None                      # LLM that produced it, if any
    prompt_version: str | None = None
    superseded_by: str | None = None              # id of the item that replaced this one
```

This is the unit of truth in the system. It doesn't matter whether a fact
came from the user typing an answer, a RAG hit, or an LLM-proposed default
assumption — it's a `ContextItem` either way, distinguished only by `source`.
`fresh=True` is the mechanism that drives re-evaluation: the orchestrator
checks for fresh items each turn (see
[Graph & Agents](02-graph-and-agents.md#orchestrator)), and the gap-finder /
critic use it to know which items are new this turn.

`superseded_by` is how a fact stops being true without being lost. An
assumption the system invented, later overruled by a user answer to the
contradiction it caused, is marked (`superseded_by` = the answer's id,
`validation_status="rejected"`) rather than deleted — see
[integrate_answers](02-graph-and-agents.md#integrate_answers). Everything that
builds a prompt or the final document filters through
`context_utils.live_items()`, so a superseded item is invisible to the agents;
everything that audits the run still sees it.

Every provenance field is defaulted, so checkpoints written before they existed
still deserialize. Who fills what:

| `source` | `created_by` | `confidence` | `validation_status` |
|---|---|---|---|
| `initial_cdc` | `system` | `1.0` | `accepted` (the author's own text) |
| `user_answer` | `user` | `1.0` | `accepted` (a human is authoritative) |
| `rag` | `rag` | LLM-graded | `accepted` if ≥ 0.80, else `needs_review` |
| `assumption` | `llm` | LLM-graded | **always** `needs_review` |

`confidence` is an *operational* score, not a probability: its only job is to
drive that acceptance policy (`decisions.validation_for`) and the UI's
high/medium/low band (≥ 0.80 / ≥ 0.50 / below). An LLM that answers `95` instead
of `0.95` is normalized by `decisions.clamp_confidence`.

### `Evidence` — a source chunk backing a `ContextItem`

```python
class Evidence(BaseModel):
    chunk_id: str            # Chroma id, e.g. "stock_process.pdf::p14::3"
    document: str            # file name
    page: int | None = None  # 1-based PDF page; None for md/txt
    retrieval_score: float   # 0–1, derived from Chroma's distance
    excerpt: str             # chunk text, truncated to EXCERPT_MAX_CHARS
```

Built from a `src.rag.retrieve` hit by `decisions.evidence_from_hit`. This is
what makes a RAG-sourced requirement checkable: the QA report and the gap cards
in the UI cite the exact document, page and chunk the answer rests on.

### `DecisionLogEntry` — why the system did what it did

```python
class DecisionLogEntry(BaseModel):
    id: str
    timestamp: str                # ISO-8601 UTC
    thread_id: str | None         # = run id
    turn: int
    agent: str                    # gap_finder, gap_filler, critic, ...
    decision_type: Literal[
        "gap_detected", "rag_answer", "rag_rejected", "question_drafted",
        "question_deduped", "assumption_built", "assumption_superseded",
        "answer_integrated", "contradiction_found", "section_status_changed",
        "section_synthesized", "loop_limit_applied", "final_check",
    ]
    summary: str
    input_ids: list[str]          # gaps / context items fed in
    output_ids: list[str]         # ids produced
    evidence_ids: list[str]       # Evidence.chunk_id values
    confidence: float | None
    model: str | None
    prompt_version: str | None
    details: dict
```

Where `turn_log` is a narrative for humans, this is the machine-readable trail:
enough to reconstruct which inputs produced which outputs, under which model and
prompt version. Agents *build* these (via `decisions.make_decision`) and return
them in their result models; only `graph.py` node wrappers write them into
state, the same rule that applies to gaps. `final_validator.write_decision_log`
dumps the run to `output/decision_log.jsonl`, one JSON object per line.

### `Gap` — a detected ambiguity, missing piece, or contradiction

```python
class Gap(BaseModel):
    id: str
    section_ids: list[str]        # gaps often span sections (esp. contradictions)
    category: Literal[
        "functional_ambiguity", "nfr", "data_model", "business_rule",
        "edge_case", "integration", "acceptance_criteria",
        "contradiction", "scope",
    ]
    description: str
    severity: Literal["blocking", "important", "nice_to_have"]
    status: Literal[
        "open", "rag_answered", "user_answered",
        "assumed", "deferred", "resolved",
    ] = "open"
    question_text: str | None = None   # the question actually asked, once asked
    answer_item_ids: list[str] = []
    questions_asked: int = 0
    conflicting_item_ids: list[str] = []  # contradictions only: the items in conflict
```

`section_ids` only ever holds ids from `config/sections.yaml`. Agents return
it as free-form LLM output and reliably confuse it with the `ctx_…` ids printed
throughout their prompts, so it is filtered through
`context_utils.coerce_section_ids()` before it reaches a `Gap` — misplaced
`ctx_…` ids land in `conflicting_item_ids` instead. `conflicting_item_ids` is
what lets a settled contradiction retire the item that lost, rather than being
re-detected every turn.

Status lifecycle: a gap starts `open`. It ends up `rag_answered` (RAG
sufficed), `user_answered` (a human answered), `assumed` (no answer given,
LLM proposed a default), `deferred` (loop-limit hit, non-blocking, pushed out
of scope), or `resolved` (superseded — e.g. the dedup gate found existing
context already covers it, or the gap-finder confirmed a fresh answer closes
it out). `questions_asked` caps how many times the same gap can round-trip to
the user (`loop.max_questions_per_gap`); once hit, it's auto-converted to an
assumption instead of asked again.

### `SectionConfig` / `SectionStatus`

```python
class SectionConfig(BaseModel):        # from config/sections.yaml, static per run
    id: str
    title: str
    description: str
    required: bool = True
    template_slot: str                 # maps to a {{ slot }} in cdc_template.qmd
    completion_hints: list[str] = []   # checklist the gap-finder audits against

class SectionStatus(BaseModel):        # dynamic, tracked in CDCState
    section_id: str
    status: Literal["empty", "in_progress", "complete", "reopened", "skipped"] = "empty"
    reopen_reason: str | None = None
```

A section is `complete` only when it has zero open `blocking` or `important`
gaps. It becomes `reopened` if the critic later finds that new information
contradicts it — see [critic](02-graph-and-agents.md#critic). `skipped` is set
before the run even starts, from the Streamlit sidebar's "Sections à ignorer"
checkboxes (see [Streamlit UI](05-streamlit-ui.md)) — the orchestrator treats
a skipped section as neither pickable nor eligible for
`all_required_complete()`, so it's excluded from the run without being
counted as done.

### `AskedQuestion` / `PendingQuestion`

```python
class PendingQuestion(BaseModel):   # queued for this turn's batch, not yet asked
    gap_id: str
    text: str

class AskedQuestion(BaseModel):     # permanent record, once actually sent to the user
    id: str
    gap_id: str
    text: str
    turn: int
```

`PendingQuestion` is ephemeral (built and consumed within a turn).
`AskedQuestion` accumulates in `state["asked_questions"]` for the life of the
run and is what the dedup gate checks against before asking anything new.

### `LoopSettings`

```python
class LoopSettings(BaseModel):
    max_turns: int = 15
    max_questions_per_batch: int = 3
    max_questions_per_gap: int = 2
```

Loaded from `config/settings.yaml` by default, but overridable per run from
the Streamlit sidebar (see [Streamlit UI](05-streamlit-ui.md)).

## `src/context_utils.py` — turning state into LLM prompts

Every agent that calls the LLM needs to hand it a text rendering of relevant
state. `context_utils.py` centralizes that so formatting stays consistent:

- `live_items(items)` — drops superseded items. Every renderer below goes
  through it, so a retired assumption cannot reach an agent's prompt.
- `format_context_items(items, section_id=None)` — bullet list of context
  items, tagging assumptions as `[ASSUMPTION]`.
- `format_all_sections_context(state)` — the above, grouped under a heading
  per section; used whenever an agent needs cross-section awareness (gap
  detection, contradiction checks, question drafting).
- `format_open_gaps(gaps, section_id=None)` — bullet list of open gaps, to
  avoid an agent re-raising a gap that's already tracked.
- `format_asked_questions(state)` — full question history, used by the dedup
  gate.
- `coerce_section_ids(state, raw_ids, fallback)` / `valid_section_ids(state)` /
  `sections_of_items(state, item_ids)` — the guard on LLM-returned section ids.
  Sections are config; anything an agent invents (typically a `ctx_…` id it
  copied out of its own prompt) is dropped here rather than propagating onto a
  Gap and, from there, onto the answer built from it.
- `get_section(state, section_id)` — lookup helper, raises `KeyError` if the
  id isn't in `sections_config`.

## `src/utils/text_match.py` — "did we already ask this?"

Accent-folded, stopword-free, entity-id-stripped content-word comparison, by
**containment** rather than Jaccard (a repeat is usually a narrowed restatement,
so the shorter side is nearly a subset of the longer one). Used by the gap
filler's repeat check; deliberately independent of `evals/simulator.py`'s
`keyword_score`, which serves the eval oracle and must not be imported by `src`.

## `src/decisions.py` — building audit records

Small, dependency-light helpers shared by every agent:

- `make_decision(state, *, agent, decision_type, summary, **kw)` — mints a
  `DecisionLogEntry`, stamping the turn from state, the thread id from
  telemetry, an ISO timestamp, and the prompt version for a given `prompt_id`.
- `evidence_from_hit(hit)` / `format_evidence(ev)` — retrieve hit → `Evidence`,
  and `Evidence` → a `"doc.pdf, p. 14"` citation.
- `confidence_band(c)` / `validation_for(c)` / `clamp_confidence(c)` — the
  confidence policy, defined once so the UI badges and the acceptance rule can
  never drift apart.

## `src/ids.py`

```python
def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"
```

Used everywhere ids are minted (`ctx_...`, `gap_...`, `q_...`) — short,
prefixed, collision-safe enough for a single-run prototype.
