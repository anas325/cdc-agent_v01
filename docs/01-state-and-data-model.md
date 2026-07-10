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
    initial_cdc_text: str
    stop_reason: str | None
    current_mode: Literal["fresh", "section"]
    active_fresh_item_ids: list[str]
    active_gap_ids: list[str]
    _raw_answers: dict            # transient, human_input -> integrate_answers only
    _qmd_text: str                # transient, synthesizer -> final_validator only
    _mapped_items: dict[str, list[str]]  # transient, synthesizer -> final_validator only
```

`turn_log` uses LangGraph's reducer pattern (`Annotated[..., operator.add]`):
every node that logs appends to the list rather than replacing it, so the
Streamlit UI can render a full history across turns.

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
```

This is the unit of truth in the system. It doesn't matter whether a fact
came from the user typing an answer, a RAG hit, or an LLM-proposed default
assumption — it's a `ContextItem` either way, distinguished only by `source`.
`fresh=True` is the mechanism that drives re-evaluation: the orchestrator
checks for fresh items before anything else each turn (see
[Graph & Agents](02-graph-and-agents.md#orchestrator)), and the gap-finder /
critic use it to know which items are new this turn.

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
    question_text: str | None = None
    answer_item_ids: list[str] = []
    questions_asked: int = 0
```

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
    status: Literal["empty", "in_progress", "complete", "reopened"] = "empty"
    reopen_reason: str | None = None
```

A section is `complete` only when it has zero open `blocking` or `important`
gaps. It becomes `reopened` if the critic later finds that new information
contradicts it — see [critic](02-graph-and-agents.md#critic).

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

- `format_context_items(items, section_id=None)` — bullet list of context
  items, tagging assumptions as `[ASSUMPTION]`.
- `format_all_sections_context(state)` — the above, grouped under a heading
  per section; used whenever an agent needs cross-section awareness (gap
  detection, contradiction checks, question drafting).
- `format_open_gaps(gaps, section_id=None)` — bullet list of open gaps, to
  avoid an agent re-raising a gap that's already tracked.
- `format_asked_questions(state)` — full question history, used by the dedup
  gate.
- `get_section(state, section_id)` — lookup helper, raises `KeyError` if the
  id isn't in `sections_config`.

## `src/ids.py`

```python
def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"
```

Used everywhere ids are minted (`ctx_...`, `gap_...`, `q_...`) — short,
prefixed, collision-safe enough for a single-run prototype.
