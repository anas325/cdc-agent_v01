# Performance Report — CDC Refinement Agent Swarm (`cdc-agent_v01`)

An inspection of the full architecture and what is hurting performance across three
axes: **compute**, **time**, and **precision** — followed by a prioritized remediation
plan.

## System at a glance

A LangGraph multi-agent pipeline, fronted by Streamlit, that iteratively refines a vague
French *cahier des charges* (CDC): it detects gaps, tries to fill them from reference
docs via RAG, asks the human targeted questions otherwise, checks for contradictions, and
synthesizes a final `.qmd`/`.docx` + QA report. Default model is Ollama cloud
`gpt-oss:20b` (`config/settings.yaml`), temperature 0. Config: 3 required sections + 1
optional, `max_turns=15`, `max_questions_per_batch=3`.

**The single structural fact driving most of this report:** every unit of real cost is a
**synchronous, sequential LLM call** through one chokepoint, `src/llm.py::call_structured`.
There is no async, no batching, and no parallelism anywhere in `src/` — confirmed zero
matches for `asyncio | ainvoke | abatch | ThreadPool | concurrent.futures | .batch(`.
Every LLM call blocks the next, and many sit inside loops.

---

## A. TIME (wall-clock — the dominant problem)

### A1. `initial_scan` runs one gap-finder LLM call **per section, serially** — `src/graph.py:149`
Before the UI shows a single question, `initial_scan_node` loops over every section and
makes an independent `run_gap_finder` call each. The `llm_cache.py:4` docstring states
this warm-up "costs 5+ minutes of LLM latency." These calls are **embarrassingly
parallel** yet run one-by-one. **Highest-leverage single fix.**

### A2. `gap_filler` fires **up to 3 LLM calls per gap**, looping the whole open pool — `src/agents/gap_filler.py:112`
Per gap, in sequence: `_grade_rag_hits` (if RAG hit) → `_draft_question` → `dedup_gate`
(`orchestrator.py:85`, yet another LLM call before the question is even shown), plus
`build_assumption` for budget-exhausted gaps. A single `gap_filler` node is ~6–9 serial
LLM calls per turn.

### A3. `synthesizer` renders **one LLM call per section, serially** — `src/agents/synthesizer.py:65`
`build_section_render_map` loops all sections, one `render_slot` call each. Independent and
parallelizable, but serial.

### A4. The JSON-repair retry loop silently multiplies wall-clock — `src/llm.py:120-141`
`call_structured` retries up to 3 attempts, re-invoking with an enlarged prompt on any
parse failure. With a small local model this fires often; each retry is a full re-invoke.
The app itself warns "coût invisible mais réel" (`app.py:628`). `retry_count` telemetry is
the metric to watch.

### A5. Whole-run LLM budget
`initial_scan` (~3–4) + up to 15 orchestrator turns × (gap_finder 1 + gap_filler ~6–9 +
critic 1) + synthesizer (~4) + final_validator (1) ≈ **100–150+ sequential LLM calls per
full run**, all on the critical path.

---

## B. COMPUTE (CPU / redundant work / I/O)

### B1. `load_settings()` / `load_sections()` re-parse YAML on **every call**, uncached — `src/config.py`
No memoization. `call_structured` alone calls `load_settings()` up to **four times per
call**. Across 100+ LLM calls that is hundreds of redundant YAML parses + Pydantic
validations per run. Cheap individually, pure waste, trivially fixable.

### B2. Prompt context is rebuilt from scratch every call, never memoized — `src/context_utils.py:50`
`format_all_sections_context` is O(sections × context_items) and is re-rendered for every
gap_finder and critic prompt; the string grows as the run grows.

### B3. RAG ingestion embeds **one chunk at a time, serially** — `src/rag.py:103`
`ingest_source_docs` calls `collection.add` inside the per-chunk loop → one embedding
round-trip per chunk. A single batched `add` would cut ingest dramatically. (Steady state
is mitigated by the `existing_ids` guard, so only new docs pay.)

### B4. RAG retrieval is per-gap and **uncached** — `src/agents/gap_filler.py:150`
`retrieve(gap.description)` runs one embed + query round-trip per gap, serial; identical
descriptions re-query from scratch. The cost is the serial round-trips, not the ANN math.

### B5. Streamlit reruns recompute everything; no `st.fragment` — `src/app.py`
- `get_state_values()` deserializes the **full** LangGraph checkpoint on every rerun
  (`app.py:218`) — grows with run length.
- The **debug tab rebuilds `telemetry.summary()` and multiple full dataframes on every
  rerun, even when it isn't the active tab** (Streamlit renders both tab bodies) —
  `app.py:735`.
- No `st.fragment`: typing in an answer box or toggling a filter recomputes the entire
  page. The answer form and gap list are prime fragment candidates.

*(The real graph execution is correctly gated behind explicit button clicks via
`graph.stream(..., stream_mode="updates")` — incidental reruns don't re-trigger LLM work.
That must not regress.)*

### B6. Real runs get **zero** LLM caching
The disk cache (`src/llm_cache.py`) exists but is gated to `CDC_LLM_CACHE=1` and described
as dev-loop-only. With temperature 0 and content-addressed keys, a scoped/TTL'd cache
could safely serve repeat prompts in real runs.

---

## C. PRECISION (output quality / correctness)

### C1. `max_tokens` / Ollama `num_ctx` are never set — `src/llm.py:33,37`
Both `ChatOllama` and `ChatAnthropic` rely on provider defaults. **Two silent failure
modes:** (a) long synthesis or the whole-document final-contradiction check gets
**output-truncated** → invalid JSON → burns retries → possibly aborts; (b) Ollama's default
`num_ctx` (often 2–4k) can silently **truncate the input** — and the gap-finder prompt
embeds the *entire document* (`gap_finder.py:87`) into every per-section call, so long CDCs
lose content with no error. **Highest-impact, invisible precision risk.**

### C2. `StructuredCallError` is uncaught in graph nodes — `src/llm.py:143`
One unparseable response after 3 attempts raises and **aborts the whole run** (only caught
by the outer UI try/except, `app.py:182`). No node-level fallback / skip.

### C3. Greedy JSON extraction regex can grab the wrong span — `src/llm.py:56`
`\{.*\}|\[.*\]` with DOTALL is greedy: prose containing braces or two JSON objects can be
mis-captured. Structured output is done by prompt instruction, **not** native
function-calling / `with_structured_output`.

### C4. Silent embedding-model fallback can corrupt RAG similarity — `src/rag.py:45`
On any Ollama embedding exception, `_embedding_function` silently falls back to a different
model (`all-MiniLM-L6-v2`) with a **different embedding space/dimensionality**. Querying a
`.chroma` collection built with one model after a fallback to the other yields meaningless
similarity. No guard checks the persisted collection's model matches. Also: **retrieval
never thresholds `distance`** (`rag.py:112`) — top-4 chunks are always returned regardless
of relevance, so precision rests entirely on the LLM `_grade_rag_hits` step.

### C5. Fixed character-based RAG chunking splits mid-sentence — `src/rag.py:20`
`chunk_size=1200 / overlap=200` on raw `text[start:end]` slices — not token- or
sentence/paragraph-aware. Overlap mitigates but doesn't eliminate boundary context loss.
PDF text quality is whatever `pypdf` yields (tables/columns often garbled).

### C6. LLM-returned `section_ids` are never validated against real sections — `src/context_utils.py:36`, `src/agents/gap_filler.py:47`
A hallucinated section id is silently ignored, making a gap effectively section-less and
**sinking its ranking** so it may never get asked.

### C7. Two ranking systems kept in sync **by hand** — `src/agents/gap_filler.py:26` vs `src/app.py:80`
Actual gap selection uses ordinal `_CATEGORY_ORDER`; the displayed section "Score" uses
weighted `CATEGORY_WEIGHTS`. Comments say they must mirror each other, but they are
separate dicts that can silently drift. (The score is cosmetic — it never feeds routing.)

### C8. Eval scoring matches **by category only**, ignoring description text — `evals/run_evals.py:137`
`score_case` counts a hit whenever categories match, ignoring content. This **overstates**
precision/recall and cannot catch content-level regressions. There is no golden-output
regression test on real LLM text; `test_graph_flow.py` uses a scripted stub.

### C9. temperature=0 for **all** tasks, including creative synthesis prose
Good for gap-finding/critic; flat/repetitive for the final document prose
(`synthesizer.render_slot`).

### C10. Anthropic path gets no JSON-mode enforcement — `src/llm.py:34-37`
Ollama gets `format="json"`; Anthropic gets only temperature and relies purely on the
prompt + retry loop. (`claude-sonnet-5` *is* a valid current model id — the concern is the
lack of structured-output enforcement, not the id.)

---

## Prioritized Remediation Plan

**Tier 1 — biggest wall-clock wins (parallelism & redundant work):**
1. Parallelize `initial_scan`'s per-section gap-finder calls (`graph.py:149`) and
   `synthesizer`'s per-section render calls (`synthesizer.py:65`) via `llm.abatch` /
   `asyncio.gather`. Add an async variant of `call_structured` (keep the sync one).
2. Memoize `load_settings()` / `load_sections()` (`src/config.py`) — removes hundreds of
   YAML re-parses per run.
3. Collapse `gap_filler`'s per-gap call chain (fold `dedup_gate` into drafting, or run the
   pool's grading/drafting concurrently) — `gap_filler.py:112`.

**Tier 2 — correctness safety nets (precision):**
4. Set explicit `max_tokens` and Ollama `num_ctx`/`num_predict`; cap/guard the size of text
   injected into prompts (`gap_finder.py:87`, `final_validator.py`) — `src/llm.py`.
5. Catch `StructuredCallError` at the node level with a graceful skip/degrade instead of
   aborting the run (`graph.py`).
6. Harden JSON extraction: prefer `with_structured_output` where supported, or replace the
   greedy regex with a balanced-brace / non-greedy parse (`llm.py:56`).
7. Stamp the embedding model name on the Chroma collection and refuse/rebuild on mismatch;
   add a `distance` threshold to `retrieve` (`rag.py`).

**Tier 3 — hygiene & measurement:**
8. Batch `collection.add` in `ingest_source_docs`; cache `retrieve` per description
   (`rag.py`, `gap_filler.py:150`).
9. Wrap the answer form / gap list in `st.fragment`; only build debug-tab dataframes when
   that tab is active (`app.py`).
10. Validate LLM-returned `section_ids` against `sections_config` (`gap_filler.py`).
11. Sharpen eval scoring beyond category-only matching; derive the UI score and selection
    order from a single shared weight table to stop drift (`app.py`, `gap_filler.py`).

---

## Verification

- **Baseline first:** run a full CDC through the app (`uv run streamlit run src/app.py`)
  with a sample from `data/` and record the "Sous le capot" tab metrics — total compute,
  LLM seconds, LLM call count, retry count, per-node timings (already exposed via
  `src/telemetry.py`). This is the before/after yardstick.
- **After Tier 1:** confirm `initial_scan` and `synthesizer` node wall-times drop roughly
  in proportion to section count; total LLM call count unchanged but elapsed time falls;
  YAML parses no longer dominate CPU.
- **After Tier 2:** run against an oversized CDC to confirm no silent truncation; force a
  malformed-JSON case to confirm graceful degradation instead of an aborted run.
- **Regression guard:** run `pytest` (14 modules) after each change; extend
  `evals/run_evals.py` with description-aware scoring and run against both providers.
- **Streamlit:** verify LLM work still only fires on explicit button clicks after adding
  fragments.
