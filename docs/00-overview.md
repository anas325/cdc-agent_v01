# Architecture Overview

## What this is

A multi-agent system that turns a vague, incomplete "cahier des charges" (CDC —
a French-style requirements specification) into a document precise enough to
hand to a development team. It does this by repeatedly:

1. detecting gaps and ambiguities in the current draft,
2. trying to fill them automatically from reference documents (RAG),
3. asking the human author targeted questions when RAG can't answer,
4. checking newly added information for contradictions with what's already
   been accepted,
5. and, once every required section is complete, synthesizing a final Word
   document plus a QA report.

The whole loop is a **LangGraph state machine**: one shared, typed state
object flows through a fixed set of agent nodes, with a human-in-the-loop
interrupt for question/answer turns. A Streamlit app is the only interface —
there is no CLI.

This is a prototype (v1) built from the spec in [`prompt.md`](../prompt.md) at
the repo root, which is a useful secondary reference for intent/rationale.

## Why it's built this way

- **Sections are config, not code.** `config/sections.yaml` defines what a
  CDC must cover (problem statement, functional spec, technical spec,
  constraints, …). Agents never hard-code section names — see
  [Configuration](06-configuration.md).
- **One shared state, many small agents.** Each agent is a pure function
  `(CDCState, ...) -> result`; only `graph.py`'s node wrappers mutate state.
  This keeps agent logic testable in isolation and keeps the control flow
  (who runs next, and why) in exactly one place. See
  [Graph & Agents](02-graph-and-agents.md).
- **LLM calls always return typed data.** There's a single `call_structured()`
  helper that every agent uses to get a Pydantic model back from the LLM,
  regardless of provider (Ollama today, Anthropic is a config flip). See
  [LLM & RAG](03-llm-and-rag.md).
- **Human answers and RAG answers are the same kind of thing.** Both become a
  `ContextItem` tagged with a `source` (`rag`, `user_answer`, `assumption`,
  `initial_cdc`). Downstream agents (critic, synthesizer) don't care where an
  answer came from, only whether it's trustworthy and where it belongs. See
  [State & Data Model](01-state-and-data-model.md).
- **Nothing is silently dropped.** Assumptions made when the user doesn't
  know an answer are visually flagged in the final document. Context that
  never made it into any section, and contradictions found on a final
  full-document read, are listed in `output/qa_report.md`. See
  [Document Generation](04-document-generation.md).

## High-level flow

```
 Streamlit UI (src/app.py)
        │  uploads initial CDC + source docs, starts/resumes a run
        ▼
 LangGraph (src/graph.py)              persisted in Supabase Postgres (PostgresSaver, keyed by thread_id)
        │
        ├─ ingest ───────────────────── load sections.yaml, ingest data/source_docs/ into Chroma
        │
        ├─ orchestrator ─┐             picks: re-evaluate fresh info? next section? done?
        │                │
        │       ┌────────┴─────────┐
        │       ▼                  ▼
        │   gap_finder         synthesizer ──▶ final_validator ──▶ END
        │       │                                  (writes cdc_final.qmd/.docx + qa_report.md)
        │       ▼
        │   gap_filler ── tries RAG first, else drafts a question
        │       │
        │  pending questions? ──no──▶ critic ──▶ orchestrator (loop)
        │       │yes
        │       ▼
        │   human_input (interrupt) ──▶ integrate_answers ──▶ critic ──▶ orchestrator (loop)
        ▼
 (user answers in Streamlit, graph resumes with Command(resume=...))
```

The loop repeats until either every required section is `complete`, or a
configured `max_turns` is hit (with different behavior for blocking vs.
non-blocking open gaps — see [Graph & Agents](02-graph-and-agents.md#loop-limits)).

## Where to go next

| Doc | Covers |
|---|---|
| [01 — State & Data Model](01-state-and-data-model.md) | `CDCState`, `ContextItem`, `Gap`, `SectionStatus`, and how they relate |
| [02 — Graph & Agents](02-graph-and-agents.md) | Every LangGraph node, each agent's responsibility, routing/loop-limit logic |
| [03 — LLM & RAG](03-llm-and-rag.md) | Provider-agnostic LLM factory, structured-output calling, Chroma ingestion/retrieval |
| [04 — Document Generation](04-document-generation.md) | Quarto template filling, DOCX rendering, QA report |
| [05 — Streamlit UI](05-streamlit-ui.md) | `src/app.py` — session state, rendering, resuming interrupted runs |
| [06 — Configuration](06-configuration.md) | `config/settings.yaml`, `config/sections.yaml`, `.env`, `pyproject.toml` dependencies |
| [07 — Évaluation](07-evaluation.md) | Benchmark annoté, partie prenante synthétique, runner de lot et artefacts de run |

## Source map

```
main.py                        # unused stub entrypoint (uv template default) — real entrypoint is src/app.py
langgraph.json                 # exposes build_graph() to the LangGraph CLI/dev server
prompt.md                      # original build spec for this prototype
config/
  sections.yaml                # CDC section definitions (id, title, template_slot, completion_hints)
  settings.yaml                # llm/embeddings/rag/loop/quarto settings
data/
  source_docs/                 # reference docs ingested into the RAG vector store
templates/
  cdc_template.qmd             # Quarto template with {{ slot }} placeholders
src/
  state.py                     # Pydantic models + CDCState TypedDict
  config.py                    # loads settings.yaml / sections.yaml into typed models
  ids.py                       # new_id() — short uuid-based ids
  context_utils.py             # renders state into LLM-readable text blocks
  llm.py                       # get_llm() factory + call_structured()
  rag.py                       # Chroma ingestion + retrieval
  db.py                        # Supabase Postgres: checkpointer + per-user runs index
  graph.py                     # StateGraph wiring — the orchestration backbone
  app.py                       # Streamlit UI, the actual entrypoint (auth + run history)
  agents/
    orchestrator.py            # section picking, loop limits, question dedup gate
    gap_finder.py               # detects gaps/ambiguities/contradictions per section or fresh item
    gap_filler.py               # RAG-first gap resolution, question drafting, assumptions
    critic.py                   # global contradiction check after every answer integration
    synthesizer.py              # renders final .qmd/.docx from accepted context
    final_validator.py          # last QA pass, writes qa_report.md
evals/                         # offline evaluation — see docs/07-evaluation.md
  run_evals.py                 # component evals: calls gap_finder/critic directly, no graph
  run_benchmark.py             # batch runner: drives the compiled graph over the benchmark
  simulator.py                 # synthetic stakeholders answering the interrupt() batches
  dataset.py                   # benchmark schema + validating loader
  harness.py                   # provider forcing, per-case settings isolation, run manifest
  datasets/
    gap_finder.jsonl           # component eval cases
    critic.jsonl
    benchmark/                 # 10 annotated CDC cases (ground truth — committed)
  results/                     # per-run artifacts (generated, gitignored)
output/                        # cdc_final.qmd, cdc_final.docx, qa_report.md (generated)
.chroma/                       # persistent Chroma vector store (generated)
```
