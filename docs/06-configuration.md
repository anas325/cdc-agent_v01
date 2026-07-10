# Configuration

Two YAML files drive runtime behavior; both are loaded through
[`src/config.py`](../src/config.py) into typed Pydantic models, never parsed
ad hoc elsewhere.

## `config/settings.yaml`

```yaml
llm:
  provider: ollama          # ollama | anthropic
  model: gpt-oss:20b
  base_url: https://ollama.com   # cloud Ollama; use http://localhost:11434 for local
  temperature: 0.2

embeddings:
  provider: ollama           # ollama | sentence-transformers
  model: nomic-embed-text-v2-moe
  fallback_model: all-MiniLM-L6-v2
  base_url: http://localhost:11434   # kept local; cloud Ollama embeddings not used here

loop:
  max_turns: 15
  max_questions_per_batch: 3
  max_questions_per_gap: 2

rag:
  persist_dir: .chroma
  source_dir: data/source_docs
  top_k: 4

quarto:
  template_path: templates/cdc_template.qmd
  output_dir: output
  output_basename: cdc_final
  target_format: docx
```

Loaded by `load_settings()` into `Settings` (composed of `LLMSettings`,
`EmbeddingSettings`, `LoopSettings`, `RagSettings`, `QuartoSettings` — all
defined in `config.py` except `LoopSettings`, which lives in `state.py`
since it's also part of the graph state).

Notes on the current committed values:

- `llm.base_url` points at **cloud Ollama** (`https://ollama.com`), not a
  local server — running this as-is requires `OLLAMA_API_KEY` in
  `src/.env` (see below). Point `base_url` back at
  `http://localhost:11434` for a fully local/offline setup with a model
  pulled via `ollama pull gpt-oss:20b` (or swap `model` to whatever's
  available locally).
- To switch to Anthropic: set `llm.provider: anthropic`,
  `llm.model: claude-sonnet-5` (or another Claude model id), and export
  `ANTHROPIC_API_KEY`. No code changes — see
  [LLM & RAG](03-llm-and-rag.md#llm-factory).
- `embeddings.base_url` stays local deliberately (comment in the file
  notes cloud Ollama embeddings aren't used here) — `rag.py` falls back to
  `sentence-transformers` if that local endpoint / model is unavailable.
- `rag.top_k` and `loop.*` are also overridable per-run from the Streamlit
  sidebar without touching this file (see
  [Streamlit UI](05-streamlit-ui.md)).

## `config/sections.yaml`

Defines what the CDC must cover. Agents never hard-code section ids or
titles — everything drives off this file (loaded via `load_sections()` into
`list[SectionConfig]`).

```yaml
sections:
  - id: problem
    title: "Problématique et contexte"
    description: "..."
    required: true
    template_slot: problem_statement
    completion_hints:
      - "Problem is stated in measurable terms"
      - "..."
  # functional, technical (required: true), constraints (required: false)
```

| Field | Meaning |
|---|---|
| `id` | Stable identifier, referenced by `Gap.section_ids`, `ContextItem.section_ids`, `SectionStatus.section_id` |
| `title` | Human-readable heading, shown in the Streamlit status table |
| `description` | Given to the LLM as the section's scope, in every gap-finder / synthesizer prompt |
| `required` | Only `required: true` sections are considered by `pick_next_section` / `all_required_complete`; a non-required section (e.g. `constraints`) can stay empty without blocking synthesis |
| `template_slot` | Must match a `{{ slot }}` placeholder in `templates/cdc_template.qmd` — this is the only link between config and template, and it's not validated at load time |
| `completion_hints` | Checklist bullets the gap-finder explicitly audits the section against |

**Adding a new section** requires three coordinated edits: a new entry here,
a matching `{{ new_slot }}` placeholder plus a `#` heading in
`templates/cdc_template.qmd`, and nothing else — no code changes.

## `src/.env`

Loaded via `python-dotenv` at import time in `config.py`
(`load_dotenv(Path(__file__).parent / ".env")`), and also referenced
directly by `langgraph.json`'s `"env": "src/.env"` for the LangGraph CLI dev
server. Gitignored. Expected keys, depending on which providers are active:

```
OLLAMA_API_KEY=...      # only if llm.base_url or embeddings use cloud Ollama
ANTHROPIC_API_KEY=...   # only if llm.provider: anthropic
```

## `pyproject.toml`

uv-managed, no `requirements.txt`, no manual venv/pip commands. Key runtime
dependencies and what they're for:

| Package | Role |
|---|---|
| `langgraph` | State machine / orchestration engine (`StateGraph`, `interrupt`, `MemorySaver`) |
| `langchain-core`, `langchain-ollama`, `langchain-anthropic` | Chat model abstraction (`BaseChatModel`) behind `src/llm.py` |
| `ollama` | Python client, used indirectly via langchain-ollama / embedding healthcheck |
| `anthropic` | Anthropic SDK, pulled in transitively for the Anthropic provider path |
| `chromadb` | Persistent vector store for RAG |
| `sentence-transformers` | Local embedding fallback when Ollama embeddings are unavailable |
| `pydantic` | All state/config/agent I/O models |
| `python-dotenv`, `pyyaml` | `.env` and YAML config loading |
| `streamlit` | The UI |
| `langgraph-cli[inmem]` (dev group) | `langgraph dev` — an alternate way to run/inspect the graph outside Streamlit, using `langgraph.json` |

Install/run:

```
uv sync
uv run streamlit run src/app.py
```

## `langgraph.json`

```json
{
  "dependencies": ["."],
  "graphs": { "cdc_agent": "./src/graph.py:build_graph" },
  "env": "src/.env"
}
```

Not used by the Streamlit app itself — this is what lets `langgraph dev`
(from the `langgraph-cli` dev dependency) discover and serve `build_graph()`
as a standalone graph server, independent of the Streamlit UI, for
inspecting/debugging runs via LangGraph's own tooling. Its checkpoint
artifacts land in `.langgraph_api/` (gitignored, local-only cache).
