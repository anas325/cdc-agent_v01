# CDC Refinement Agent Swarm

A multi-agent system that iteratively refines a vague "cahier des charges"
(CDC / requirements spec) by detecting gaps and ambiguities, resolving them
via RAG when possible, and asking the author targeted questions otherwise —
until the document is complete enough to hand to a dev team. Ends by
generating a final DOCX (via Quarto) plus a QA report.

See [`docs/00-overview.md`](docs/00-overview.md) for full architecture docs.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```
uv sync
```

### LLM & embeddings (Ollama)

By default the app talks to Ollama for both the chat model and embeddings.

- **Local Ollama**: install [Ollama](https://ollama.com/download), then pull
  the models referenced in `config/settings.yaml`:
  ```
  ollama pull gpt-oss:20b
  ollama pull nomic-embed-text-v2-moe
  ```
  and set `llm.base_url: http://localhost:11434` in `config/settings.yaml`.
- **Cloud Ollama** (the committed default, `llm.base_url: https://ollama.com`):
  create an API key at https://ollama.com/settings/keys and set it in
  `src/.env`:
  ```
  OLLAMA_API_KEY=...
  ```

To use Anthropic instead, set `llm.provider: anthropic` and
`llm.model: claude-sonnet-5` (or another Claude model) in
`config/settings.yaml`, and set `ANTHROPIC_API_KEY` in `src/.env`. No code
changes needed — see [`docs/03-llm-and-rag.md`](docs/03-llm-and-rag.md).

### Quarto (optional, for DOCX output)

Install [Quarto](https://quarto.org/docs/get-started/) to render the final
document to `.docx`. If it's not installed, the app still writes
`output/cdc_final.qmd` and shows a warning instead of a `.docx` download.

## Run

```
uv run streamlit run src/app.py
```

Upload an initial CDC (optional) and/or reference documents for RAG in the
sidebar, click "Démarrer", and answer questions as they come up. Download
the final `.qmd`/`.docx` and QA report once the run completes.

## Project structure

```
config/            # sections.yaml, settings.yaml
data/source_docs/  # reference docs ingested into the RAG vector store
templates/         # Quarto template for the final document
src/
  graph.py         # LangGraph orchestration
  agents/          # orchestrator, gap_finder, gap_filler, critic, synthesizer, final_validator
  app.py           # Streamlit UI (entrypoint)
output/            # generated cdc_final.qmd / .docx / qa_report.md
docs/              # architecture documentation
```

Full breakdown in [`docs/00-overview.md`](docs/00-overview.md#source-map).
