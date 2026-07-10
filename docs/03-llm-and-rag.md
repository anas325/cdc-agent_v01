# LLM & RAG

## LLM factory — [`src/llm.py`](../src/llm.py)

Every agent gets a chat model through this module; no agent imports a
provider-specific class directly, so switching providers is a config-only
change (`config/settings.yaml` → `llm.provider`).

```python
def _build_llm(cfg: LLMSettings, json_mode: bool = False) -> BaseChatModel:
    if cfg.provider == "ollama":
        return ChatOllama(model=cfg.model, base_url=cfg.base_url,
                           temperature=cfg.temperature,
                           **({"format": "json"} if json_mode else {}))
    if cfg.provider == "anthropic":
        return ChatAnthropic(model=cfg.model, temperature=cfg.temperature)
    raise ValueError(...)
```

- `get_llm()` — cached (`lru_cache`), general-purpose chat model per current
  settings.
- `_get_structured_llm()` — cached, used only by `call_structured`. For
  Ollama it's built with `format: "json"` (Ollama's native JSON mode);
  for other providers it's the same as `get_llm()`.
- Ollama picks up `OLLAMA_API_KEY` from the environment (for
  [Ollama's cloud offering](https://ollama.com), used as the default
  `base_url` in `settings.yaml`) and sends it as a Bearer token.

Caching via `lru_cache(maxsize=1)` means the model is built once per process
and settings changes require a process restart to take effect — acceptable
for this single-session prototype.

### `call_structured()` — provider-agnostic structured output

```python
def call_structured(prompt: str, model: type[T], llm=None, max_retries: int = 2) -> T
```

This is the one function every agent uses to get typed data back from the
LLM. Rationale (from the module docstring): small local Ollama models rarely
support reliable native function-calling, so structured output is obtained
by:

1. Appending the target Pydantic model's JSON schema to the prompt with an
   explicit "respond with ONLY a JSON object matching this schema" instruction.
2. Using Ollama's native JSON mode where available (`_get_structured_llm`).
3. Repairing common formatting issues before parsing — stripping ` ```json `
   code fences, extracting the first `{...}`/`[...]` block from surrounding
   prose via regex (`_extract_json`).
4. On parse failure, retrying (`max_retries`, default 2) with the parse
   error and the previous raw response fed back into the prompt, asking for
   a corrected JSON object.
5. Raising `StructuredCallError` if all attempts fail.

Every agent module defines small Pydantic "wire" models for what it expects
back (e.g. `DedupVerdict`, `GapFinderOutput`, `RagGrade`, `CriticOutput`) and
passes them straight to `call_structured`.

## RAG — [`src/rag.py`](../src/rag.py)

Chroma-backed, persisted locally, indexing `data/source_docs/`.

### Chunking

`_chunk_text(text, chunk_size=1200, overlap=200)` — naive fixed-size
character chunking with overlap; no sentence/paragraph awareness. Adequate
for the short reference docs this prototype targets.

### Embeddings

`_embedding_function()` tries Ollama first (`OllamaEmbeddingFunction`,
default model `nomic-embed-text-v2-moe`), probing it with a one-off
`ef(["healthcheck"])` call. If that raises for any reason (server down,
model not pulled), it silently falls back to a local
`SentenceTransformerEmbeddingFunction` (`all-MiniLM-L6-v2` by default) —
no network dependency in the fallback path.

### Collection lifecycle

`get_collection()` — `lru_cache(maxsize=1)`, opens (or creates)
`chromadb.PersistentClient(path=".chroma")`'s `cdc_source_docs` collection
with the configured embedding function. Cached for process lifetime, same
caveat as the LLM factory re: settings changes needing a restart.

### Ingestion

`ingest_source_docs(source_dir=None)` — walks `data/source_docs/*.md` and
`*.txt` (default source dir, from `settings.rag.source_dir`), chunks each
file, and adds chunks whose id (`"{filename}::{chunk_index}"`) isn't already
in the collection. This makes ingestion idempotent/incremental: re-running it
(as `ingest_node` does at the start of every graph run) only adds genuinely
new files/chunks, it doesn't re-embed everything. There's no mechanism to
detect *changed* content in an existing file under the same name+chunk-index
— editing a source doc without renaming it won't re-index the edited chunk.

### Retrieval

`retrieve(query, top_k=None)` — semantic query against the collection
(`top_k` from `settings.rag.top_k`, default 4), returns
`[{content, source, distance}, ...]` sorted by relevance. Returns `[]`
immediately if the collection is empty (no source docs ingested yet), rather
than erroring. Called by `gap_filler._grade_rag_hits` flow, keyed on
`gap.description` as the query text.

## Provider swap reference

| Setting | Ollama (default) | Anthropic |
|---|---|---|
| `llm.provider` | `ollama` | `anthropic` |
| `llm.model` | e.g. `gpt-oss:20b` | e.g. `claude-sonnet-5` |
| `llm.base_url` | `http://localhost:11434` or `https://ollama.com` (cloud) | unused |
| env var needed | `OLLAMA_API_KEY` (cloud only) | `ANTHROPIC_API_KEY` |

Embeddings are configured independently (`embeddings.provider`) and are
**not** swapped to Anthropic — Anthropic has no embeddings API; the fallback
is always `sentence-transformers`.
