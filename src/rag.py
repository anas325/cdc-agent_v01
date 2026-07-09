"""Chroma-backed RAG ingestion and retrieval over data/source_docs/."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from src.config import RagSettings, load_settings

ROOT_DIR = Path(__file__).resolve().parent.parent
COLLECTION_NAME = "cdc_source_docs"


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


def _embedding_function(settings=None):
    settings = settings or load_settings()
    emb = settings.embeddings
    if emb.provider == "ollama":
        try:
            ef = embedding_functions.OllamaEmbeddingFunction(
                url=f"{settings.llm.base_url}/api/embeddings", model_name=emb.model
            )
            ef(["healthcheck"])  # probe: raises if server/model unavailable
            return ef
        except Exception:
            pass
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=emb.fallback_model)


@lru_cache(maxsize=1)
def get_collection():
    settings = load_settings()
    rag: RagSettings = settings.rag
    persist_path = str(ROOT_DIR / rag.persist_dir)
    client = chromadb.PersistentClient(path=persist_path)
    ef = _embedding_function(settings)
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)


def ingest_source_docs(source_dir: Path | None = None) -> int:
    """(Re)ingest every .md/.txt file under data/source_docs/. Returns count of chunks added."""
    settings = load_settings()
    source_dir = source_dir or (ROOT_DIR / settings.rag.source_dir)
    collection = get_collection()

    existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()

    added = 0
    for path in sorted(Path(source_dir).glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for i, chunk in enumerate(_chunk_text(text)):
            chunk_id = f"{path.name}::{i}"
            if chunk_id in existing_ids:
                continue
            collection.add(ids=[chunk_id], documents=[chunk], metadatas=[{"source": path.name}])
            added += 1
    return added


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """Returns list of {content, source, distance} sorted by relevance."""
    settings = load_settings()
    collection = get_collection()
    if collection.count() == 0:
        return []
    top_k = top_k or settings.rag.top_k
    results = collection.query(query_texts=[query], n_results=min(top_k, collection.count()))
    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append({"content": doc, "source": meta.get("source", "unknown"), "distance": dist})
    return hits
