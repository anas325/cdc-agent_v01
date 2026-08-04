"""Chroma-backed RAG ingestion and retrieval over data/source_docs/."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}

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
                url=f"{emb.base_url}/api/embeddings", model_name=emb.model
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


def _extract_pages(path: Path) -> list[tuple[int | None, str]]:
    """Extract a document as (page_number, text) pairs.

    PDFs yield one pair per page (1-based) so the page number survives into the
    chunk id and metadata, and a RAG answer can cite "doc.pdf, page 14". Flat
    formats have no pagination and yield a single (None, text) pair.
    """
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return [(None, path.read_text(encoding="utf-8", errors="ignore"))]

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing
            raise RuntimeError("pypdf is required to ingest PDF files") from exc

        reader = PdfReader(str(path))
        return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]

    raise ValueError(f"Unsupported document format: {path.suffix}")


def extract_document_text(path: Path) -> str:
    """Whole-document plain text, pages joined — for callers that don't index.

    The Streamlit sidebar uses this to read an uploaded PDF CDC, where page
    boundaries carry no meaning (the text goes straight to the section splitter).
    Ingestion uses `_extract_pages` instead, to keep page provenance.
    """
    return "\n\n".join(text for _, text in _extract_pages(path) if text.strip()).strip()


def _chunk_id(document: str, page: int | None, index: int) -> str:
    return f"{document}::p{page}::{index}" if page is not None else f"{document}::{index}"


def ingest_source_docs(source_dir: Path | None = None) -> int:
    """(Re)ingest every supported document under data/source_docs/. Returns count of chunks added.

    Chunking happens *within* a page, never across pages, so every chunk maps to
    exactly one page number.
    """
    settings = load_settings()
    source_dir = source_dir or (ROOT_DIR / settings.rag.source_dir)
    collection = get_collection()

    existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()

    added = 0
    for path in sorted(Path(source_dir).glob("*")):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            pages = _extract_pages(path)
        except (RuntimeError, ValueError):
            continue

        if not any(text.strip() for _, text in pages):
            continue

        chunks = [
            (page, i, chunk)
            for page, text in pages
            if text.strip()
            for i, chunk in enumerate(_chunk_text(text))
        ]
        wanted_ids = {_chunk_id(path.name, page, i) for page, i, _ in chunks}
        existing_ids -= _purge_stale_chunks(collection, path.name, existing_ids, wanted_ids)

        for page, i, chunk in chunks:
            chunk_id = _chunk_id(path.name, page, i)
            if chunk_id in existing_ids:
                continue
            metadata: dict = {"source": path.name, "file_type": path.suffix.lower()}
            if page is not None:
                metadata["page"] = page
            collection.add(ids=[chunk_id], documents=[chunk], metadatas=[metadata])
            added += 1
    return added


def _purge_stale_chunks(
    collection, document: str, existing_ids: set[str], wanted_ids: set[str]
) -> set[str]:
    """Drop chunks of `document` that the current chunking scheme no longer produces.

    Page-aware chunk ids (`doc.pdf::p14::3`) superseded the flat `doc.pdf::3`
    scheme, so a collection built before that change holds ids that would never
    be overwritten — leaving the same text indexed twice under two id schemes.
    Also covers an edited document that now yields fewer chunks. Returns the ids
    actually removed.
    """
    prefix = f"{document}::"
    stale = {cid for cid in existing_ids if cid.startswith(prefix)} - wanted_ids
    if stale:
        collection.delete(ids=sorted(stale))
    return stale


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """Returns list of {content, chunk_id, document, source, page, distance, score}.

    `score` is a 0–1 relevance score derived from Chroma's (unbounded) L2
    distance via 1/(1+d), so it can be shown and thresholded directly.
    `source` is kept as an alias of `document` for backwards compatibility.
    """
    settings = load_settings()
    collection = get_collection()
    if collection.count() == 0:
        return []
    top_k = top_k or settings.rag.top_k
    results = collection.query(query_texts=[query], n_results=min(top_k, collection.count()))
    hits = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
        meta = meta or {}
        document = meta.get("source", "unknown")
        hits.append(
            {
                "content": doc,
                "chunk_id": chunk_id,
                "document": document,
                "source": document,
                "page": meta.get("page"),
                "distance": dist,
                "score": 1.0 / (1.0 + dist) if dist is not None else 0.0,
            }
        )
    return hits
