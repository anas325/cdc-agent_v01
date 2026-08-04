from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.rag import ingest_source_docs


class DummyCollection:
    def __init__(self, ids: list[str] | None = None) -> None:
        self.add_calls: list[tuple[list[str], list[str], list[dict]]] = []
        self.deleted: list[str] = []
        self._ids = list(ids or [])

    def get(self):
        return {"ids": list(self._ids)}

    def count(self) -> int:
        return len(self._ids)

    def add(self, *, ids, documents, metadatas):
        self.add_calls.append((ids, documents, metadatas))
        self._ids.extend(ids)

    def delete(self, *, ids):
        self.deleted.extend(ids)
        self._ids = [i for i in self._ids if i not in set(ids)]


@pytest.fixture
def rag_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.rag.load_settings",
        lambda: SimpleNamespace(rag=SimpleNamespace(source_dir=str(tmp_path), top_k=3, persist_dir=".chroma")),
    )


def _fake_pdf(tmp_path, monkeypatch, pages: list[str]):
    pdf_path = tmp_path / "spec.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(
        "src.rag._extract_pages",
        lambda path: [(i, text) for i, text in enumerate(pages, start=1)]
        if path.suffix.lower() == ".pdf"
        else [(None, "")],
    )
    return pdf_path


def test_ingest_source_docs_processes_pdf_files(tmp_path, monkeypatch, rag_settings):
    _fake_pdf(tmp_path, monkeypatch, ["PDF text"])
    collection = DummyCollection()
    monkeypatch.setattr("src.rag.get_collection", lambda: collection)

    added = ingest_source_docs(source_dir=tmp_path)

    assert added == 1
    assert collection.add_calls[0][1] == ["PDF text"]


def test_pdf_chunks_carry_their_page_number(tmp_path, monkeypatch, rag_settings):
    """Page provenance is what lets a RAG answer cite 'spec.pdf, page 2'."""
    _fake_pdf(tmp_path, monkeypatch, ["page one text", "page two text"])
    collection = DummyCollection()
    monkeypatch.setattr("src.rag.get_collection", lambda: collection)

    added = ingest_source_docs(source_dir=tmp_path)

    assert added == 2
    ids = [call[0][0] for call in collection.add_calls]
    metas = [call[2][0] for call in collection.add_calls]
    assert ids == ["spec.pdf::p1::0", "spec.pdf::p2::0"]
    assert [m["page"] for m in metas] == [1, 2]
    assert all(m["source"] == "spec.pdf" for m in metas)


def test_reingesting_the_same_document_adds_nothing(tmp_path, monkeypatch, rag_settings):
    _fake_pdf(tmp_path, monkeypatch, ["page one text", "page two text"])
    collection = DummyCollection()
    monkeypatch.setattr("src.rag.get_collection", lambda: collection)

    ingest_source_docs(source_dir=tmp_path)
    added_again = ingest_source_docs(source_dir=tmp_path)

    assert added_again == 0
    assert collection.deleted == []


def test_chunks_from_the_old_flat_id_scheme_are_purged(tmp_path, monkeypatch, rag_settings):
    """A .chroma built before page-aware ids must not keep the same text twice."""
    _fake_pdf(tmp_path, monkeypatch, ["page one text"])
    collection = DummyCollection(ids=["spec.pdf::0", "other.md::0"])
    monkeypatch.setattr("src.rag.get_collection", lambda: collection)

    added = ingest_source_docs(source_dir=tmp_path)

    assert added == 1
    assert collection.deleted == ["spec.pdf::0"]  # other.md is left alone
    assert "spec.pdf::p1::0" in collection.get()["ids"]
