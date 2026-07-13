from __future__ import annotations

from types import SimpleNamespace

from src.rag import ingest_source_docs


class DummyCollection:
    def __init__(self) -> None:
        self.add_calls: list[tuple[list[str], list[str], list[dict]]] = []

    def get(self):
        return {"ids": []}

    def count(self) -> int:
        return 0

    def add(self, *, ids, documents, metadatas):
        self.add_calls.append((ids, documents, metadatas))


def test_ingest_source_docs_processes_pdf_files(tmp_path, monkeypatch):
    pdf_path = tmp_path / "spec.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    collection = DummyCollection()
    monkeypatch.setattr("src.rag.get_collection", lambda: collection)
    monkeypatch.setattr(
        "src.rag.load_settings",
        lambda: SimpleNamespace(rag=SimpleNamespace(source_dir=str(tmp_path), top_k=3, persist_dir=".chroma")),
    )
    monkeypatch.setattr("src.rag._extract_text_from_path", lambda path: "PDF text" if path.suffix.lower() == ".pdf" else "")

    added = ingest_source_docs(source_dir=tmp_path)

    assert added == 1
    assert collection.add_calls[0][1] == ["PDF text"]
