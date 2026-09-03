from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from app.bootstrap import collect_indexed_documents


class FakeDocstore:
    def __init__(self, documents):
        self._documents = documents

    def search(self, document_id):
        return self._documents.get(document_id)


class FakeVectorstore:
    def __init__(self, documents):
        self.index_to_docstore_id = {
            position: document_id
            for position, document_id in enumerate(documents)
        }
        self.docstore = FakeDocstore(documents)
        self.index = SimpleNamespace(ntotal=len(documents))


def make_document(chunk_id: str) -> Document:
    return Document(
        page_content=f"Conteúdo do chunk {chunk_id}",
        metadata={"chunk_id": chunk_id},
    )


def test_collect_indexed_documents_uses_deterministic_order():
    indexes = {
        "txt": FakeVectorstore({"txt-1": make_document("txt-1")}),
        "csv": FakeVectorstore({"csv-1": make_document("csv-1")}),
    }

    documents = collect_indexed_documents(indexes)

    assert [document.metadata["chunk_id"] for document in documents] == [
        "csv-1",
        "txt-1",
    ]


def test_collect_indexed_documents_rejects_duplicate_chunk_ids():
    indexes = {
        "csv": FakeVectorstore({"csv-1": make_document("duplicado")}),
        "txt": FakeVectorstore({"txt-1": make_document("duplicado")}),
    }

    with pytest.raises(ValueError, match="chunk_id duplicado"):
        collect_indexed_documents(indexes)


def test_collect_indexed_documents_rejects_missing_chunk_id():
    document = Document(page_content="Sem identificador", metadata={})
    indexes = {"txt": FakeVectorstore({"txt-1": document})}

    with pytest.raises(ValueError, match="sem chunk_id"):
        collect_indexed_documents(indexes)
