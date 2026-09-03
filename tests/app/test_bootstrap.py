from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from app import bootstrap


class FakeDocstore:
    def __init__(self, documents: dict[str, Document]) -> None:
        self.documents = documents

    def search(self, document_id: str) -> Document | None:
        return self.documents.get(document_id)


def fake_index(*documents: Document):
    stored = {
        f"doc-{position}": document
        for position, document in enumerate(documents)
    }
    return SimpleNamespace(
        index_to_docstore_id={
            position: document_id
            for position, document_id in enumerate(stored)
        },
        docstore=FakeDocstore(stored),
    )


def make_document(chunk_id: str) -> Document:
    return Document(
        page_content=f"Conteúdo de {chunk_id}",
        metadata={"chunk_id": chunk_id},
    )


def test_documents_from_all_indexes_uses_deterministic_order() -> None:
    first = make_document("csv:001")
    second = make_document("txt:001")

    documents = bootstrap.documents_from_indexes(
        {"txt": fake_index(second), "csv": fake_index(first)}
    )

    assert documents == [first, second]


def test_documents_from_indexes_rejects_duplicate_chunk_id() -> None:
    duplicate = make_document("duplicado:001")

    with pytest.raises(ValueError, match="chunk_id duplicado"):
        bootstrap.documents_from_indexes(
            {"jsonl": fake_index(duplicate), "txt": fake_index(duplicate)}
        )


def test_build_pipeline_fails_before_openai_when_index_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    embeddings_called = False

    def fail_if_called():
        nonlocal embeddings_called
        embeddings_called = True

    monkeypatch.setattr(bootstrap, "create_openai_embeddings", fail_if_called)

    with pytest.raises(FileNotFoundError, match="markdown"):
        bootstrap.build_rag_pipeline({"markdown": tmp_path / "ausente"})

    assert embeddings_called is False


def test_build_pipeline_connects_models_retriever_and_pipeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    index_path = tmp_path / "index"
    index_path.mkdir()
    (index_path / "index.faiss").touch()
    (index_path / "index.pkl").touch()
    document = make_document("markdown:001")
    vectorstore = fake_index(document)
    query_model = object()
    scope_model = object()
    generation_model = object()
    retriever = object()
    pipeline = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(bootstrap, "load_project_environment", lambda: None)
    monkeypatch.setattr(
        bootstrap, "create_openai_embeddings", lambda: "embeddings"
    )
    monkeypatch.setattr(
        bootstrap,
        "load_faiss_index",
        lambda path, embeddings: vectorstore,
    )
    monkeypatch.setattr(bootstrap, "create_chat_model", lambda: query_model)
    monkeypatch.setattr(bootstrap, "create_scope_model", lambda: scope_model)
    monkeypatch.setattr(
        bootstrap, "create_generation_model", lambda: generation_model
    )

    def build_retriever(indexes, embeddings, documents, *, llm, config):
        captured["retriever_args"] = (indexes, embeddings, documents, llm)
        captured["config"] = config
        return retriever

    def build_pipeline(retriever_arg, *, scope_llm, generation_llm):
        captured["pipeline_args"] = (
            retriever_arg,
            scope_llm,
            generation_llm,
        )
        return pipeline

    monkeypatch.setattr(bootstrap, "HybridRetriever", build_retriever)
    monkeypatch.setattr(bootstrap, "RAGPipeline", build_pipeline)

    result = bootstrap.build_rag_pipeline(
        {"markdown": index_path},
        top_k=3,
        candidate_k=12,
    )

    assert result is pipeline
    assert captured["retriever_args"] == (
        {"markdown": vectorstore},
        "embeddings",
        [document],
        query_model,
    )
    assert captured["config"].top_k == 3
    assert captured["config"].candidate_k == 12
    assert captured["pipeline_args"] == (
        retriever,
        scope_model,
        generation_model,
    )
