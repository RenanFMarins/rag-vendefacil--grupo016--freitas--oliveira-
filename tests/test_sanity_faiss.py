from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from ingestion.build._faiss import save_faiss_index
from ingestion.preview.sanity_faiss import index_statistics
from ingestion.preview.sanity_faiss import load_indexes
from ingestion.preview.sanity_faiss import search_all_indexes
from ingestion.preview.sanity_faiss import shorten_content


def make_document(chunk_id: str, doc_type: str, content: str) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source_file": f"{chunk_id}.txt",
            "doc_type": doc_type,
            "chunk_id": chunk_id,
            "sensitivity": "interno",
        },
    )


def test_reloads_indexes_calculates_statistics_and_combines_results(
    tmp_path: Path,
) -> None:
    embeddings = DeterministicFakeEmbedding(size=32)
    index_paths = {
        "jsonl": tmp_path / "faiss_jsonl",
        "txt": tmp_path / "faiss_txt",
        "markdown": tmp_path / "faiss_markdown",
    }
    documents_by_index = {
        "jsonl": [
            make_document("ticket-1", "ticket", "Erro de estoque"),
            make_document("ticket-2", "ticket", "Erro de pagamento"),
        ],
        "txt": [
            make_document("email-1", "email", "Mensagem do cliente"),
        ],
        "markdown": [
            make_document("manual-1", "manual", "Manual de sangria"),
            make_document("policy-1", "policy", "Política de atendimento"),
        ],
    }

    for name, documents in documents_by_index.items():
        save_faiss_index(documents, index_paths[name], embeddings=embeddings)

    indexes = load_indexes(embeddings, index_paths=index_paths)
    total, distribution = index_statistics(indexes)
    results = search_all_indexes(
        "Como fazer uma sangria?",
        indexes,
        embeddings,
        top_k=5,
    )

    assert total == 5
    assert distribution == {"ticket": 2, "email": 1, "manual": 1, "policy": 1}
    assert len(results) == 5
    assert {result.index_name for result in results} == {
        "jsonl",
        "txt",
        "markdown",
    }
    assert [result.distance for result in results] == sorted(
        result.distance for result in results
    )


def test_shorten_content_normalizes_and_limits_text() -> None:
    assert shorten_content("linha 1\n\nlinha 2", 0) == "linha 1 linha 2"
    assert shorten_content("abcdefghij", 8) == "abcde..."
