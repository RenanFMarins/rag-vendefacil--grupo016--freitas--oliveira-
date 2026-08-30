import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.dense_search import DenseSearchConfig
from src.dense_search import dense_search
from src.query_analyzer_models import QueryFilters
from src.sparse_search import BM25SparseRetriever
from src.sparse_search import tokenize_for_bm25


class SemanticEmbeddings(Embeddings):

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        normalized = text.casefold()
        cash_concept = (
            "sangria",
            "numerário",
            "gaveteiro",
            "dinheiro",
            "caixa",
        )
        if any(term in normalized for term in cash_concept):
            return [1.0, 0.0]
        return [0.0, 1.0]


def make_document(
    chunk_id: str,
    content: str,
    *,
    state: str = "MG",
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source_file": "test.txt",
            "doc_type": "ticket",
            "chunk_id": chunk_id,
            "sensitivity": "interno",
            "state": state,
        },
    )


def comparison_documents() -> list[Document]:
    return [
        make_document("chunk-other", "Falha genérica no terminal"),
        make_document(
            "chunk-code",
            "Incidente TCK-8472 registrado para Ana Souza",
        ),
        make_document(
            "chunk-cash",
            "A sangria retira numerário acumulado no gaveteiro",
            state="SP",
        ),
    ]


def build_dense_index(
    documents: list[Document],
    embeddings: Embeddings,
) -> dict[str, FAISS]:
    return {
        "test": FAISS.from_documents(
            documents,
            embeddings,
            ids=[document.metadata["chunk_id"] for document in documents],
        )
    }


def test_tokenizer_preserves_codes_names_numbers_and_normalizes_accents() -> None:
    tokens = tokenize_for_bm25("TCK-8472, Ana Souza: versão 3.4")

    assert tokens == ["tck-8472", "ana", "souza", "versao", "3.4"]


def test_bm25_is_better_for_an_exact_code_than_dense() -> None:
    documents = comparison_documents()
    embeddings = SemanticEmbeddings()
    indexes = build_dense_index(documents, embeddings)

    dense = dense_search(
        "TCK-8472",
        indexes,
        embeddings,
        config=DenseSearchConfig(k=1, fetch_k=1),
    )
    sparse = BM25SparseRetriever(documents).search("TCK-8472", k=1)

    assert dense.results[0].chunk_id != "chunk-code"
    assert sparse[0].chunk_id == "chunk-code"


def test_dense_is_better_for_a_semantic_paraphrase_than_bm25() -> None:
    documents = comparison_documents()
    embeddings = SemanticEmbeddings()
    indexes = build_dense_index(documents, embeddings)
    question = "Como reduzir dinheiro guardado no caixa?"

    dense = dense_search(
        question,
        indexes,
        embeddings,
        config=DenseSearchConfig(k=1, fetch_k=1),
    )
    sparse = BM25SparseRetriever(documents).search(question, k=3)

    assert dense.results[0].chunk_id == "chunk-cash"
    assert all(result.chunk_id != "chunk-cash" for result in sparse)


def test_bm25_applies_only_validated_filters_and_preserves_chunk_id() -> None:
    retriever = BM25SparseRetriever(comparison_documents())

    results = retriever.search(
        "incidente TCK-8472",
        validated_filters=QueryFilters(state="MG"),
    )

    assert results[0].chunk_id == "chunk-code"
    assert results[0].document.metadata["chunk_id"] == "chunk-code"
    assert results[0].rank == 1


def test_bm25_rejects_missing_or_duplicated_chunk_ids() -> None:
    missing_id = Document(page_content="conteúdo", metadata={})
    with pytest.raises(ValueError, match="chunk_id"):
        BM25SparseRetriever([missing_id])

    duplicated = [
        make_document("same-id", "primeiro"),
        make_document("same-id", "segundo"),
    ]
    with pytest.raises(ValueError, match="únicos"):
        BM25SparseRetriever(duplicated)
