from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ingestion.build._faiss import save_faiss_index
from src.dense_search import DenseSearchConfig
from src.dense_search import dense_search
from src.query_analyzer_models import QueryFilters


class KeywordEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        normalized = text.casefold()
        if "estoque" in normalized:
            return [1.0, 0.0]
        if "pay" in normalized or "pagamento" in normalized:
            return [0.0, 1.0]
        return [0.5, 0.5]


def make_document(number: int, *, state: str, module: str) -> Document:
    return Document(
        page_content=f"Documento sobre {module} número {number}",
        metadata={
            "source_file": "documents.txt",
            "doc_type": "ticket",
            "chunk_id": f"chunk-{number:03d}",
            "sensitivity": "interno",
            "state": state,
            "module": module,
        },
    )


def build_test_index(tmp_path: Path):
    documents = [
        make_document(0, state="MG", module="estoque"),
        make_document(1, state="SP", module="estoque"),
        make_document(2, state="SP", module="estoque"),
        make_document(3, state="SP", module="estoque"),
        *[
            make_document(number, state="SP", module="pay")
            for number in range(4, 10)
        ],
    ]
    embeddings = KeywordEmbeddings()
    vectorstore = save_faiss_index(
        documents,
        tmp_path / "faiss_test",
        embeddings=embeddings,
    )
    return {"test": vectorstore}, embeddings


def test_uses_exact_prefilter_for_extremely_selective_filter(
    tmp_path: Path,
) -> None:
    indexes, embeddings = build_test_index(tmp_path)
    config = DenseSearchConfig(
        k=5,
        fetch_k=5,
        prefilter_selectivity_threshold=0.15,
    )

    response = dense_search(
        "problema de estoque",
        indexes,
        embeddings,
        validated_filters=QueryFilters(state="MG", module="estoque"),
        config=config,
    )

    assert len(response.results) == 1
    assert response.results[0].document.metadata["state"] == "MG"
    assert response.results[0].document.metadata["module"] == "estoque"
    assert response.diagnostics[0].strategy == "exact_prefilter"
    assert response.diagnostics[0].eligible_documents == 1
    assert response.diagnostics[0].selectivity == 0.1


def test_uses_adaptive_fetch_k_for_less_selective_filter(
    tmp_path: Path,
) -> None:
    indexes, embeddings = build_test_index(tmp_path)
    config = DenseSearchConfig(
        k=2,
        fetch_k=3,
        oversampling_factor=1.5,
        prefilter_selectivity_threshold=0.15,
    )

    response = dense_search(
        "problema de estoque",
        indexes,
        embeddings,
        validated_filters=QueryFilters(module="estoque"),
        config=config,
    )

    assert len(response.results) == 2
    assert all(
        result.document.metadata["module"] == "estoque"
        for result in response.results
    )
    assert response.diagnostics[0].strategy == "adaptive_postfilter"
    assert response.diagnostics[0].fetch_k == 8


def test_compares_unfiltered_and_filtered_results(tmp_path: Path) -> None:
    indexes, embeddings = build_test_index(tmp_path)
    config = DenseSearchConfig(k=3, fetch_k=5)

    unfiltered = dense_search(
        "problema de pagamento",
        indexes,
        embeddings,
        config=config,
    )
    filtered = dense_search(
        "problema de pagamento",
        indexes,
        embeddings,
        validated_filters=QueryFilters(state="MG"),
        config=config,
    )

    assert unfiltered.diagnostics[0].strategy == "unfiltered"
    assert any(
        result.document.metadata["state"] == "SP"
        for result in unfiltered.results
    )
    assert all(
        result.document.metadata["state"] == "MG"
        for result in filtered.results
    )


def test_returns_no_results_when_no_document_matches(tmp_path: Path) -> None:
    indexes, embeddings = build_test_index(tmp_path)

    response = dense_search(
        "problema de estoque",
        indexes,
        embeddings,
        validated_filters=QueryFilters(state="RJ"),
    )

    assert response.results == []
    assert response.diagnostics[0].strategy == "no_matches"


def test_rejects_invalid_search_configuration() -> None:
    try:
        DenseSearchConfig(k=5, fetch_k=4)
    except ValueError as error:
        assert "fetch_k" in str(error)
    else:
        raise AssertionError("A configuração inválida deveria falhar.")
