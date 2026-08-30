from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.hybrid_retrieval import HybridRetrievalConfig
from src.hybrid_retrieval import HybridRetriever
from src.query_analyzer_models import QueryAnalysis
from src.query_analyzer_models import QueryFilters


class ConstantEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FakeStructuredModel:
    def __init__(self, response: QueryAnalysis) -> None:
        self.response = response

    def invoke(self, messages):
        return self.response


class FakeChatModel:
    def __init__(self, response: QueryAnalysis) -> None:
        self.response = response

    def with_structured_output(self, schema, *, method, strict):
        return FakeStructuredModel(self.response)


def make_document(
    chunk_id: str,
    content: str,
    *,
    state: str,
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source_file": "tickets.jsonl",
            "doc_type": "ticket",
            "chunk_id": chunk_id,
            "sensitivity": "interno",
            "state": state,
            "module": "estoque",
        },
    )


def test_runs_complete_pipeline_with_validated_filters_rrf_and_top_k() -> None:
    documents = [
        make_document("chunk-generic", "Falha genérica no estoque", state="MG"),
        make_document("chunk-code", "Ticket TCK-8472 de Ana Souza", state="MG"),
        make_document("chunk-sp", "Ticket TCK-8472 de São Paulo", state="SP"),
    ]
    embeddings = ConstantEmbeddings()
    vectorstore = FAISS.from_documents(
        documents,
        embeddings,
        ids=[document.metadata["chunk_id"] for document in documents],
    )
    llm = FakeChatModel(
        QueryAnalysis(
            query="TCK-8472",
            filters=QueryFilters(state="Minas Gerais"),
        )
    )
    retriever = HybridRetriever(
        {"test": vectorstore},
        embeddings,
        documents,
        llm=llm,
        config=HybridRetrievalConfig(
            top_k=1,
            candidate_k=3,
            dense_fetch_k=3,
        ),
    )

    response = retriever.retrieve(
        "Qual é o ticket TCK-8472 de Minas Gerais?"
    )

    assert response.analysis.query == "TCK-8472"
    assert response.analysis.filters.state == "MG"
    assert all(
        result.document.metadata["state"] == "MG"
        for result in response.dense_response.results
    )
    assert all(
        result.document.metadata["state"] == "MG"
        for result in response.sparse_results
    )
    assert len(response.results) == 1
    assert response.results[0].chunk_id == "chunk-code"
    assert response.results[0].matched_retrievers == ("dense", "bm25")


def test_catalog_is_built_dynamically_and_returned_as_a_copy() -> None:
    documents = [
        make_document("chunk-mg", "Estoque MG", state="MG"),
        make_document("chunk-sp", "Estoque SP", state="SP"),
    ]
    embeddings = ConstantEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = HybridRetriever(
        {"test": vectorstore},
        embeddings,
        documents,
        llm=FakeChatModel(QueryAnalysis(query="estoque")),
    )

    catalog = retriever.metadata_catalog
    catalog["state"].append("RJ")

    assert retriever.metadata_catalog["state"] == ["MG", "SP"]


def test_rejects_configuration_that_cannot_supply_top_k() -> None:
    try:
        HybridRetrievalConfig(top_k=5, candidate_k=4)
    except ValueError as error:
        assert "candidate_k" in str(error)
    else:
        raise AssertionError("Configuração inválida deveria falhar.")
