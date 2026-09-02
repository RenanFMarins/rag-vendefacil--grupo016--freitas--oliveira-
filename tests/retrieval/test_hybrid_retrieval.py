from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.retrieval.models import QueryAnalysis
from src.retrieval.models import QueryFilters
from src.retrieval.pipeline import HybridRetrievalConfig
from src.retrieval.pipeline import HybridRetriever


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
    doc_type: str = "ticket",
    sensitivity: str = "interno",
    source_file: str = "tickets.jsonl",
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source_file": source_file,
            "doc_type": doc_type,
            "chunk_id": chunk_id,
            "sensitivity": sensitivity,
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


def test_multi_document_retrieval_removes_wrong_type_filter_and_backfills() -> None:
    question = (
        "O que consta sobre o Supermercado Boa Compra nos e-mails, "
        "tickets e reuniões?"
    )
    documents = [
        make_document(
            "email:001",
            "E-mail do Supermercado Boa Compra sobre sincronização",
            state="MG",
            doc_type="email",
            source_file="customer_001.txt",
        ),
        make_document(
            "ticket:001",
            "Ticket do Supermercado Boa Compra sobre sincronização",
            state="MG",
        ),
        make_document(
            "ata:001",
            "Reunião de janeiro sobre o Supermercado Boa Compra",
            state="MG",
            doc_type="ata",
            source_file="janeiro.md",
        ),
        make_document(
            "ata:002",
            "Reunião de março sobre o Supermercado Boa Compra",
            state="MG",
            doc_type="ata",
            source_file="marco.md",
        ),
        make_document(
            "email:restricted",
            "E-mail restrito do Supermercado Boa Compra",
            state="MG",
            doc_type="email",
            sensitivity="restrito",
            source_file="credenciais.txt",
        ),
    ]
    embeddings = ConstantEmbeddings()
    vectorstore = FAISS.from_documents(
        documents,
        embeddings,
        ids=[document.metadata["chunk_id"] for document in documents],
    )
    retriever = HybridRetriever(
        {"test": vectorstore},
        embeddings,
        documents,
        llm=FakeChatModel(
            QueryAnalysis(
                query="falha de sincronização",
                filters=QueryFilters(doc_type="email"),
            )
        ),
        config=HybridRetrievalConfig(
            top_k=2,
            multi_document_top_k=4,
            candidate_k=5,
            dense_fetch_k=5,
        ),
    )

    response = retriever.retrieve(
        question,
        allowed_sensitivities={"publico", "interno"},
    )

    assert response.analysis.query == question
    assert response.analysis.filters.doc_type is None
    assert len(response.results) == 4
    assert {
        result.document.metadata["doc_type"] for result in response.results
    } == {"email", "ticket", "ata"}
    assert all(
        result.document.metadata["sensitivity"] != "restrito"
        for result in response.results
    )


def test_exact_identifier_from_bm25_is_not_buried_by_rrf_overlap() -> None:
    generic_documents = [
        make_document(
            f"generic:{number:03d}",
            "Log de erro no serviço de pagamento pay",
            state="SP",
            doc_type="log",
            source_file="system_logs.csv",
        )
        for number in range(12)
    ]
    for number, document in enumerate(generic_documents):
        document.metadata["customer_id"] = f"CUST{number + 100:03d}"
    target = make_document(
        "target:CUST008",
        "Log de erro no serviço de pagamento pay",
        state="SP",
        doc_type="log",
        source_file="system_logs.csv",
    )
    target.metadata["customer_id"] = "CUST008"
    documents = [*generic_documents, target]
    embeddings = ConstantEmbeddings()
    vectorstore = FAISS.from_documents(
        documents,
        embeddings,
        ids=[document.metadata["chunk_id"] for document in documents],
    )
    retriever = HybridRetriever(
        {"test": vectorstore},
        embeddings,
        documents,
        llm=FakeChatModel(
            QueryAnalysis(
                query="logs de erro do cliente CUST008 no pagamento",
                filters=QueryFilters(doc_type="log"),
            )
        ),
        config=HybridRetrievalConfig(
            top_k=3,
            candidate_k=10,
            dense_fetch_k=10,
        ),
    )

    response = retriever.retrieve(
        "Liste os logs de erro do cliente CUST008 no pagamento."
    )

    assert response.results[0].chunk_id == "target:CUST008"
    assert response.results[0].sparse_rank == 1


def test_repeated_error_code_does_not_fill_the_entire_top_k() -> None:
    logs = [
        make_document(
            f"log:{number:03d}",
            "Erro STK-409 durante sincronização de estoque",
            state="SP",
            doc_type="log",
            source_file="system_logs.csv",
        )
        for number in range(6)
    ]
    for document in logs:
        document.metadata["error_code"] = "STK-409"
    manual = make_document(
        "manual:stk-409",
        "Procedimento para resolver Conflict during inventory sync STK-409",
        state="SP",
        doc_type="manual",
        source_file="sincronizacao_estoque.md",
    )
    documents = [manual, *logs]
    embeddings = ConstantEmbeddings()
    vectorstore = FAISS.from_documents(
        documents,
        embeddings,
        ids=[document.metadata["chunk_id"] for document in documents],
    )
    retriever = HybridRetriever(
        {"test": vectorstore},
        embeddings,
        documents,
        llm=FakeChatModel(
            QueryAnalysis(
                query="procedimento para resolver STK-409",
                filters=QueryFilters(module="estoque"),
            )
        ),
        config=HybridRetrievalConfig(
            top_k=3,
            candidate_k=7,
            dense_fetch_k=7,
        ),
    )

    response = retriever.retrieve(
        "Como resolver o erro STK-409 no módulo de estoque?"
    )

    assert response.results[0].document.metadata["error_code"] == "STK-409"
    assert any(
        result.chunk_id == "manual:stk-409"
        for result in response.results
    )
