from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ingestion.loaders.csv_loader import load_csv_documents
from ingestion.loaders.json_loader import load_json_documents
from src.config import DATA_DIR
from src.retrieval.models import QueryAnalysis
from src.retrieval.models import QueryFilters
from src.retrieval.pipeline import HybridRetrievalConfig
from src.retrieval.pipeline import HybridRetriever
from src.retrieval.sparse import BM25SparseRetriever


CUSTOMERS = DATA_DIR / "structured" / "customers.csv"
STORES = DATA_DIR / "structured" / "stores.json"


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


def build_retriever(
    documents: list[Document],
    analysis: QueryAnalysis,
) -> HybridRetriever:
    embeddings = ConstantEmbeddings()
    vectorstore = FAISS.from_documents(
        documents,
        embeddings,
        ids=[str(document.metadata["chunk_id"]) for document in documents],
    )
    return HybridRetriever(
        {"structured": vectorstore},
        embeddings,
        documents,
        llm=FakeChatModel(analysis),
        config=HybridRetrievalConfig(
            top_k=5,
            candidate_k=10,
            dense_fetch_k=10,
        ),
    )


def test_hybrid_retrieval_filters_real_stores_by_state_and_module() -> None:
    documents = load_json_documents(STORES)
    retriever = build_retriever(
        documents,
        QueryAnalysis(
            query="lojas com estoque em Minas Gerais",
            filters=QueryFilters(
                doc_type="store",
                state="Minas Gerais",
                module="ESTOQUE",
            ),
        ),
    )

    response = retriever.retrieve(
        "Quais lojas de Minas Gerais possuem o módulo de estoque?"
    )

    assert response.analysis.filters.model_dump(exclude_none=True) == {
        "doc_type": "store",
        "state": "MG",
        "module": "estoque",
    }
    assert response.results
    assert all(
        result.document.metadata["doc_type"] == "store"
        and result.document.metadata["state"] == "MG"
        and "estoque" in result.document.metadata["module"]
        for result in response.results
    )
    assert any(
        result.matched_retrievers == ("dense", "bm25")
        for result in response.results
    )


def test_hybrid_retrieval_filters_real_customers_by_commercial_plan() -> None:
    customers = load_csv_documents(CUSTOMERS)
    enterprise = [
        document
        for document in customers
        if document.metadata["plan"] == "Enterprise"
    ][:5]
    other_plans = [
        next(
            document
            for document in customers
            if document.metadata["plan"] == plan
        )
        for plan in ("Basic", "Pro")
    ]
    documents = [*enterprise, *other_plans]
    retriever = build_retriever(
        documents,
        QueryAnalysis(
            query="clientes e valor mensal",
            filters=QueryFilters(doc_type="customer", plan="enterprise"),
        ),
    )

    response = retriever.retrieve(
        "Quais clientes estão no plano Enterprise e qual o valor mensal?"
    )

    assert response.analysis.filters.model_dump(exclude_none=True) == {
        "doc_type": "customer",
        "plan": "Enterprise",
    }
    assert len(response.results) == 5
    assert all(
        result.document.metadata["plan"] == "Enterprise"
        for result in response.results
    )


def test_bm25_finds_exact_customer_identifier_in_real_csv_documents() -> None:
    customers = load_csv_documents(CUSTOMERS)
    target = next(
        document
        for document in customers
        if document.metadata["customer_id"] == "CUST001"
    )
    distractors = [
        document
        for document in customers
        if document.metadata["customer_id"] != "CUST001"
    ][:20]
    retriever = BM25SparseRetriever([target, *distractors])

    results = retriever.search("CUST001", k=1)

    assert results[0].chunk_id == "customers:CUST001"
