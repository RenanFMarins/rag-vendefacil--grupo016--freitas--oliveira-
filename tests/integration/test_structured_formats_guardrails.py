from types import SimpleNamespace

from ingestion.loaders.csv_loader import load_csv_documents
from ingestion.loaders.json_loader import load_json_documents
from ingestion.loaders.pdf_loader import load_pdf_documents
from src.config import DATA_DIR
from src.generation.evidence import MAX_QUOTATION_LENGTH
from src.generation.evidence import build_source_evidence
from src.generation.generator import GenerationDraft
from src.pipeline import RAGPipeline
from src.retrieval.metadata_catalog import build_metadata_catalog


EMPLOYEES = DATA_DIR / "structured" / "employees.csv"
PRODUCTS = DATA_DIR / "structured" / "products.json"
POLICIES = DATA_DIR / "unstructured" / "policies"


class FakeStructuredModel:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return self.response


class FakeChatModel:
    def __init__(self, response: object) -> None:
        self.structured_model = FakeStructuredModel(response)

    def with_structured_output(self, schema, *, method, strict):
        return self.structured_model


class FakeRetriever:
    def __init__(self, documents) -> None:
        self.documents = list(documents)
        self.calls = 0
        self.allowed_sensitivities = None
        self.metadata_catalog = build_metadata_catalog(self.documents)

    def retrieve(
        self,
        question: str,
        *,
        debug: bool = False,
        allowed_sensitivities=None,
    ):
        self.calls += 1
        self.allowed_sensitivities = allowed_sensitivities
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    chunk_id=document.metadata["chunk_id"],
                    document=document,
                )
                for document in self.documents
            ]
        )


def test_salary_question_is_refused_before_accessing_employee_csv() -> None:
    employee = load_csv_documents(EMPLOYEES)[0]
    retriever = FakeRetriever([employee])
    generation_llm = FakeChatModel({"invalid": "must not be called"})
    pipeline = RAGPipeline(retriever, generation_llm=generation_llm)

    response = pipeline.answer(
        "Qual é o salário atual da funcionária Ana Souza na VendeFácil?"
    )

    assert employee.metadata["sensitivity"] == "restrito"
    assert response.is_refusal is True
    assert response.refusal_reason == "lgpd"
    assert response.sources_used == []
    assert retriever.calls == 0
    assert generation_llm.structured_model.calls == 0


def test_restricted_employee_chunk_never_reaches_generation() -> None:
    employee = load_csv_documents(EMPLOYEES)[0]
    retriever = FakeRetriever([employee])
    generation_llm = FakeChatModel({"invalid": "must not be called"})
    pipeline = RAGPipeline(retriever, generation_llm=generation_llm)

    response = pipeline.answer(
        "Quais informações existem sobre os funcionários da VendeFácil?"
    )

    assert retriever.allowed_sensitivities == {"publico", "interno"}
    assert response.is_refusal is True
    assert response.refusal_reason == "sem_evidencia"
    assert generation_llm.structured_model.calls == 0


def test_json_product_generates_grounded_structured_response() -> None:
    product = next(
        document
        for document in load_json_documents(PRODUCTS)
        if document.metadata.get("product_id") == "PROD-ESTOQUE"
    )
    retriever = FakeRetriever([product])
    generation_llm = FakeChatModel(
        GenerationDraft(
            answer="O VendeFácil Estoque gerencia inventário e suprimentos.",
            confidence_level="alta",
            selected_chunk_ids=[product.metadata["chunk_id"]],
            reasoning="O produto e suas funções constam no cadastro.",
            has_sufficient_evidence=True,
        )
    )
    pipeline = RAGPipeline(retriever, generation_llm=generation_llm)

    response = pipeline.answer(
        "Quais são as funções do VendeFácil Estoque?"
    )

    assert response.is_refusal is False
    assert response.confidence_level == "alta"
    assert response.sources_used[0].filepath == "products.json"
    assert response.sources_used[0].chunk_id == product.metadata["chunk_id"]
    assert response.sources_used[0].quotation in product.page_content


def test_pdf_policy_builds_literal_bounded_evidence() -> None:
    document = load_pdf_documents(POLICIES)[0]

    evidence = build_source_evidence(document)

    assert evidence.filepath.endswith(".pdf")
    assert evidence.chunk_id == document.metadata["chunk_id"]
    assert evidence.quotation == document.page_content[:MAX_QUOTATION_LENGTH]
    assert evidence.quotation in document.page_content
    assert len(evidence.quotation) <= 500

