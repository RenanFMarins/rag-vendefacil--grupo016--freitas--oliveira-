import logging
from types import SimpleNamespace

from langchain_core.documents import Document

from src.generation.generator import GenerationDraft
from src.pipeline import RAGPipeline
from src.guardrails.scope import ScopeDecision


class FakeStructuredModel:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls = 0
        self.messages = []

    def invoke(self, messages):
        self.calls += 1
        self.messages = messages
        return self.response


class FakeChatModel:
    def __init__(self, response: object) -> None:
        self.structured_model = FakeStructuredModel(response)

    def with_structured_output(self, schema, *, method, strict):
        return self.structured_model


class FakeRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.calls = 0
        self.questions: list[str] = []
        self.allowed_sensitivities = None
        self.metadata_catalog = {
            "doc_type": ["email", "manual", "ticket"],
            "module": ["estoque", "pdv"],
            "sensitivity": ["interno", "publico", "restrito"],
        }

    def retrieve(
        self,
        question: str,
        *,
        debug: bool = False,
        allowed_sensitivities=None,
    ):
        self.calls += 1
        self.questions.append(question)
        self.allowed_sensitivities = allowed_sensitivities
        results = [
            SimpleNamespace(
                chunk_id=document.metadata["chunk_id"],
                document=document,
            )
            for document in self.documents
        ]
        return SimpleNamespace(results=results)


def make_document(
    content: str,
    *,
    chunk_id: str = "manual:pdv:001",
    sensitivity: str = "publico",
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source_file": "manual_pdv.md",
            "chunk_id": chunk_id,
            "doc_type": "manual",
            "sensitivity": sensitivity,
        },
    )


def corporate_scope_model() -> FakeChatModel:
    return FakeChatModel(
        ScopeDecision(
            classification="corporativa",
            reason="Pergunta relacionada à operação da VendeFácil.",
        )
    )


def test_lgpd_refusal_happens_before_scope_and_retrieval() -> None:
    retriever = FakeRetriever([])
    scope_llm = corporate_scope_model()
    generation_llm = FakeChatModel({"invalid": "should not be called"})
    pipeline = RAGPipeline(
        retriever,
        scope_llm=scope_llm,
        generation_llm=generation_llm,
    )

    response = pipeline.answer("Qual o salário do funcionário João Pereira?")

    assert response.is_refusal is True
    assert response.confidence_level == "recusado"
    assert response.sources_used == []
    assert response.refusal_reason == "lgpd"
    assert scope_llm.structured_model.calls == 0
    assert retriever.calls == 0
    assert generation_llm.structured_model.calls == 0


def test_out_of_scope_refusal_happens_before_retrieval() -> None:
    retriever = FakeRetriever([])
    scope_llm = FakeChatModel(
        ScopeDecision(
            classification="fora_de_escopo",
            reason="Pergunta de conhecimento geral.",
        )
    )
    generation_llm = FakeChatModel({"invalid": "should not be called"})
    pipeline = RAGPipeline(
        retriever,
        scope_llm=scope_llm,
        generation_llm=generation_llm,
    )

    response = pipeline.answer("Quem descobriu o Brasil?")

    assert response.is_refusal is True
    assert response.refusal_reason == "fora_de_escopo"
    assert retriever.calls == 0
    assert generation_llm.structured_model.calls == 0


def test_generates_grounded_response_for_allowed_question() -> None:
    document = make_document(
        "A sangria retira valores do gaveteiro por segurança."
    )
    retriever = FakeRetriever([document])
    generation_llm = FakeChatModel(
        GenerationDraft(
            answer="A sangria retira valores do gaveteiro.",
            confidence_level="alta",
            selected_chunk_ids=[document.metadata["chunk_id"]],
            reasoning="O manual define a sangria diretamente.",
            has_sufficient_evidence=True,
        )
    )
    pipeline = RAGPipeline(
        retriever,
        scope_llm=corporate_scope_model(),
        generation_llm=generation_llm,
    )

    response = pipeline.answer("O que é sangria no VendeFácil PDV?")

    assert response.is_refusal is False
    assert response.confidence_level == "alta"
    assert response.sources_used[0].chunk_id == "manual:pdv:001"
    assert response.sources_used[0].quotation == document.page_content
    assert retriever.calls == 1
    assert retriever.allowed_sensitivities == {"publico", "interno"}


def test_answer_with_trace_exposes_only_safe_retrieval_identifiers() -> None:
    document = make_document("Manual público do PDV.")
    retriever = FakeRetriever([document])
    generation_llm = FakeChatModel(
        GenerationDraft(
            answer="O documento é um manual do PDV.",
            confidence_level="alta",
            selected_chunk_ids=[document.metadata["chunk_id"]],
            reasoning="O tipo está identificado no documento.",
            has_sufficient_evidence=True,
        )
    )
    pipeline = RAGPipeline(
        retriever,
        scope_llm=corporate_scope_model(),
        generation_llm=generation_llm,
    )

    execution = pipeline.answer_with_trace(
        "O que informa o manual do VendeFácil PDV?"
    )

    assert execution.response.is_refusal is False
    assert len(execution.retrieved_chunks) == 1
    assert execution.retrieved_chunks[0].chunk_id == "manual:pdv:001"
    assert execution.retrieved_chunks[0].source_file == "manual_pdv.md"
    assert execution.retrieved_chunks[0].rank == 1
    assert not hasattr(execution.retrieved_chunks[0], "page_content")


def test_masks_context_and_final_response_without_modifying_document() -> None:
    raw_email = "maria@email.com"
    document = make_document(
        f"E-mail pessoal do cliente: {raw_email}",
        chunk_id="email:customer:001",
        sensitivity="interno",
    )
    retriever = FakeRetriever([document])
    generation_llm = FakeChatModel(
        GenerationDraft(
            answer=f"O e-mail é {raw_email}.",
            confidence_level="alta",
            selected_chunk_ids=[document.metadata["chunk_id"]],
            reasoning=f"O cadastro apresenta {raw_email}.",
            has_sufficient_evidence=True,
        )
    )
    pipeline = RAGPipeline(
        retriever,
        scope_llm=corporate_scope_model(),
        generation_llm=generation_llm,
    )

    response = pipeline.answer(
        f"O e-mail pessoal {raw_email} é do cliente CUST001 na VendeFácil?"
    )

    prompt = str(generation_llm.structured_model.messages)
    assert raw_email not in prompt
    assert raw_email not in retriever.questions[0]
    assert raw_email not in response.answer
    assert "ma***@***.com" in response.answer
    assert raw_email not in response.sources_used[0].quotation
    assert document.page_content == f"E-mail pessoal do cliente: {raw_email}"


def test_restricted_chunks_never_reach_generation() -> None:
    restricted = make_document(
        "Conteúdo extremamente sensível.",
        chunk_id="email:restricted:001",
        sensitivity="restrito",
    )
    retriever = FakeRetriever([restricted])
    generation_llm = FakeChatModel({"invalid": "should not be called"})
    pipeline = RAGPipeline(
        retriever,
        scope_llm=corporate_scope_model(),
        generation_llm=generation_llm,
    )

    response = pipeline.answer("Quais informações existem na VendeFácil?")

    assert response.is_refusal is True
    assert response.refusal_reason == "sem_evidencia"
    assert generation_llm.structured_model.calls == 0


def test_debug_logs_only_safe_operational_fields(caplog) -> None:
    document = make_document("Manual público do PDV.")
    retriever = FakeRetriever([document])
    generation_llm = FakeChatModel(
        GenerationDraft(
            answer="O manual descreve o PDV.",
            confidence_level="alta",
            selected_chunk_ids=[document.metadata["chunk_id"]],
            reasoning="Informação presente no manual.",
            has_sufficient_evidence=True,
        )
    )
    pipeline = RAGPipeline(
        retriever,
        scope_llm=corporate_scope_model(),
        generation_llm=generation_llm,
    )

    with caplog.at_level(logging.INFO):
        pipeline.answer("Explique o manual do VendeFácil PDV.", debug=True)

    assert "Ação LGPD: RESPONDER" in caplog.text
    assert "Classificação de escopo: corporativa" in caplog.text
    assert "chunk_ids recuperados: ['manual:pdv:001']" in caplog.text
    assert "Quantidade de fontes: 1" in caplog.text
    assert "confidence_level: alta" in caplog.text
    assert "is_refusal: False" in caplog.text
    assert "Manual público do PDV" not in caplog.text


def test_debug_does_not_log_sensitive_question_values(caplog) -> None:
    retriever = FakeRetriever([])
    pipeline = RAGPipeline(
        retriever,
        scope_llm=corporate_scope_model(),
        generation_llm=FakeChatModel({"invalid": "should not be called"}),
    )

    with caplog.at_level(logging.INFO):
        pipeline.answer(
            "Qual o salário individual de João Pereira?",
            debug=True,
        )

    assert "Ação LGPD: RECUSAR" in caplog.text
    assert "João Pereira" not in caplog.text
    assert "Qual o salário" not in caplog.text

