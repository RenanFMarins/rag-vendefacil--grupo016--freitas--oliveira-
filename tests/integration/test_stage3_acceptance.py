"""Testes de aceitação dos critérios de pronto da Etapa 3.

A suíte é totalmente offline: modelos falsos simulam somente o contrato de
structured output, sem chamadas de rede ou consumo de API.
"""

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from src.generation.evidence import MAX_QUOTATION_LENGTH
from src.generation.evidence import build_source_evidence
from src.generation.generator import GenerationConfig
from src.generation.generator import GenerationDraft
from src.generation.generator import generate_rag_response
from src.guardrails.lgpd import classify_lgpd_question
from src.guardrails.masking import mask_personal_data
from starter.schema import RAGResponse
from starter.schema import SourceEvidence
from src.guardrails.scope import ScopeDecision
from src.guardrails.scope import build_out_of_scope_refusal
from src.guardrails.scope import classify_question_scope


CATALOG = {
    "doc_type": ["manual", "policy", "ticket"],
    "module": ["estoque", "pay", "pdv"],
}


class SequenceStructuredModel:
    """Devolve respostas predeterminadas para exercitar structured output."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls = 0

    def invoke(self, messages):
        del messages
        response = self.responses[self.calls]
        self.calls += 1
        return response


class FakeChatModel:
    def __init__(self, *responses: object) -> None:
        self.structured_model = SequenceStructuredModel(list(responses))

    def with_structured_output(self, schema, *, method, strict):
        del schema, method, strict
        return self.structured_model


def make_document(content: str | None = None) -> Document:
    return Document(
        page_content=content or "A sangria retira valores do caixa por segurança.",
        metadata={
            "source_file": "manual_pdv.md",
            "chunk_id": "manual:pdv:sangria",
            "doc_type": "manual",
            "sensitivity": "publico",
        },
    )


def valid_draft() -> GenerationDraft:
    return GenerationDraft(
        answer="A sangria retira valores do caixa.",
        confidence_level="alta",
        selected_chunk_ids=["manual:pdv:sangria"],
        reasoning="O manual descreve diretamente a operação.",
        has_sufficient_evidence=True,
    )


@pytest.mark.parametrize(
    ("question", "category"),
    [
        ("Qual o salário individual de João Pereira?", "salario_individual"),
        ("Informe o CPF do cliente CUST001.", "cpf"),
    ],
)
def test_lgpd_refuses_protected_data(question: str, category: str) -> None:
    decision = classify_lgpd_question(question)

    assert decision.action == "RECUSAR"
    assert decision.category == category


@pytest.mark.parametrize(
    ("question", "raw_value"),
    [
        ("Qual é o e-mail pessoal do cliente?", "maria@email.com"),
        ("Qual é o telefone do cliente?", "(31) 98765-4312"),
    ],
)
def test_lgpd_masks_allowed_pii(question: str, raw_value: str) -> None:
    decision = classify_lgpd_question(question)
    masked = mask_personal_data(f"Contato: {raw_value}")

    assert decision.action == "MASCARAR"
    assert raw_value not in masked
    assert "*" in masked


@pytest.mark.parametrize(
    "question",
    [
        "Quais informações existem no manual do PDV?",
        "Quais produtos estão cadastrados na VendeFácil?",
    ],
)
def test_lgpd_allows_non_sensitive_corporate_questions(question: str) -> None:
    assert classify_lgpd_question(question).action == "RESPONDER"


@pytest.mark.parametrize(
    "question",
    ["Quem descobriu o Brasil?", "Me escreva um poema."],
)
def test_out_of_scope_questions_are_refused(question: str) -> None:
    llm = FakeChatModel(
        ScopeDecision(
            classification="fora_de_escopo",
            reason="Assunto sem relação com o corpus corporativo.",
        )
    )

    decision = classify_question_scope(question, CATALOG, llm=llm)
    response = build_out_of_scope_refusal(decision)

    assert response.is_refusal is True
    assert response.refusal_reason == "fora_de_escopo"
    assert response.sources_used == []


def test_question_without_evidence_returns_controlled_refusal() -> None:
    response = generate_rag_response(
        "Existe suporte para uma funcionalidade ausente do corpus?",
        [],
    )

    assert response.is_refusal is True
    assert response.confidence_level == "recusado"
    assert response.refusal_reason == "sem_evidencia"


def test_valid_rag_response_is_accepted() -> None:
    response = RAGResponse(
        answer="Procedimento descrito no manual.",
        confidence_level="alta",
        sources_used=[
            SourceEvidence(
                filepath="manual_pdv.md",
                chunk_id="manual:pdv:001",
                quotation="Trecho literal do manual.",
            )
        ],
        reasoning="A fonte sustenta a resposta.",
        is_refusal=False,
    )

    assert response.is_refusal is False


def test_invalid_confidence_level_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RAGResponse(
            answer="Resposta.",
            confidence_level="muito alta",
            sources_used=[],
            reasoning="Inválida.",
            is_refusal=False,
        )


def test_refusal_without_reason_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RAGResponse(
            answer="Recusa.",
            confidence_level="recusado",
            sources_used=[],
            reasoning="Sem motivo tipado.",
            is_refusal=True,
        )


def test_refusal_with_sources_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RAGResponse(
            answer="Recusa.",
            confidence_level="recusado",
            sources_used=[
                SourceEvidence(
                    filepath="manual.md",
                    chunk_id="chunk:001",
                    quotation="Trecho.",
                )
            ],
            reasoning="Recusas não carregam fontes.",
            is_refusal=True,
            refusal_reason="lgpd",
        )


def test_normal_response_without_source_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RAGResponse(
            answer="Resposta sem evidência.",
            confidence_level="baixa",
            sources_used=[],
            reasoning="Sem fonte.",
            is_refusal=False,
        )


def test_evidence_has_identity_and_literal_bounded_quotation() -> None:
    document = make_document("Trecho literal " + "x" * 600)

    evidence = build_source_evidence(document)

    assert evidence.filepath == document.metadata["source_file"]
    assert evidence.chunk_id == document.metadata["chunk_id"]
    assert evidence.quotation == document.page_content[:MAX_QUOTATION_LENGTH]
    assert evidence.quotation in document.page_content
    assert len(evidence.quotation) <= 500


def test_retry_succeeds_after_first_invalid_output() -> None:
    llm = FakeChatModel({"answer": "incompleta"}, valid_draft())

    response = generate_rag_response(
        "O que é sangria no PDV?",
        [make_document()],
        llm=llm,
        config=GenerationConfig(max_attempts=2),
    )

    assert llm.structured_model.calls == 2
    assert response.is_refusal is False
    assert response.sources_used


def test_retry_exhaustion_returns_valid_controlled_refusal() -> None:
    llm = FakeChatModel(
        {"answer": "primeira incompleta"},
        {"confidence_level": "inválida"},
    )

    response = generate_rag_response(
        "O que é sangria no PDV?",
        [make_document()],
        llm=llm,
        config=GenerationConfig(max_attempts=2),
    )

    assert llm.structured_model.calls == 2
    assert isinstance(response, RAGResponse)
    assert response.is_refusal is True
    assert response.refusal_reason == "sem_evidencia"

