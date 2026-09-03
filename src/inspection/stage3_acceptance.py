"""Demonstração offline dos principais critérios da Etapa 3."""

from langchain_core.documents import Document
from pydantic import ValidationError

from src.generation.generator import GenerationConfig
from src.generation.generator import GenerationDraft
from src.generation.generator import generate_rag_response
from src.guardrails.lgpd import classify_lgpd_question
from src.guardrails.masking import mask_personal_data
from starter.schema import RAGResponse
from src.guardrails.scope import ScopeDecision
from src.guardrails.scope import build_out_of_scope_refusal


class SequenceStructuredModel:
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


def _check(label: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"[PASS] {label}")


def _document() -> Document:
    return Document(
        page_content="A sangria retira valores do caixa por segurança.",
        metadata={
            "source_file": "manual_pdv.md",
            "chunk_id": "manual:pdv:sangria",
            "doc_type": "manual",
            "sensitivity": "publico",
        },
    )


def _draft() -> GenerationDraft:
    return GenerationDraft(
        answer="A sangria retira valores do caixa.",
        confidence_level="alta",
        selected_chunk_ids=["manual:pdv:sangria"],
        reasoning="A informação está no manual.",
        has_sufficient_evidence=True,
    )


def main() -> None:
    salary = classify_lgpd_question(
        "Qual o salário individual de João Pereira?"
    )
    _check("salário individual recusado", salary.action == "RECUSAR")

    cpf = classify_lgpd_question("Informe o CPF do cliente CUST001.")
    _check("CPF recusado", cpf.action == "RECUSAR")

    email = "maria@email.com"
    email_decision = classify_lgpd_question("Qual é o e-mail pessoal?")
    masked_email = mask_personal_data(email)
    _check(
        "email mascarado",
        email_decision.action == "MASCARAR" and email not in masked_email,
    )

    phone = "(31) 98765-4312"
    phone_decision = classify_lgpd_question("Qual é o telefone do cliente?")
    masked_phone = mask_personal_data(phone)
    _check(
        "telefone mascarado",
        phone_decision.action == "MASCARAR" and phone not in masked_phone,
    )

    document = _document()
    answered = generate_rag_response(
        "O que é sangria no PDV?",
        [document],
        llm=FakeChatModel(_draft()),
    )
    _check(
        "consulta permitida respondida com evidência",
        not answered.is_refusal
        and bool(answered.sources_used)
        and answered.sources_used[0].quotation in document.page_content,
    )

    outside = build_out_of_scope_refusal(
        ScopeDecision(
            classification="fora_de_escopo",
            reason="Assunto sem relação com a VendeFácil.",
        )
    )
    _check(
        "fora de escopo recusado",
        outside.is_refusal and outside.refusal_reason == "fora_de_escopo",
    )

    no_evidence = generate_rag_response("Pergunta sem documentos.", [])
    _check(
        "sem evidência tratado",
        no_evidence.is_refusal
        and no_evidence.refusal_reason == "sem_evidencia",
    )

    consistency_rejected = False
    try:
        RAGResponse(
            answer="Resposta sem fonte.",
            confidence_level="alta",
            sources_used=[],
            reasoning="Inválida.",
            is_refusal=False,
        )
    except ValidationError:
        consistency_rejected = True
    _check("Pydantic consistency", consistency_rejected)

    retry_llm = FakeChatModel({"answer": "incompleta"}, _draft())
    retried = generate_rag_response(
        "O que é sangria no PDV?",
        [document],
        llm=retry_llm,
        config=GenerationConfig(max_attempts=2),
    )
    _check(
        "retry",
        retry_llm.structured_model.calls == 2 and not retried.is_refusal,
    )


if __name__ == "__main__":
    main()

