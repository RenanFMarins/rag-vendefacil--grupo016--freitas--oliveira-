import pytest

from src.guardrails.scope import ScopeDecision
from src.guardrails.scope import build_out_of_scope_refusal
from src.guardrails.scope import classify_question_scope


CATALOG = {
    "doc_type": ["manual", "policy", "ticket"],
    "module": ["analytics", "estoque", "pay", "pdv"],
    "state": ["MG", "SP"],
}


class FakeStructuredModel:
    def __init__(self, response: ScopeDecision) -> None:
        self.response = response
        self.messages = []

    def invoke(self, messages):
        self.messages = messages
        return self.response


class FakeChatModel:
    def __init__(self, response: ScopeDecision) -> None:
        self.structured_model = FakeStructuredModel(response)
        self.schema = None
        self.method = None
        self.strict = None

    def with_structured_output(self, schema, *, method, strict):
        self.schema = schema
        self.method = method
        self.strict = strict
        return self.structured_model


@pytest.mark.parametrize(
    "question",
    [
        "Quais tickets de estoque estão abertos?",
        "Como realizar uma sangria no VendeFácil PDV?",
    ],
)
def test_accepts_clearly_corporate_questions(question: str) -> None:
    fake_llm = FakeChatModel(
        ScopeDecision(
            classification="corporativa",
            reason="A pergunta está relacionada aos módulos e documentos do corpus.",
        )
    )

    decision = classify_question_scope(question, CATALOG, llm=fake_llm)

    assert decision.is_in_scope is True
    if "VendeFácil" not in question:
        assert fake_llm.schema is ScopeDecision
        assert fake_llm.method == "json_schema"
        assert fake_llm.strict is True


@pytest.mark.parametrize(
    "question",
    [
        "Agora gostaria de saber sobre o sistema Venda Fácil.",
        "Qual é a política da VendeFácil para reembolso de cursos?",
        "Qual é a política de home office para a equipe de Engenharia?",
    ],
)
def test_accepts_explicit_corporate_context_without_llm(
    question: str,
) -> None:
    fake_llm = FakeChatModel(
        ScopeDecision(
            classification="fora_de_escopo",
            reason="Resposta incorreta que não deve ser utilizada.",
        )
    )

    decision = classify_question_scope(question, CATALOG, llm=fake_llm)

    assert decision.is_in_scope is True
    assert fake_llm.schema is None


@pytest.mark.parametrize(
    "question",
    [
        "Quem descobriu o Brasil?",
        "Me escreva um poema.",
    ],
)
def test_refuses_clearly_out_of_scope_questions(question: str) -> None:
    fake_llm = FakeChatModel(
        ScopeDecision(
            classification="fora_de_escopo",
            reason="A pergunta solicita assunto sem relação com a VendeFácil.",
        )
    )

    decision = classify_question_scope(question, CATALOG, llm=fake_llm)
    response = build_out_of_scope_refusal(decision)

    assert decision.is_in_scope is False
    assert response.is_refusal is True
    assert response.confidence_level == "recusado"
    assert response.sources_used == []
    assert response.refusal_reason == "fora_de_escopo"


@pytest.mark.parametrize(
    "question",
    [
        "Como resolvo esse problema?",
        "O pagamento está errado, o que devo fazer?",
    ],
)
def test_treats_ambiguous_questions_conservatively(question: str) -> None:
    fake_llm = FakeChatModel(
        ScopeDecision(
            classification="ambigua",
            reason="Falta contexto para relacionar a pergunta à VendeFácil.",
        )
    )

    decision = classify_question_scope(question, CATALOG, llm=fake_llm)
    response = build_out_of_scope_refusal(decision)

    assert decision.classification == "ambigua"
    assert response.is_refusal is True
    assert response.refusal_reason == "fora_de_escopo"


def test_prompt_receives_dynamic_metadata_catalog_and_question() -> None:
    fake_llm = FakeChatModel(
        ScopeDecision(
            classification="corporativa",
            reason="Consulta do corpus.",
        )
    )

    classify_question_scope(
        "Quais manuais existem sobre estoque?",
        CATALOG,
        llm=fake_llm,
    )

    prompt_text = str(fake_llm.structured_model.messages)
    assert '- module: ["analytics", "estoque", "pay", "pdv"]' in prompt_text
    assert "Quais manuais existem sobre estoque?" in prompt_text


def test_does_not_build_scope_refusal_for_corporate_question() -> None:
    decision = ScopeDecision(
        classification="corporativa",
        reason="Consulta relacionada ao corpus.",
    )

    with pytest.raises(ValueError, match="corporativa"):
        build_out_of_scope_refusal(decision)

