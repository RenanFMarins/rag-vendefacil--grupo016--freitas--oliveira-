import logging

from src.retrieval.models import QueryAnalysis
from src.retrieval.models import QueryFilters
from src.retrieval.normalization import normalize_query_filters
from src.retrieval.query_analyzer import analyze_question
from src.retrieval.validation import validate_normalized_filters


CATALOG = {
    "doc_type": ["ata", "email", "manual", "ticket"],
    "sensitivity": ["interno", "publico"],
    "state": ["MG", "SP"],
    "module": ["analytics", "ecommerce", "estoque", "pay", "pdv"],
    "plan": ["Basic", "Enterprise", "Pro"],
    "priority": ["Alta", "Baixa"],
    "status": ["Aberto", "Resolvido"],
}


class FakeStructuredModel:
    def __init__(self, response: QueryAnalysis) -> None:
        self.response = response
        self.messages = []

    def invoke(self, messages):
        self.messages = messages
        return self.response


class FakeChatModel:
    def __init__(self, response: QueryAnalysis) -> None:
        self.structured_model = FakeStructuredModel(response)
        self.schema = None
        self.method = None
        self.strict = None

    def with_structured_output(self, schema, *, method, strict):
        self.schema = schema
        self.method = method
        self.strict = strict
        return self.structured_model


def test_analyzes_question_with_pydantic_structured_output() -> None:
    fake_llm = FakeChatModel(
        QueryAnalysis(
            query="tickets relacionados a estoque",
            filters=QueryFilters(
                doc_type="ticket",
                state="Minas Gerais",
                module="ESTOQUE",
            ),
        )
    )

    analysis = analyze_question(
        "Quais tickets de Minas Gerais são relacionados a estoque?",
        CATALOG,
        llm=fake_llm,
    )

    assert analysis.filters.model_dump(exclude_none=True) == {
        "doc_type": "ticket",
        "state": "MG",
        "module": "estoque",
    }
    assert fake_llm.schema is QueryAnalysis
    assert fake_llm.method == "json_schema"
    assert fake_llm.strict is True
    prompt_text = str(fake_llm.structured_model.messages)
    assert '- state: ["MG", "SP"]' in prompt_text
    assert "Quais tickets de Minas Gerais" in prompt_text


def test_keeps_filters_empty_when_llm_finds_no_clear_restriction() -> None:
    fake_llm = FakeChatModel(
        QueryAnalysis(query="problemas mais comuns de pagamento")
    )

    analysis = analyze_question(
        "Explique os problemas mais comuns de pagamento",
        CATALOG,
        llm=fake_llm,
    )

    assert analysis.filters.model_dump(exclude_none=True) == {}


def test_moves_invented_values_to_rejected_filters() -> None:
    analysis = QueryAnalysis(
        query="problemas financeiros",
        filters=QueryFilters(module="financeiro", priority="alta"),
    )

    normalized = normalize_query_filters(analysis.filters)
    validation = validate_normalized_filters(normalized, CATALOG)

    assert validation.valid_filters.model_dump(exclude_none=True) == {
        "priority": "Alta"
    }
    assert validation.invalid_filters[0].field == "module"
    assert validation.invalid_filters[0].value == "financeiro"


def test_canonicalizes_catalog_value_with_residual_edge_punctuation() -> None:
    analysis = QueryAnalysis(
        query="documentos internos sobre estoque",
        filters=QueryFilters(module="estoque},", sensitivity="interno},"),
    )

    normalized = normalize_query_filters(analysis.filters)
    validation = validate_normalized_filters(normalized, CATALOG)

    assert validation.valid_filters.model_dump(exclude_none=True) == {
        "module": "estoque",
        "sensitivity": "interno",
    }
    assert validation.invalid_filters == []


def test_debug_logs_original_question_query_and_filters(caplog) -> None:
    fake_llm = FakeChatModel(
        QueryAnalysis(
            query="tickets abertos",
            filters=QueryFilters(doc_type="ticket", status="Aberto"),
        )
    )

    with caplog.at_level(logging.INFO):
        analyze_question(
            "Quais tickets estão abertos?",
            CATALOG,
            llm=fake_llm,
            debug=True,
        )

    assert "Pergunta original: Quais tickets estão abertos?" in caplog.text
    assert "Query semântica: tickets abertos" in caplog.text
    assert "Filtros normalizados:" in caplog.text
    assert "Filtros validados:" in caplog.text
    assert "Filtros ausentes:" in caplog.text


def test_multi_document_question_keeps_entities_and_removes_scalar_type() -> None:
    question = (
        "O que consta sobre o Supermercado Boa Compra nos e-mails, "
        "tickets e reuniões da empresa?"
    )
    fake_llm = FakeChatModel(
        QueryAnalysis(
            query="informações sobre falha",
            filters=QueryFilters(doc_type="email", module="estoque"),
        )
    )

    analysis = analyze_question(question, CATALOG, llm=fake_llm)

    assert analysis.query == question
    assert analysis.filters.doc_type is None
    assert analysis.filters.module == "estoque"


def test_single_document_question_keeps_valid_type_filter() -> None:
    fake_llm = FakeChatModel(
        QueryAnalysis(
            query="tickets de estoque",
            filters=QueryFilters(doc_type="ticket", module="estoque"),
        )
    )

    analysis = analyze_question(
        "Quais tickets existem sobre estoque?",
        CATALOG,
        llm=fake_llm,
    )

    assert analysis.query == "tickets de estoque"
    assert analysis.filters.doc_type == "ticket"


def test_does_not_force_ambiguous_business_entity_as_document_type() -> None:
    catalog = {**CATALOG, "doc_type": [*CATALOG["doc_type"], "customer"]}
    fake_llm = FakeChatModel(
        QueryAnalysis(
            query="clientes de Minas Gerais",
            filters=QueryFilters(state="MG"),
        )
    )

    analysis = analyze_question(
        "Quais clientes existem em Minas Gerais?",
        catalog,
        llm=fake_llm,
    )

    assert analysis.filters.doc_type is None


def test_singular_plan_is_not_treated_as_product_document_type() -> None:
    catalog = {**CATALOG, "doc_type": [*CATALOG["doc_type"], "product"]}
    fake_llm = FakeChatModel(
        QueryAnalysis(query="cancelamento do plano anual")
    )

    analysis = analyze_question(
        "Qual é o prazo de cancelamento do plano anual?",
        catalog,
        llm=fake_llm,
    )

    assert analysis.filters.doc_type is None


def test_validates_commercial_plan_against_dynamic_catalog() -> None:
    catalog = {**CATALOG, "doc_type": [*CATALOG["doc_type"], "customer"]}
    fake_llm = FakeChatModel(
        QueryAnalysis(
            query="clientes e valor mensal",
            filters=QueryFilters(doc_type="customer", plan="enterprise"),
        )
    )

    analysis = analyze_question(
        "Quais clientes estão no plano Enterprise e qual o valor mensal?",
        catalog,
        llm=fake_llm,
    )

    assert analysis.filters.model_dump(exclude_none=True) == {
        "doc_type": "customer",
        "plan": "Enterprise",
    }


def test_ticket_and_sla_question_removes_scalar_document_type() -> None:
    catalog = {
        **CATALOG,
        "doc_type": [*CATALOG["doc_type"], "policy"],
        "priority": [*CATALOG["priority"], "Crítica"],
    }
    question = (
        "Quais chamados de prioridade Crítica foram registrados e qual "
        "é o SLA desse nível?"
    )
    fake_llm = FakeChatModel(
        QueryAnalysis(
            query="chamados críticos e SLA",
            filters=QueryFilters(doc_type="ticket", priority="Crítica"),
        )
    )

    analysis = analyze_question(question, catalog, llm=fake_llm)

    assert analysis.query == question
    assert analysis.filters.doc_type is None
    assert analysis.filters.priority is None


def test_tef_context_canonicalizes_module_as_pay() -> None:
    fake_llm = FakeChatModel(
        QueryAnalysis(
            query="timeout de confirmação TEF no PDV",
            filters=QueryFilters(module="pdv"),
        )
    )

    analysis = analyze_question(
        "Por que o PDV exibe Timeout de confirmação TEF?",
        CATALOG,
        llm=fake_llm,
    )

    assert analysis.filters.module == "pay"


def test_tef_meeting_does_not_force_module_filter() -> None:
    fake_llm = FakeChatModel(
        QueryAnalysis(
            query="decisões da retrospectiva do incidente TEF",
            filters=QueryFilters(module="pay"),
        )
    )

    analysis = analyze_question(
        "Quais decisões foram aprovadas na reunião sobre o incidente TEF?",
        CATALOG,
        llm=fake_llm,
    )

    assert analysis.filters.module is None


def test_product_name_canonicalizes_vendefacil_loja_as_ecommerce() -> None:
    fake_llm = FakeChatModel(
        QueryAnalysis(
            query="regra de Safety Stock",
            filters=QueryFilters(module="estoque"),
        )
    )

    analysis = analyze_question(
        "Qual é a regra de Safety Stock no VendeFácil Loja?",
        CATALOG,
        llm=fake_llm,
    )

    assert analysis.filters.module == "ecommerce"


def test_multiple_product_names_do_not_force_a_single_module() -> None:
    fake_llm = FakeChatModel(
        QueryAnalysis(
            query="integração entre Loja e Estoque",
            filters=QueryFilters(),
        )
    )

    analysis = analyze_question(
        "Como VendeFácil Loja e VendeFácil Estoque se integram?",
        CATALOG,
        llm=fake_llm,
    )

    assert analysis.filters.module is None
