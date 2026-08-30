import logging

from src.query_analyzer import analyze_question
from src.filter_normalization import normalize_query_filters
from src.filter_validation import validate_normalized_filters
from src.query_analyzer_models import QueryAnalysis
from src.query_analyzer_models import QueryFilters


CATALOG = {
    "doc_type": ["manual", "ticket"],
    "sensitivity": ["interno", "publico"],
    "state": ["MG", "SP"],
    "module": ["estoque", "pay"],
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
