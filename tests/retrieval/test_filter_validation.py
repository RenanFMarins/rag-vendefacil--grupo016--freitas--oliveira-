from src.retrieval.models import QueryFilters
from src.retrieval.normalization import normalize_query_filters
from src.retrieval.validation import validate_normalized_filters


CATALOG = {
    "doc_type": ["manual", "ticket"],
    "sensitivity": ["interno", "publico"],
    "state": ["MG", "SP"],
    "module": ["estoque", "pay"],
    "plan": ["Basic", "Enterprise", "Pro"],
    "priority": ["Alta", "Baixa", "Crítica"],
    "status": ["Aberto", "Em Andamento", "Resolvido"],
}


def normalize_and_validate(filters: QueryFilters):
    normalized = normalize_query_filters(filters)
    return validate_normalized_filters(normalized, CATALOG)


def test_accepts_values_that_exist_in_catalog_after_normalization() -> None:
    result = normalize_and_validate(
        QueryFilters(
            doc_type="TICKET",
            state="Minas Gerais",
            module="ESTOQUE",
            plan="enterprise",
            priority="crítica",
            status=" em andamento ",
        )
    )

    assert result.valid_filters.model_dump(exclude_none=True) == {
        "doc_type": "ticket",
        "state": "MG",
        "module": "estoque",
        "plan": "Enterprise",
        "priority": "Crítica",
        "status": "Em Andamento",
    }
    assert result.invalid_filters == []


def test_rejects_partial_state_name_instead_of_accepting_silently() -> None:
    result = normalize_and_validate(QueryFilters(state="Minas"))

    assert result.valid_filters.model_dump(exclude_none=True) == {}
    assert len(result.invalid_filters) == 1
    assert result.invalid_filters[0].field == "state"
    assert result.invalid_filters[0].value == "Minas"
    assert "Valor não encontrado" in result.invalid_filters[0].reason


def test_rejects_value_that_does_not_exist_in_corpus_catalog() -> None:
    result = normalize_and_validate(QueryFilters(module="financeiro"))

    assert result.valid_filters.model_dump(exclude_none=True) == {}
    assert result.invalid_filters[0].field == "module"
    assert result.invalid_filters[0].value == "financeiro"


def test_rejects_filter_field_not_available_in_catalog() -> None:
    result = normalize_and_validate(QueryFilters(customer_id="CUST001"))

    assert result.valid_filters.model_dump(exclude_none=True) == {}
    assert result.invalid_filters[0].field == "customer_id"
    assert "Campo não disponível" in result.invalid_filters[0].reason


def test_identifies_filters_not_provided_by_query_analyzer_as_absent() -> None:
    result = normalize_and_validate(QueryFilters(state="MG"))

    assert "state" not in result.absent_fields
    assert "doc_type" in result.absent_fields
    assert "module" in result.absent_fields
    assert "priority" in result.absent_fields
    assert "customer_id" in result.absent_fields


def test_empty_filters_are_absent_and_never_invalid() -> None:
    result = normalize_and_validate(QueryFilters())

    assert result.valid_filters.model_dump(exclude_none=True) == {}
    assert result.invalid_filters == []
    assert result.absent_fields == list(QueryFilters.model_fields)

