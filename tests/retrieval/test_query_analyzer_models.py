import pytest
from pydantic import ValidationError

from src.retrieval.models import QueryAnalysis
from src.retrieval.models import QueryFilters
from src.retrieval.models import RejectedFilter


def test_represents_semantic_query_with_explicit_filters() -> None:
    analysis = QueryAnalysis(
        query="tickets relacionados ao módulo de estoque",
        filters=QueryFilters(
            doc_type="ticket",
            state="mg",
            module="estoque",
            sensitivity="interno",
        ),
    )

    assert analysis.model_dump(exclude_none=True) == {
        "query": "tickets relacionados ao módulo de estoque",
        "filters": {
            "doc_type": "ticket",
            "state": "mg",
            "module": "estoque",
            "sensitivity": "interno",
        },
        "rejected_filters": [],
    }


def test_uses_empty_filters_when_question_has_no_clear_restriction() -> None:
    analysis = QueryAnalysis(query="problemas de pagamento")

    assert analysis.filters.model_dump(exclude_none=True) == {}
    assert analysis.rejected_filters == []


def test_represents_filter_rejected_by_future_catalog_validation() -> None:
    analysis = QueryAnalysis(
        query="problemas de pagamento",
        rejected_filters=[
            RejectedFilter(
                field="module",
                value="financeiro",
                reason="Valor não encontrado no catálogo de metadados.",
            )
        ],
    )

    assert analysis.filters.model_dump(exclude_none=True) == {}
    assert analysis.rejected_filters[0].field == "module"
    assert analysis.rejected_filters[0].value == "financeiro"


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "tickets", "filters": {"estado": "MG"}},
        {"query": "tickets", "campo_desconhecido": True},
    ],
)
def test_rejects_invalid_contract_payloads(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        QueryAnalysis.model_validate(payload)


def test_schema_preserves_identifiers_before_normalization() -> None:
    filters = QueryFilters(
        customer_id=" cust001 ",
        ticket_id=" tck-1001 ",
    )

    assert filters.customer_id == "cust001"
    assert filters.ticket_id == "tck-1001"
