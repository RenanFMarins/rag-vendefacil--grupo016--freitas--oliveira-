from src.retrieval.models import QueryFilters
from src.retrieval.normalization import normalize_module
from src.retrieval.normalization import normalize_query_filters
from src.retrieval.normalization import normalize_state


def test_normalizes_state_names_codes_case_accents_and_spaces() -> None:
    assert normalize_state("Minas Gerais") == "MG"
    assert normalize_state("mg") == "MG"
    assert normalize_state(" MG ") == "MG"
    assert normalize_state("são paulo") == "SP"
    assert normalize_state("ESPIRITO SANTO") == "ES"


def test_normalizes_module_case_spaces_and_required_aliases() -> None:
    assert normalize_module("Estoque") == "estoque"
    assert normalize_module("ESTOQUE") == "estoque"
    assert normalize_module("  E-commerce  ") == "ecommerce"
    assert normalize_module("VendeFácil Pay") == "pay"


def test_normalizes_all_filter_fields_without_modifying_original() -> None:
    original = QueryFilters(
        doc_type=" Ticket ",
        state=" Minas Gerais ",
        module=" ESTOQUE ",
        plan=" ENTERPRISE ",
        priority=" CRÍTICA ",
        status=" Em   Andamento ",
        category=" Sincronização / API ",
        sentiment=" Insatisfeito ",
        sensitivity=" INTERNO ",
        customer_id=" cust001 ",
        ticket_id=" tck-1001 ",
    )

    normalized = normalize_query_filters(original)

    assert normalized.model_dump(exclude_none=True) == {
        "doc_type": "ticket",
        "state": "MG",
        "module": "estoque",
        "plan": "enterprise",
        "priority": "critica",
        "status": "em andamento",
        "category": "sincronizacao / api",
        "sentiment": "insatisfeito",
        "sensitivity": "interno",
        "customer_id": "CUST001",
        "ticket_id": "TCK-1001",
    }
    assert original.state == "Minas Gerais"
    assert original.module == "ESTOQUE"
    assert original.priority == "CRÍTICA"


def test_keeps_unknown_full_state_available_for_later_validation() -> None:
    filters = QueryFilters(state="Estado Inventado")

    normalized = normalize_query_filters(filters)

    assert normalized.state == "Estado Inventado"

