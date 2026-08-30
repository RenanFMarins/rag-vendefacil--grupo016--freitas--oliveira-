from collections.abc import Iterable
from typing import Any

from langchain_core.documents import Document


CLOSED_VOCABULARY_FIELDS = (
    "doc_type",
    "sensitivity",
    "state",
    "module",
    "priority",
    "status",
    "category",
    "sentiment",
)


def _deterministic_sort_key(value: Any) -> tuple[str, str]:
    """Cria uma chave estável sem alterar o tipo ou o valor original."""
    return type(value).__name__, str(value).casefold()


def build_metadata_catalog(
    documents: Iterable[Document],
    fields: Iterable[str] = CLOSED_VOCABULARY_FIELDS,
) -> dict[str, list[Any]]:
    
    selected_fields = tuple(dict.fromkeys(fields))
    discovered_values: dict[str, set[Any]] = {
        field: set() for field in selected_fields
    }

    for document in documents:
        for field in selected_fields:
            value = document.metadata.get(field)
            if value is not None:
                discovered_values[field].add(value)

    return {
        field: sorted(values, key=_deterministic_sort_key)
        for field, values in discovered_values.items()
        if values
    }
