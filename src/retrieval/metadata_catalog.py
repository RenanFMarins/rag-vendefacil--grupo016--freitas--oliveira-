from collections.abc import Iterable
from typing import Any

from langchain_core.documents import Document


CLOSED_VOCABULARY_FIELDS = (
    "doc_type",
    "sensitivity",
    "state",
    "module",
    "plan",
    "priority",
    "status",
    "category",
    "sentiment",
)


def _deterministic_sort_key(value: Any) -> tuple[str, str]:
    """Cria uma chave estável sem alterar o tipo ou o valor original."""
    return type(value).__name__, str(value).casefold()


def _metadata_values(value: Any) -> tuple[Any, ...]:
    """Expande metadados multivalorados sem alterar o Document original."""
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    return (value,)


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
                discovered_values[field].update(
                    item for item in _metadata_values(value) if item is not None
                )

    return {
        field: sorted(values, key=_deterministic_sort_key)
        for field, values in discovered_values.items()
        if values
    }

