
from collections.abc import Mapping
from collections.abc import Sequence

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from src.filter_normalization import normalization_key
from src.query_analyzer_models import QueryFilters
from src.query_analyzer_models import RejectedFilter


class FilterValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid_filters: QueryFilters = Field(default_factory=QueryFilters)
    absent_fields: list[str] = Field(default_factory=list)
    invalid_filters: list[RejectedFilter] = Field(default_factory=list)


def _catalog_values_by_key(
    values: Sequence[object],
) -> dict[str, object]:
    return {normalization_key(value): value for value in values}


def validate_normalized_filters(
    filters: QueryFilters,
    metadata_catalog: Mapping[str, Sequence[object]],
) -> FilterValidationResult:
    provided_filters = filters.model_dump(exclude_none=True)
    valid_values: dict[str, object] = {}
    absent_fields: list[str] = []
    invalid_filters: list[RejectedFilter] = []

    for field in QueryFilters.model_fields:
        if field not in provided_filters:
            absent_fields.append(field)
            continue

        value = provided_filters[field]
        if field not in metadata_catalog:
            invalid_filters.append(
                RejectedFilter(
                    field=field,
                    value=str(value),
                    reason="Campo não disponível no catálogo de metadados.",
                )
            )
            continue

        catalog_values = _catalog_values_by_key(metadata_catalog[field])
        canonical_value = catalog_values.get(normalization_key(value))
        if canonical_value is None:
            invalid_filters.append(
                RejectedFilter(
                    field=field,
                    value=str(value),
                    reason="Valor não encontrado no catálogo de metadados.",
                )
            )
            continue

        valid_values[field] = canonical_value

    return FilterValidationResult(
        valid_filters=QueryFilters.model_validate(valid_values),
        absent_fields=absent_fields,
        invalid_filters=invalid_filters,
    )
