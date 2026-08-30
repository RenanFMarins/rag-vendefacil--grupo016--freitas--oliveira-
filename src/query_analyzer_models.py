from typing import Annotated

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import StringConstraints


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class QueryFilters(BaseModel):

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    doc_type: NonEmptyText | None = Field(
        default=None,
        description="Natureza do documento, como ticket, manual ou policy.",
    )
    state: NonEmptyText | None = Field(
        default=None,
        description="Sigla de duas letras de um estado brasileiro.",
    )
    module: NonEmptyText | None = Field(
        default=None,
        description="Módulo do VendeFácil mencionado na pergunta.",
    )
    priority: NonEmptyText | None = Field(
        default=None,
        description="Prioridade de ticket explicitamente solicitada.",
    )
    status: NonEmptyText | None = Field(
        default=None,
        description="Status de ticket explicitamente solicitado.",
    )
    category: NonEmptyText | None = Field(
        default=None,
        description="Categoria de ticket explicitamente solicitada.",
    )
    sentiment: NonEmptyText | None = Field(
        default=None,
        description="Sentimento do cliente explicitamente solicitado.",
    )
    sensitivity: NonEmptyText | None = Field(
        default=None,
        description="Classificação de acesso explicitamente solicitada.",
    )
    customer_id: NonEmptyText | None = Field(
        default=None,
        description="Identificador exato de cliente mencionado na pergunta.",
    )
    ticket_id: NonEmptyText | None = Field(
        default=None,
        description="Identificador exato de ticket mencionado na pergunta.",
    )


class RejectedFilter(BaseModel):

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    field: NonEmptyText = Field(
        description="Nome do campo solicitado ou inferido.",
    )
    value: NonEmptyText | None = Field(
        default=None,
        description="Valor rejeitado, quando houver um valor identificável.",
    )
    reason: NonEmptyText = Field(
        description="Motivo objetivo pelo qual o filtro foi rejeitado.",
    )


class QueryAnalysis(BaseModel):

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: NonEmptyText = Field(
        description="Consulta semântica limpa, preservando a intenção da pergunta.",
    )
    filters: QueryFilters = Field(
        default_factory=QueryFilters,
        description="Somente restrições claras extraídas da pergunta.",
    )
    rejected_filters: list[RejectedFilter] = Field(
        default_factory=list,
        description="Restrições inválidas, ambíguas ou fora do catálogo.",
    )
