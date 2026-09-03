"""
Esquema de Dados Pydantic para o Mini Desafio RAG VendeFácil.

Este arquivo define a estrutura estrita de resposta esperada do assistente RAG,
garantindo a rastreabilidade das fontes, fundamentação e tratamento de guardrails/recusas.

Esse é um esquema inicial para o assistente RAG VendeFácil.
Você pode alterar o esquema para atender às necessidades da sua aplicação.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SourceEvidence(BaseModel):
    """Trecho de evidência extraído das fontes recuperadas pelo RAG."""

    model_config = ConfigDict(extra="forbid")

    filepath: str = Field(
        description="Caminho relativo do arquivo de onde a informação foi extraída (ex: data/semi_structured/tickets.jsonl)"
    )
    chunk_id: str = Field(
        description="Identificador único do chunk que sustenta a resposta."
    )
    quotation: str = Field(
        max_length=500,
        description="Trecho exato do texto ou dado utilizado para fundamentar a afirmação."
    )


class QueryMetadataFilter(BaseModel):
    """Estrutura para extração automatizada de filtros de metadados a partir da pergunta do usuário."""
    state: Optional[str] = Field(
        default=None,
        description="Estado da federação de duas letras (ex: 'MG', 'SP', 'RJ', 'RS', 'PR')"
    )
    module: Optional[str] = Field(
        default=None,
        description="Módulo do sistema VendeFácil (ex: 'pdv', 'estoque', 'ecommerce', 'analytics', 'pay')"
    )
    customer_id: Optional[str] = Field(
        default=None,
        description="Identificador único do cliente se mencionado (ex: 'CUST001', 'CUST008')"
    )
    priority: Optional[str] = Field(
        default=None,
        description="Prioridade do ticket se aplicável (ex: 'Baixa', 'Média', 'Alta', 'Crítica')"
    )
    is_sensitive_query: bool = Field(
        default=False,
        description="Indica se a consulta solicita dados confidenciais (ex: salários, senhas, cartões, CPFs)"
    )


class RAGResponse(BaseModel):
    """Resposta estruturada final produzida pelo assistente VendeFácil RAG."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        description="Resposta em linguagem natural, clara, objetiva e estritamente fundamentada no contexto recuperado."
    )
    confidence_level: Literal["alta", "media", "baixa", "recusado"] = Field(
        description="Nível de confiança validado da resposta."
    )
    sources_used: List[SourceEvidence] = Field(
        default_factory=list,
        description="Lista de fontes e trechos específicos que comprovam a resposta gerada."
    )
    reasoning: str = Field(
        description="Breve explicação do raciocínio lógico utilizado para construir a resposta a partir do contexto."
    )
    is_refusal: bool = Field(
        default=False,
        description="True se o assistente recusou responder a pergunta devido a violação de LGPD/segurança ou pergunta fora do escopo."
    )
    refusal_reason: Literal[
        "lgpd",
        "fora_de_escopo",
        "sem_evidencia",
        None,
    ] = Field(
        default=None,
        description="Motivo tipado da recusa quando is_refusal for True."
    )

    @model_validator(mode="after")
    def validate_response_consistency(self) -> "RAGResponse":
        """Impede combinações contraditórias entre resposta e recusa."""
        if self.is_refusal:
            if self.confidence_level != "recusado":
                raise ValueError(
                    "Recusas devem possuir confidence_level='recusado'."
                )
            if self.sources_used:
                raise ValueError("Recusas não podem possuir sources_used.")
            if self.refusal_reason is None:
                raise ValueError("Recusas devem possuir refusal_reason.")
        else:
            if not self.sources_used:
                raise ValueError(
                    "Respostas normais devem possuir ao menos uma evidência."
                )
            if self.refusal_reason is not None:
                raise ValueError(
                    "Respostas normais não podem possuir refusal_reason."
                )

        return self
