"""
Esquema de Dados Pydantic para o Mini Desafio RAG VendeFácil.

Este arquivo define a estrutura estrita de resposta esperada do assistente RAG,
garantindo a rastreabilidade das fontes, fundamentação e tratamento de guardrails/recusas.

Esse é um esquema inicial para o assistente RAG VendeFácil.
Você pode alterar o esquema para atender às necessidades da sua aplicação.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

class SourceEvidence(BaseModel):
    """Trecho de evidência extraído das fontes recuperadas pelo RAG."""
    filepath: str = Field(
        description="Caminho relativo do arquivo de onde a informação foi extraída (ex: data/semi_structured/tickets.jsonl)"
    )
    quotation: str = Field(
        description="Trecho exato do texto ou dado utilizado para fundamentar a afirmação."
    )
    doc_type: Optional[str] = Field(
        default=None,
        description="Tipo de documento (ex: policy, documentation, ticket, log, structured)"
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
    answer: str = Field(
        description="Resposta em linguagem natural, clara, objetiva e estritamente fundamentada no contexto recuperado."
    )
    confidence_level: str = Field(
        description="Nível de confiança da resposta com base nas evidências encontradas: 'Alta', 'Média', 'Baixa' ou 'Recusado'."
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
    refusal_reason: Optional[str] = Field(
        default=None,
        description="Motivo da recusa caso is_refusal seja True (ex: 'LGPD_PROTECTION', 'OUT_OF_DOMAIN', 'CREDENTIAL_PROTECTION')."
    )
