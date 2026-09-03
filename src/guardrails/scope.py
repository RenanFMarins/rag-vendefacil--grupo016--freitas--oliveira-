import logging
import os
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from pydantic import ConfigDict

from src.config import CHAT_MODEL_TEMPERATURE
from src.config import DEFAULT_CHAT_MODEL
from src.config import OPENAI_API_KEY_ENV
from src.config import OPENAI_SCOPE_MODEL_ENV
from src.config import load_project_environment
from src.retrieval.query_analyzer import format_metadata_catalog
from starter.schema import RAGResponse


LOGGER = logging.getLogger(__name__)

ScopeClassification = Literal[
    "corporativa",
    "fora_de_escopo",
    "ambigua",
]

BRAND_ALIASES = (
    "vendefacil",
    "vende facil",
    "venda facil",
)
INTERNAL_POLICY_TERMS = (
    "colaborador",
    "equipe",
    "funcionario",
    "home office",
)


class ScopeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    classification: ScopeClassification
    reason: str

    @property
    def is_in_scope(self) -> bool:
        return self.classification == "corporativa"


SYSTEM_PROMPT = """
Você é um classificador de escopo da base de conhecimento VendeFácil.
Não responda à pergunta e não use conhecimento externo para completá-la.

Classifique como:
- corporativa: pergunta claramente relacionada à operação da VendeFácil
  ou aos tipos, módulos e valores presentes no catálogo do corpus;
- fora_de_escopo: conhecimento geral, criação literária, entretenimento,
  aconselhamento ou qualquer assunto sem relação com a VendeFácil;
- ambigua: pode ter relação corporativa, mas a pergunta não fornece contexto
  suficiente para afirmar isso com segurança.

O escopo corporativo inclui suporte, tickets, clientes, lojas, produtos,
políticas, manuais, reuniões e os módulos PDV, Estoque, Ecommerce, Analytics
e Pay. Não considere uma pergunta corporativa apenas porque ela usa palavras
genéricas como sistema, problema, pagamento ou produto.

Na dúvida entre corporativa e ambigua, escolha ambigua. Na dúvida entre
ambigua e fora_de_escopo, escolha fora_de_escopo. A razão deve explicar apenas
a classificação, sem tentar responder à pergunta.

Exemplos obrigatórios de decisão:
- "Quais tickets de estoque estão abertos?" -> corporativa;
- "Como realizar uma sangria no VendeFácil PDV?" -> corporativa;
- "Qual é a política de home office da equipe de Engenharia?" ->
  corporativa;
- "Qual é a política da VendeFácil para reembolso de cursos?" ->
  corporativa;
- "Quem descobriu o Brasil?" -> fora_de_escopo;
- "Me escreva um poema." -> fora_de_escopo;
- "Como resolvo esse problema?" -> ambigua;
- "O pagamento está errado, o que devo fazer?" -> ambigua, pois pagamento
  sem VendeFácil, Pay, ticket, cliente ou outro contexto do corpus é genérico.
""".strip()

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "<metadata_catalog>\n{catalog}\n</metadata_catalog>\n\n"
            "<question>\n{question}\n</question>",
        ),
    ]
)


def create_scope_model() -> ChatOpenAI:
    load_project_environment()
    api_key = os.getenv(OPENAI_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"A variável {OPENAI_API_KEY_ENV} não foi configurada."
        )

    model = os.getenv(OPENAI_SCOPE_MODEL_ENV, DEFAULT_CHAT_MODEL)
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=CHAT_MODEL_TEMPERATURE,
    )


def _normalize_scope_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )


def _deterministic_corporate_decision(
    question: str,
) -> ScopeDecision | None:
    """Aceita somente sinais corporativos explícitos de alta precisão."""
    normalized = _normalize_scope_text(question)
    if any(alias in normalized for alias in BRAND_ALIASES):
        return ScopeDecision(
            classification="corporativa",
            reason="A pergunta menciona explicitamente a VendeFácil.",
        )

    if "politica" in normalized and any(
        term in normalized for term in INTERNAL_POLICY_TERMS
    ):
        return ScopeDecision(
            classification="corporativa",
            reason=(
                "A pergunta trata explicitamente de uma política interna "
                "e de seu contexto organizacional."
            ),
        )

    return None


def classify_question_scope(
    question: str,
    metadata_catalog: Mapping[str, Sequence[object]],
    *,
    llm: BaseChatModel | None = None,
    debug: bool = False,
) -> ScopeDecision:
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("A pergunta não pode estar vazia.")

    deterministic_decision = _deterministic_corporate_decision(
        normalized_question
    )
    if deterministic_decision is not None:
        if debug:
            LOGGER.info(
                "Classificação de escopo: %s",
                deterministic_decision.classification,
            )
            LOGGER.info(
                "Motivo da classificação: %s",
                deterministic_decision.reason,
            )
        return deterministic_decision

    chat_model = llm or create_scope_model()
    structured_model = chat_model.with_structured_output(
        ScopeDecision,
        method="json_schema",
        strict=True,
    )
    messages = PROMPT.format_messages(
        catalog=format_metadata_catalog(metadata_catalog),
        question=normalized_question,
    )
    raw_decision = structured_model.invoke(messages)
    decision = ScopeDecision.model_validate(raw_decision)

    if debug:
        LOGGER.info("Classificação de escopo: %s", decision.classification)
        LOGGER.info("Motivo da classificação: %s", decision.reason)

    return decision


def build_out_of_scope_refusal(decision: ScopeDecision) -> RAGResponse:
    if decision.is_in_scope:
        raise ValueError(
            "Uma pergunta corporativa não deve gerar recusa de escopo."
        )

    return RAGResponse(
        answer=(
            "Não posso responder porque a pergunta não está claramente "
            "relacionada à operação da VendeFácil ou ao corpus disponível."
        ),
        confidence_level="recusado",
        sources_used=[],
        reasoning="A pergunta foi classificada como fora do escopo corporativo.",
        is_refusal=True,
        refusal_reason="fora_de_escopo",
    )

