import json
import logging
import os
import re
from collections.abc import Mapping
from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import CHAT_MODEL_TEMPERATURE
from src.config import DEFAULT_CHAT_MODEL
from src.config import OPENAI_API_KEY_ENV
from src.config import OPENAI_QUERY_ANALYZER_MODEL_ENV
from src.config import load_project_environment
from src.retrieval.models import QueryAnalysis
from src.retrieval.normalization import normalize_query_filters
from src.retrieval.normalization import normalization_key
from src.retrieval.validation import validate_normalized_filters


LOGGER = logging.getLogger(__name__)

DOCUMENT_TYPE_TERMS = {
    "ticket": ("ticket", "tickets", "chamado", "chamados"),
    "email": ("email", "emails", "e mail", "e mails"),
    "ata": ("ata", "atas", "reuniao", "reunioes"),
    "manual": ("manual", "manuais"),
    "policy": ("politica", "politicas", "sla"),
    "log": ("log", "logs"),
}
TEF_CONTEXT_TERMS = ("tef", "pinpad", "pay 504", "porta 6090", "maquininha")
PRODUCT_MODULE_TERMS = {
    "analytics": ("vendefacil analytics",),
    "ecommerce": ("vendefacil loja",),
    "estoque": ("vendefacil estoque",),
    "pay": ("vendefacil pay",),
    "pdv": ("vendefacil pdv",),
}

SYSTEM_PROMPT = """
Você é o Query Analyzer da base de conhecimento VendeFácil.
Sua única tarefa é interpretar a pergunta; não responda à pergunta e não
faça busca em documentos.

Regras:
- Produza uma query semântica curta que preserve o significado principal.
- Extraia somente restrições claramente expressas pelo usuário.
- Use apenas campos e valores presentes no catálogo fornecido.
- Copie para os filtros a grafia canônica exibida no catálogo.
- Nunca invente, traduza ou complete um valor ausente no catálogo.
- Se uma restrição explícita não existir no catálogo, registre-a como
  filtro rejeitado e explique o motivo.
- Se a pergunta não trouxer uma restrição clara, deixe os filtros vazios.
- Uma palavra pertencente ao assunto não é automaticamente um filtro. Por
  exemplo, "problemas de pagamento" descreve o assunto; não implica o filtro
  module=pay sem menção clara ao módulo ou produto.
- Nomes de tipos de documento são restrições explícitas. Assim, "tickets"
  seleciona doc_type=ticket quando esse valor existir no catálogo.
- Cliente, loja, produto, plano, venda, funcionário e colaborador podem ser
  apenas entidades ou assuntos da pergunta. Só use doc_type para esses termos
  quando o usuário pedir claramente os respectivos registros. Por exemplo,
  "erro entre lojas" não seleciona doc_type=store e "do cliente X" não
  seleciona doc_type=customer.
- Se a pergunta solicitar dois ou mais tipos, como "e-mails, tickets e
  reuniões", não selecione um único doc_type: deixe esse filtro vazio para
  permitir busca em todos os tipos pedidos.
- Preserve na query nomes próprios, customer_id, ticket_id e códigos exatos.
- "de Minas Gerais" seleciona state=MG; "módulo de estoque" seleciona
  module=estoque; "plano Enterprise" seleciona plan=Enterprise; "alta
  prioridade" seleciona priority=Alta; "abertos" seleciona status=Aberto;
  e "documentos internos" seleciona sensitivity=interno, desde que os
  valores estejam no catálogo.
- Antes de finalizar, revise separadamente doc_type, state, module, priority,
  plan, status, category, sentiment e sensitivity. Para cada restrição
  explícita, copie exatamente um valor do catálogo, sem pontuação adicional.
- A query semântica nunca pode ficar vazia depois da retirada das restrições.

Exemplos de decisão:
- Para "Quais tickets de Minas Gerais estão relacionados ao módulo de
  estoque?", extraia doc_type=ticket, state=MG e module=estoque.
- Para "Mostre tickets de alta prioridade", extraia doc_type=ticket e
  priority=Alta. Um termo pode permanecer na query e também ser um filtro.
- Para "Quais informações existem sobre problemas de pagamento?", não extraia
  nem rejeite filtros: pagamento é apenas o assunto da busca.
- Para "Quero documentos internos", extraia sensitivity=interno.
- Para "Quais clientes estão no plano Enterprise?", extraia
  doc_type=customer e plan=Enterprise.
- Para "Qual é o prazo de cancelamento de planos?", planos é o assunto;
  não selecione doc_type=product.
- Para "Quais chamados críticos e qual o SLA desse nível?", não restrinja
  doc_type, pois a resposta exige tickets e a política de SLA.
- Em incidentes com TEF, maquininha, Pinpad, PAY-504 ou porta 6090, use
  module=pay mesmo quando a mensagem de erro aparecer na tela do PDV.
- Em "decisões aprovadas na reunião", aprovada qualifica a decisão e não
  representa o campo status; não extraia status nesse caso.
- Para "O que consta sobre o Supermercado Boa Compra nos e-mails, tickets e
  reuniões?", mantenha "Supermercado Boa Compra" na query e não extraia
  doc_type, pois há três tipos solicitados.
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


def create_chat_model() -> ChatOpenAI:
    """Cria o modelo configurado para interpretar as perguntas."""
    load_project_environment()
    api_key = os.getenv(OPENAI_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"A variável {OPENAI_API_KEY_ENV} não foi configurada."
        )

    model = os.getenv(
        OPENAI_QUERY_ANALYZER_MODEL_ENV,
        DEFAULT_CHAT_MODEL,
    )
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=CHAT_MODEL_TEMPERATURE,
    )


def format_metadata_catalog(
    metadata_catalog: Mapping[str, Sequence[object]],
) -> str:
    """Formata o catálogo com um campo por linha para o prompt."""
    return "\n".join(
        f"- {field}: "
        + json.dumps(list(values), ensure_ascii=False)
        for field, values in sorted(metadata_catalog.items())
    )


def requested_document_types(question: str) -> frozenset[str]:
    """Identifica tipos explicitamente pedidos, sem inferir pelo assunto."""
    normalized = re.sub(
        r"[^a-z0-9]+",
        " ",
        normalization_key(question),
    )
    padded = f" {normalized} "
    return frozenset(
        doc_type
        for doc_type, terms in DOCUMENT_TYPE_TERMS.items()
        if any(f" {term} " in padded for term in terms)
    )


def _apply_multi_document_policy(
    question: str,
    analysis: QueryAnalysis,
) -> QueryAnalysis:
    requested_types = requested_document_types(question)
    if len(requested_types) < 2:
        return analysis

    filter_updates: dict[str, object] = {"doc_type": None}
    if {"ticket", "policy"} <= requested_types:
        filter_updates.update({"priority": None, "status": None})
    filters = analysis.filters.model_copy(update=filter_updates)
    return analysis.model_copy(
        update={
            "query": question,
            "filters": filters,
        }
    )


def _apply_module_context_policy(
    question: str,
    analysis: QueryAnalysis,
    metadata_catalog: Mapping[str, Sequence[object]],
) -> QueryAnalysis:
    normalized_question = normalization_key(question).replace("-", " ")
    available_modules = metadata_catalog.get("module", ())
    asks_for_meeting = "ata" in requested_document_types(question)
    has_pay_context = any(
        term in normalized_question for term in TEF_CONTEXT_TERMS
    )
    if asks_for_meeting and has_pay_context:
        filters = analysis.filters.model_copy(update={"module": None})
        return analysis.model_copy(update={"filters": filters})
    if "pay" not in available_modules or not has_pay_context:
        return analysis

    filters = analysis.filters.model_copy(update={"module": "pay"})
    return analysis.model_copy(update={"filters": filters})


def _apply_product_module_policy(
    question: str,
    analysis: QueryAnalysis,
    metadata_catalog: Mapping[str, Sequence[object]],
) -> QueryAnalysis:
    """Converte nomes comerciais inequívocos no módulo do catálogo."""
    normalized_question = normalization_key(question).replace("-", " ")
    available_modules = set(metadata_catalog.get("module", ()))
    mentioned_modules = {
        module
        for module, product_terms in PRODUCT_MODULE_TERMS.items()
        if module in available_modules
        and any(term in normalized_question for term in product_terms)
    }
    if len(mentioned_modules) != 1:
        return analysis

    module = mentioned_modules.pop()
    filters = analysis.filters.model_copy(update={"module": module})
    return analysis.model_copy(update={"filters": filters})


def analyze_question(
    question: str,
    metadata_catalog: Mapping[str, Sequence[object]],
    *,
    llm: BaseChatModel | None = None,
    debug: bool = False,
) -> QueryAnalysis:
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("A pergunta não pode estar vazia.")

    chat_model = llm or create_chat_model()
    structured_model = chat_model.with_structured_output(
        QueryAnalysis,
        method="json_schema",
        strict=True,
    )
    messages = PROMPT.format_messages(
        catalog=format_metadata_catalog(metadata_catalog),
        question=normalized_question,
    )
    raw_analysis = structured_model.invoke(messages)
    raw_validated_analysis = QueryAnalysis.model_validate(raw_analysis)
    normalized_analysis = raw_validated_analysis.model_copy(
        update={
            "filters": normalize_query_filters(raw_validated_analysis.filters)
        }
    )
    validation = validate_normalized_filters(
        normalized_analysis.filters,
        metadata_catalog,
    )
    validated_analysis = QueryAnalysis(
        query=normalized_analysis.query,
        filters=validation.valid_filters,
        rejected_filters=[
            *normalized_analysis.rejected_filters,
            *validation.invalid_filters,
        ],
    )
    validated_analysis = _apply_multi_document_policy(
        normalized_question,
        validated_analysis,
    )
    validated_analysis = _apply_product_module_policy(
        normalized_question,
        validated_analysis,
        metadata_catalog,
    )
    validated_analysis = _apply_module_context_policy(
        normalized_question,
        validated_analysis,
        metadata_catalog,
    )

    if debug:
        LOGGER.info("Pergunta original: %s", normalized_question)
        LOGGER.info("Query semântica: %s", validated_analysis.query)
        LOGGER.info(
            "Filtros normalizados: %s",
            normalized_analysis.filters.model_dump(exclude_none=True),
        )
        LOGGER.info(
            "Filtros validados: %s",
            validated_analysis.filters.model_dump(exclude_none=True),
        )
        LOGGER.info("Filtros ausentes: %s", validation.absent_fields)
        if validated_analysis.rejected_filters:
            LOGGER.info(
                "Filtros rejeitados: %s",
                [
                    item.model_dump(exclude_none=True)
                    for item in validated_analysis.rejected_filters
                ],
            )

    return validated_analysis
