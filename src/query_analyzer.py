import json
import logging
import os
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.filter_normalization import normalize_query_filters
from src.filter_validation import validate_normalized_filters
from src.query_analyzer_models import QueryAnalysis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)

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
- "de Minas Gerais" seleciona state=MG; "módulo de estoque" seleciona
  module=estoque; "alta prioridade" seleciona priority=Alta; "abertos"
  seleciona status=Aberto; e "documentos internos" seleciona
  sensitivity=interno, desde que os valores estejam no catálogo.
- Antes de finalizar, revise separadamente doc_type, state, module, priority,
  status, category, sentiment e sensitivity. Para cada restrição explícita,
  copie exatamente um valor do catálogo, sem pontuação adicional.
- A query semântica nunca pode ficar vazia depois da retirada das restrições.

Exemplos de decisão:
- Para "Quais tickets de Minas Gerais estão relacionados ao módulo de
  estoque?", extraia doc_type=ticket, state=MG e module=estoque.
- Para "Mostre tickets de alta prioridade", extraia doc_type=ticket e
  priority=Alta. Um termo pode permanecer na query e também ser um filtro.
- Para "Quais informações existem sobre problemas de pagamento?", não extraia
  nem rejeite filtros: pagamento é apenas o assunto da busca.
- Para "Quero documentos internos", extraia sensitivity=interno.
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
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("A variável OPENAI_API_KEY não foi configurada.")

    model = os.getenv("OPENAI_QUERY_ANALYZER_MODEL", "gpt-4o-mini")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=0,
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
