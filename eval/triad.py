"""Métricas da RAG Triad para o benchmark VendeFácil."""

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from eval.judge_prompt import JUDGE_SYSTEM_PROMPT
from eval.judge_prompt import format_triad_judge_input
from src.config import CHAT_MODEL_TEMPERATURE
from src.config import DEFAULT_CHAT_MODEL
from src.config import OPENAI_API_KEY_ENV
from src.config import OPENAI_JUDGE_MODEL_ENV
from src.config import load_project_environment
from src.pipeline import RetrievedChunk
from starter.schema import RAGResponse


LOGGER = logging.getLogger(__name__)


class TriadJudgeDraft(BaseModel):
    """Saída validada do LLM-as-a-Judge."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer_relevance: float = Field(ge=0, le=1)
    answer_relevance_reason: str = Field(min_length=1, max_length=300)
    groundedness: float = Field(ge=0, le=1)
    groundedness_reason: str = Field(min_length=1, max_length=300)


class TriadEvaluation(BaseModel):
    """Scores persistíveis sem resposta, citações ou justificativas do juiz."""

    model_config = ConfigDict(extra="forbid")

    context_relevance: float | None = Field(default=None, ge=0, le=1)
    context_relevance_basis: Literal[
        "chunk_id", "source_file", "not_applicable"
    ]
    answer_relevance: float | None = Field(default=None, ge=0, le=1)
    groundedness: float | None = Field(default=None, ge=0, le=1)
    judge_status: Literal[
        "not_requested", "not_applicable", "passed", "failed"
    ]
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_sources: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class JudgeConfig:
    max_attempts: int = 2

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts deve ser maior que zero.")


def create_judge_model() -> ChatOpenAI:
    """Cria o modelo isolado usado somente como avaliador."""
    load_project_environment()
    api_key = os.getenv(OPENAI_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"A variável {OPENAI_API_KEY_ENV} não foi configurada."
        )
    model = os.getenv(OPENAI_JUDGE_MODEL_ENV, DEFAULT_CHAT_MODEL)
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=CHAT_MODEL_TEMPERATURE,
    )


def _source_groups(expected_sources: Sequence[str]) -> dict[str, set[str]]:
    """Agrupa extensões alternativas do mesmo documento pelo nome-base."""
    groups: dict[str, set[str]] = {}
    for source in expected_sources:
        path = Path(source)
        groups.setdefault(path.stem.casefold(), set()).add(
            path.name.casefold()
        )
    return groups


def calculate_context_relevance(
    retrieved_chunks: Sequence[RetrievedChunk],
    *,
    expected_chunk_ids: Sequence[str] = (),
    expected_sources: Sequence[str] = (),
    is_applicable: bool = True,
) -> tuple[float | None, Literal["chunk_id", "source_file", "not_applicable"]]:
    """Calcula recall do contexto usando o gabarito mais preciso disponível."""
    if not is_applicable:
        return None, "not_applicable"

    if expected_chunk_ids:
        expected = set(expected_chunk_ids)
        retrieved = {chunk.chunk_id for chunk in retrieved_chunks}
        return len(expected & retrieved) / len(expected), "chunk_id"

    if expected_sources:
        groups = _source_groups(expected_sources)
        retrieved_names = {
            Path(chunk.source_file).name.casefold()
            for chunk in retrieved_chunks
        }
        matched_groups = sum(
            bool(alternatives & retrieved_names)
            for alternatives in groups.values()
        )
        return matched_groups / len(groups), "source_file"

    return None, "not_applicable"


def _judge_messages(
    *,
    question: str,
    response: RAGResponse,
    expected_behavior: str,
    ground_truth_answer: str | None,
    key_points: Sequence[str],
) -> list[SystemMessage | HumanMessage]:
    evidence = [source.quotation for source in response.sources_used]
    judge_input = format_triad_judge_input(
        question=question,
        answer=response.answer,
        is_refusal=response.is_refusal,
        expected_behavior=expected_behavior,
        ground_truth_answer=ground_truth_answer,
        key_points=key_points,
        evidence=evidence,
    )
    return [
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=judge_input),
    ]


def invoke_triad_judge(
    *,
    question: str,
    response: RAGResponse,
    expected_behavior: str,
    ground_truth_answer: str | None,
    key_points: Sequence[str],
    llm: BaseChatModel,
    config: JudgeConfig | None = None,
) -> TriadJudgeDraft | None:
    """Executa structured output com retry limitado e falha controlada."""
    judge_config = config or JudgeConfig()
    structured_model = llm.with_structured_output(
        TriadJudgeDraft,
        method="json_schema",
        strict=True,
    )
    messages = _judge_messages(
        question=question,
        response=response,
        expected_behavior=expected_behavior,
        ground_truth_answer=ground_truth_answer,
        key_points=key_points,
    )
    correction = HumanMessage(
        content=(
            "A saída anterior foi inválida. Retorne somente o objeto "
            "estruturado solicitado, com notas numéricas entre 0 e 1 e "
            "justificativas curtas."
        )
    )

    for attempt in range(1, judge_config.max_attempts + 1):
        try:
            raw_result = structured_model.invoke(messages)
            return TriadJudgeDraft.model_validate(raw_result)
        except (ValidationError, OutputParserException):
            LOGGER.warning(
                "Judge retornou structured output inválido na tentativa %d/%d.",
                attempt,
                judge_config.max_attempts,
            )
            if attempt < judge_config.max_attempts:
                messages = [*messages, correction]
    return None


def build_triad_evaluation(
    *,
    response: RAGResponse,
    retrieved_chunks: Sequence[RetrievedChunk],
    expected_behavior: str,
    expected_chunk_ids: Sequence[str] = (),
    expected_sources: Sequence[str] = (),
    judge_result: TriadJudgeDraft | None = None,
    judge_requested: bool = False,
    judge_applicable: bool = True,
) -> TriadEvaluation:
    """Combina a métrica determinística e os scores opcionais do judge."""
    context_score, context_basis = calculate_context_relevance(
        retrieved_chunks,
        expected_chunk_ids=expected_chunk_ids,
        expected_sources=expected_sources,
        is_applicable=expected_behavior != "refusal",
    )
    if not judge_applicable:
        judge_status = "not_applicable"
        answer_relevance = None
        groundedness = None
    elif judge_result is not None:
        judge_status = "passed"
        answer_relevance = judge_result.answer_relevance
        groundedness = (
            None if response.is_refusal else judge_result.groundedness
        )
    else:
        judge_status = "failed" if judge_requested else "not_requested"
        answer_relevance = None
        groundedness = None

    return TriadEvaluation(
        context_relevance=context_score,
        context_relevance_basis=context_basis,
        answer_relevance=answer_relevance,
        groundedness=groundedness,
        judge_status=judge_status,
        retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieved_chunks],
        retrieved_sources=[chunk.source_file for chunk in retrieved_chunks],
    )

