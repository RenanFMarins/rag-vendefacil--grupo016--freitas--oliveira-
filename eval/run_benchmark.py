"""Executa o benchmark determinístico do pipeline RAG VendeFácil."""

import argparse
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ingestion.build._faiss import create_openai_embeddings
from ingestion.preview.sanity_faiss import load_indexes
from eval.triad import JudgeConfig
from eval.triad import TriadEvaluation
from eval.triad import build_triad_evaluation
from eval.triad import create_judge_model
from eval.triad import invoke_triad_judge
from src.config import PROJECT_ROOT
from src.guardrails.masking import mask_personal_data
from src.pipeline import RAGPipeline
from src.pipeline import RAGPipelineExecution
from src.retrieval.pipeline import HybridRetriever
from starter.schema import RAGResponse, SourceEvidence


LOGGER = logging.getLogger(__name__)
DEFAULT_CASES_PATH = (
    PROJECT_ROOT / "benchmark" / "questions_and_ground_truth.json"
)
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

ExpectedBehavior = Literal["answer", "masked_answer", "refusal"]
RefusalReason = Literal["lgpd", "fora_de_escopo", "sem_evidencia"]
ANSWER_RELEVANCE_PASS_THRESHOLD = 0.7
ANSWER_MAX_POINTS = 0.5
CITATION_MAX_POINTS = 0.3
CONSISTENCY_MAX_POINTS = 0.2


class BenchmarkCase(BaseModel):
    """Critérios objetivos esperados para uma pergunta do benchmark."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    category: str = Field(min_length=1)
    expected_behavior: ExpectedBehavior
    expected_sources: list[str] = Field(default_factory=list)
    reference_sources: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    source_match_mode: Literal["any", "all"] = "any"
    expected_refusal_reason: RefusalReason | None = None
    aggregate_group_size: int | None = Field(default=None, ge=1)
    expected_metadata: dict[str, Any] = Field(default_factory=dict)
    ground_truth_answer: str | None = None
    key_points_for_evaluation: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_expectations(self) -> "BenchmarkCase":
        if self.expected_behavior == "refusal":
            if self.expected_refusal_reason is None:
                raise ValueError(
                    "Caso de recusa deve informar expected_refusal_reason."
                )
            if self.expected_sources:
                raise ValueError(
                    "Caso de recusa não deve declarar expected_sources."
                )
        elif self.expected_refusal_reason is not None:
            raise ValueError(
                "Caso de resposta não deve informar expected_refusal_reason."
            )
        return self


class OfficialQuestion(BaseModel):
    """Pergunta no formato distribuído pelo professor."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_sources: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_metadata: dict[str, Any] = Field(default_factory=dict)
    ground_truth_answer: str = Field(min_length=1)
    key_points_for_evaluation: list[str] = Field(default_factory=list)


class OfficialBenchmarkSuite(BaseModel):
    """Envelope e metadados do benchmark oficial."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    benchmark_name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    questions: list[OfficialQuestion] = Field(min_length=1)


class QuestionScore(BaseModel):
    """Pontuação de 1,0 ponto definida pela rubrica oficial."""

    model_config = ConfigDict(extra="forbid")

    answer_correct: bool | None
    answer_points: float | None = Field(ge=0, le=ANSWER_MAX_POINTS)
    citation_correct: bool
    citation_points: float = Field(ge=0, le=CITATION_MAX_POINTS)
    consistency_correct: bool
    consistency_points: float = Field(ge=0, le=CONSISTENCY_MAX_POINTS)
    total_points: float | None = Field(default=None, ge=0, le=1)
    complete: bool

    @model_validator(mode="after")
    def validate_score_consistency(self) -> "QuestionScore":
        if self.complete:
            if self.answer_correct is None or self.answer_points is None:
                raise ValueError("Score completo exige avaliação da resposta.")
            expected_total = round(
                self.answer_points
                + self.citation_points
                + self.consistency_points,
                2,
            )
            if self.total_points != expected_total:
                raise ValueError("total_points não corresponde aos componentes.")
        elif self.total_points is not None:
            raise ValueError("Score incompleto não pode possuir total_points.")
        return self


class CaseEvaluation(BaseModel):
    """Resultado serializável de uma única pergunta."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    category: str
    status: Literal["passed", "failed", "error"]
    duration_ms: float = Field(ge=0)
    checks: dict[str, bool] = Field(default_factory=dict)
    response_summary: dict[str, Any] | None = None
    triad: TriadEvaluation | None = None
    score: QuestionScore | None = None
    error: dict[str, str] | None = None


@dataclass(frozen=True)
class BenchmarkArtifacts:
    """Caminhos dos artefatos gerados por uma execução."""

    results_json: Path
    summary_markdown: Path
    timestamped_json: Path


def _official_expected_behavior(
    question: OfficialQuestion,
) -> tuple[ExpectedBehavior, RefusalReason | None]:
    metadata = question.expected_metadata
    if metadata.get("out_of_domain") is True:
        return "refusal", "fora_de_escopo"
    if metadata.get("sensitive") is True:
        return "refusal", "lgpd"
    return "answer", None


def _official_source_match_mode(
    sources: Sequence[str],
) -> Literal["any", "all"]:
    """Usa fontes oficiais como alternativas para o check funcional.

    A cobertura do conjunto completo de fontes continua sendo medida
    separadamente por Context Relevance na RAG Triad.
    """
    del sources
    return "any"


def _case_from_official_question(
    question: OfficialQuestion,
) -> BenchmarkCase:
    expected_behavior, refusal_reason = _official_expected_behavior(question)
    expected_sources = (
        [] if expected_behavior == "refusal" else question.expected_sources
    )
    return BenchmarkCase(
        id=question.id,
        question=question.question,
        category=question.category,
        expected_behavior=expected_behavior,
        expected_sources=expected_sources,
        reference_sources=question.expected_sources,
        expected_chunk_ids=question.expected_chunk_ids,
        source_match_mode=_official_source_match_mode(
            question.expected_sources
        ),
        expected_refusal_reason=refusal_reason,
        expected_metadata=question.expected_metadata,
        ground_truth_answer=question.ground_truth_answer,
        key_points_for_evaluation=question.key_points_for_evaluation,
    )


def _validate_unique_case_ids(cases: Sequence[BenchmarkCase]) -> None:
    case_ids = [case.id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("O benchmark possui IDs duplicados.")


def load_cases(path: str | Path = DEFAULT_CASES_PATH) -> list[BenchmarkCase]:
    """Carrega o benchmark oficial ou a lista interna legada."""
    cases_path = Path(path)
    raw_data = json.loads(cases_path.read_text(encoding="utf-8"))
    if isinstance(raw_data, dict):
        suite = OfficialBenchmarkSuite.model_validate(raw_data)
        cases = [
            _case_from_official_question(question)
            for question in suite.questions
        ]
    elif isinstance(raw_data, list) and raw_data:
        cases = [
            BenchmarkCase.model_validate(raw_case) for raw_case in raw_data
        ]
    else:
        raise ValueError(
            "O benchmark deve conter o objeto oficial ou uma lista não vazia."
        )

    _validate_unique_case_ids(cases)
    return cases


def select_cases(
    cases: Sequence[BenchmarkCase],
    *,
    case_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[BenchmarkCase]:
    """Seleciona casos sem alterar a ordem pedida nem o arquivo oficial."""
    selected = list(cases)
    if case_ids:
        unique_ids = list(dict.fromkeys(case_ids))
        cases_by_id = {case.id: case for case in cases}
        unknown_ids = [case_id for case_id in unique_ids if case_id not in cases_by_id]
        if unknown_ids:
            names = ", ".join(unknown_ids)
            raise ValueError(f"IDs de benchmark desconhecidos: {names}.")
        selected = [cases_by_id[case_id] for case_id in unique_ids]

    if limit is not None:
        if limit < 1:
            raise ValueError("limit deve ser maior que zero.")
        selected = selected[:limit]
    return selected


def _source_match(case: BenchmarkCase, response: RAGResponse) -> bool:
    actual_sources = {
        Path(source.filepath).name for source in response.sources_used
    }
    expected_sources = {
        Path(source).name for source in case.expected_sources
    }
    if case.source_match_mode == "all":
        return expected_sources <= actual_sources
    return bool(expected_sources & actual_sources)


def _chunk_match(case: BenchmarkCase, response: RAGResponse) -> bool:
    actual_chunk_ids = {
        source.chunk_id for source in response.sources_used
    }
    expected_chunk_ids = set(case.expected_chunk_ids)
    if case.source_match_mode == "all":
        return expected_chunk_ids <= actual_chunk_ids
    return bool(expected_chunk_ids & actual_chunk_ids)


def _evidence_is_complete(source: SourceEvidence) -> bool:
    return bool(
        source.filepath.strip()
        and source.chunk_id.strip()
        and source.quotation.strip()
        and len(source.quotation) <= 500
    )


def _quotation_is_grounded(
    source: SourceEvidence,
    documents_by_id: Mapping[str, Document],
    *,
    masked: bool,
) -> bool:
    document = documents_by_id.get(source.chunk_id)
    if document is None:
        return False
    if document.metadata.get("source_file") != source.filepath:
        return False

    content = (
        mask_personal_data(document.page_content)
        if masked
        else document.page_content
    )
    return source.quotation in content


def _response_is_masked(response: RAGResponse) -> bool:
    texts = [
        response.answer,
        response.reasoning,
        *(source.quotation for source in response.sources_used),
    ]
    return all(mask_personal_data(text) == text for text in texts)


def _question_score(
    case: BenchmarkCase,
    response: RAGResponse,
    checks: Mapping[str, bool],
    triad: TriadEvaluation | None,
) -> QuestionScore:
    expected_refusal = case.expected_behavior == "refusal"

    if expected_refusal:
        answer_correct: bool | None = bool(
            checks.get("refusal_expected")
            and checks.get("expected_refusal_reason_match")
        )
    elif response.is_refusal:
        answer_correct = False
    elif triad is not None and triad.answer_relevance is not None:
        answer_correct = (
            triad.answer_relevance >= ANSWER_RELEVANCE_PASS_THRESHOLD
        )
        if case.expected_behavior == "masked_answer":
            answer_correct = bool(
                answer_correct and checks.get("masking_safe")
            )
    else:
        # Sem judge, não é correto fingir que uma resposta textual está
        # semanticamente certa apenas porque possui o schema esperado.
        answer_correct = None

    if expected_refusal:
        citation_correct = bool(checks.get("refusal_without_sources"))
    else:
        if case.expected_chunk_ids:
            expected_evidence_matches = checks.get(
                "expected_chunk_match", False
            )
        elif case.expected_sources:
            expected_evidence_matches = checks.get("source_match", False)
        else:
            expected_evidence_matches = True
        citation_correct = bool(
            checks.get("evidence_complete")
            and expected_evidence_matches
            and checks.get("quotation_grounded", True)
        )

    consistency_correct = bool(
        checks.get("refusal_expected")
        and checks.get("confidence_consistent")
    )
    answer_points = (
        None
        if answer_correct is None
        else ANSWER_MAX_POINTS if answer_correct else 0.0
    )
    citation_points = CITATION_MAX_POINTS if citation_correct else 0.0
    consistency_points = (
        CONSISTENCY_MAX_POINTS if consistency_correct else 0.0
    )
    complete = answer_points is not None
    total_points = (
        round(answer_points + citation_points + consistency_points, 2)
        if answer_points is not None
        else None
    )
    return QuestionScore(
        answer_correct=answer_correct,
        answer_points=answer_points,
        citation_correct=citation_correct,
        citation_points=citation_points,
        consistency_correct=consistency_correct,
        consistency_points=consistency_points,
        total_points=total_points,
        complete=complete,
    )


def evaluate_response(
    case: BenchmarkCase,
    response: RAGResponse | Mapping[str, Any],
    *,
    documents_by_id: Mapping[str, Document] | None = None,
    triad: TriadEvaluation | None = None,
    duration_ms: float = 0,
) -> CaseEvaluation:
    """Avalia somente regras estruturais, evidenciais e de guardrail."""
    try:
        validated = RAGResponse.model_validate(response)
    except ValidationError:
        return CaseEvaluation(
            id=case.id,
            question=case.question,
            category=case.category,
            status="failed",
            duration_ms=duration_ms,
            checks={"response_schema_valid": False},
            response_summary=None,
            triad=triad,
            score=QuestionScore(
                answer_correct=False,
                answer_points=0.0,
                citation_correct=False,
                citation_points=0.0,
                consistency_correct=False,
                consistency_points=0.0,
                total_points=0.0,
                complete=True,
            ),
        )

    expected_refusal = case.expected_behavior == "refusal"
    checks = {
        "response_schema_valid": True,
        "refusal_expected": validated.is_refusal == expected_refusal,
        "confidence_consistent": (
            validated.confidence_level == "recusado"
            if validated.is_refusal
            else validated.confidence_level in {"alta", "media", "baixa"}
        ),
    }

    if expected_refusal:
        checks.update(
            {
                "refusal_confidence": (
                    validated.confidence_level == "recusado"
                ),
                "refusal_without_sources": not validated.sources_used,
                "expected_refusal_reason_match": (
                    validated.refusal_reason
                    == case.expected_refusal_reason
                ),
            }
        )
    else:
        checks["evidence_present"] = bool(validated.sources_used)
        checks["evidence_complete"] = bool(validated.sources_used) and all(
            _evidence_is_complete(source)
            for source in validated.sources_used
        )
        if case.expected_sources:
            checks["source_match"] = _source_match(case, validated)
        if case.expected_chunk_ids:
            checks["expected_chunk_match"] = _chunk_match(case, validated)
        if documents_by_id is not None:
            checks["quotation_grounded"] = bool(validated.sources_used) and all(
                _quotation_is_grounded(
                    source,
                    documents_by_id,
                    masked=case.expected_behavior == "masked_answer",
                )
                for source in validated.sources_used
            )
        if case.expected_behavior == "masked_answer":
            checks["masking_safe"] = _response_is_masked(validated)
        if triad is not None and triad.answer_relevance is not None:
            checks["answer_relevance_pass"] = (
                triad.answer_relevance >= ANSWER_RELEVANCE_PASS_THRESHOLD
            )

    response_summary = {
        "is_refusal": validated.is_refusal,
        "confidence_level": validated.confidence_level,
        "refusal_reason": validated.refusal_reason,
        "sources": [
            {
                "filepath": source.filepath,
                "chunk_id": source.chunk_id,
            }
            for source in validated.sources_used
        ],
    }
    status = "passed" if all(checks.values()) else "failed"
    score = _question_score(case, validated, checks, triad)
    return CaseEvaluation(
        id=case.id,
        question=case.question,
        category=case.category,
        status=status,
        duration_ms=duration_ms,
        checks=checks,
        response_summary=response_summary,
        triad=triad,
        score=score,
    )


def _metric(
    evaluations: Sequence[CaseEvaluation],
    check_name: str,
) -> dict[str, int | float | None]:
    observed = [
        evaluation.checks[check_name]
        for evaluation in evaluations
        if check_name in evaluation.checks
    ]
    correct = sum(observed)
    return {
        "correct": correct,
        "evaluated": len(observed),
        "rate": correct / len(observed) if observed else None,
    }


def _triad_metric(
    evaluations: Sequence[CaseEvaluation],
    field_name: Literal[
        "context_relevance", "answer_relevance", "groundedness"
    ],
) -> dict[str, int | float | None]:
    observed = [
        value
        for evaluation in evaluations
        if evaluation.triad is not None
        if (value := getattr(evaluation.triad, field_name)) is not None
    ]
    return {
        "average": sum(observed) / len(observed) if observed else None,
        "evaluated": len(observed),
    }


def _rubric_summary(
    evaluations: Sequence[CaseEvaluation],
) -> dict[str, int | float | None]:
    scored = [
        evaluation.score
        for evaluation in evaluations
        if evaluation.score is not None
        and evaluation.score.total_points is not None
    ]
    points_earned = round(
        sum(score.total_points or 0.0 for score in scored),
        2,
    )
    points_possible = len(scored)
    return {
        "points_earned": points_earned,
        "points_possible": points_possible,
        "rate": (
            points_earned / points_possible
            if points_possible
            else None
        ),
        "scored_cases": len(scored),
        "incomplete_cases": len(evaluations) - len(scored),
    }


def calculate_category_summary(
    evaluations: Sequence[CaseEvaluation],
) -> dict[str, dict[str, Any]]:
    """Agrega status, rubrica e Triad por categoria do benchmark."""
    categories = sorted({evaluation.category for evaluation in evaluations})
    summary: dict[str, dict[str, Any]] = {}
    for category in categories:
        category_evaluations = [
            evaluation
            for evaluation in evaluations
            if evaluation.category == category
        ]
        summary[category] = {
            "total_cases": len(category_evaluations),
            "passed_cases": sum(
                evaluation.status == "passed"
                for evaluation in category_evaluations
            ),
            "failed_cases": sum(
                evaluation.status == "failed"
                for evaluation in category_evaluations
            ),
            "error_cases": sum(
                evaluation.status == "error"
                for evaluation in category_evaluations
            ),
            "rubric_score": _rubric_summary(category_evaluations),
            "rag_triad": {
                "context_relevance": _triad_metric(
                    category_evaluations,
                    "context_relevance",
                ),
                "answer_relevance": _triad_metric(
                    category_evaluations,
                    "answer_relevance",
                ),
                "groundedness": _triad_metric(
                    category_evaluations,
                    "groundedness",
                ),
            },
        }
    return summary


def calculate_summary(
    evaluations: Sequence[CaseEvaluation],
) -> dict[str, Any]:
    """Calcula contagens e taxas sem misturar avaliação qualitativa."""
    return {
        "total_cases": len(evaluations),
        "passed_cases": sum(
            evaluation.status == "passed" for evaluation in evaluations
        ),
        "failed_cases": sum(
            evaluation.status == "failed" for evaluation in evaluations
        ),
        "error_cases": sum(
            evaluation.status == "error" for evaluation in evaluations
        ),
        "rubric_score": _rubric_summary(evaluations),
        "by_category": calculate_category_summary(evaluations),
        "metrics": {
            "response_schema_valid": _metric(
                evaluations,
                "response_schema_valid",
            ),
            "refusal_accuracy": _metric(
                evaluations,
                "refusal_expected",
            ),
            "expected_refusal_reason_match": _metric(
                evaluations,
                "expected_refusal_reason_match",
            ),
            "evidence_present": _metric(
                evaluations,
                "evidence_present",
            ),
            "source_match": _metric(evaluations, "source_match"),
            "quotation_grounded": _metric(
                evaluations,
                "quotation_grounded",
            ),
            "masking_safe": _metric(evaluations, "masking_safe"),
        },
        "rag_triad": {
            "context_relevance": _triad_metric(
                evaluations, "context_relevance"
            ),
            "answer_relevance": _triad_metric(
                evaluations, "answer_relevance"
            ),
            "groundedness": _triad_metric(
                evaluations, "groundedness"
            ),
            "judge_failures": sum(
                evaluation.triad is not None
                and evaluation.triad.judge_status == "failed"
                for evaluation in evaluations
            ),
        },
    }


def run_cases(
    pipeline: RAGPipeline,
    cases: Sequence[BenchmarkCase],
    *,
    documents_by_id: Mapping[str, Document] | None = None,
    judge_llm=None,
    judge_config: JudgeConfig | None = None,
) -> list[CaseEvaluation]:
    """Executa casos isoladamente; um erro não interrompe os seguintes."""
    evaluations: list[CaseEvaluation] = []
    for case in cases:
        started_at = time.perf_counter()
        try:
            execution: RAGPipelineExecution = pipeline.answer_with_trace(
                case.question,
                aggregate_group_size=case.aggregate_group_size,
                debug=False,
            )
            response = execution.response
            try:
                validated_response = RAGResponse.model_validate(response)
            except ValidationError:
                validated_response = None

            judge_requested = judge_llm is not None
            judge_applicable = case.expected_behavior != "refusal"
            judge_result = None
            if (
                judge_requested
                and judge_applicable
                and validated_response is not None
            ):
                try:
                    judge_result = invoke_triad_judge(
                        question=case.question,
                        response=validated_response,
                        expected_behavior=case.expected_behavior,
                        ground_truth_answer=case.ground_truth_answer,
                        key_points=case.key_points_for_evaluation,
                        llm=judge_llm,
                        config=judge_config,
                    )
                except Exception as error:  # Fronteira do serviço de judge.
                    LOGGER.warning(
                        "Judge falhou no caso %s: %s.",
                        case.id,
                        type(error).__name__,
                    )

            triad = build_triad_evaluation(
                response=validated_response or response,
                retrieved_chunks=execution.retrieved_chunks,
                expected_behavior=case.expected_behavior,
                expected_chunk_ids=case.expected_chunk_ids,
                expected_sources=case.reference_sources,
                judge_result=judge_result,
                judge_requested=judge_requested,
                judge_applicable=judge_applicable,
            ) if validated_response is not None else None
            duration_ms = (time.perf_counter() - started_at) * 1000
            evaluation = evaluate_response(
                case,
                response,
                documents_by_id=documents_by_id,
                triad=triad,
                duration_ms=duration_ms,
            )
        except Exception as error:  # Mantém os demais casos executáveis.
            duration_ms = (time.perf_counter() - started_at) * 1000
            evaluation = CaseEvaluation(
                id=case.id,
                question=case.question,
                category=case.category,
                status="error",
                duration_ms=duration_ms,
                error={
                    "type": type(error).__name__,
                    "message": "Falha controlada durante a execução do caso.",
                },
            )

        LOGGER.info(
            "Caso %s | categoria=%s | status=%s | duracao_ms=%.2f",
            case.id,
            case.category,
            evaluation.status,
            evaluation.duration_ms,
        )
        evaluations.append(evaluation)

    return evaluations


def _documents_from_indexes(
    indexes: Mapping[str, FAISS],
) -> dict[str, Document]:
    documents_by_id: dict[str, Document] = {}
    for vectorstore in indexes.values():
        for document in vectorstore.docstore._dict.values():
            chunk_id = document.metadata.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError("Document indexado sem chunk_id válido.")
            if chunk_id in documents_by_id:
                raise ValueError(f"chunk_id duplicado nos índices: {chunk_id}.")
            documents_by_id[chunk_id] = document
    return documents_by_id


def build_runtime_pipeline() -> tuple[RAGPipeline, dict[str, Document]]:
    """Monta o pipeline real usando somente índices já persistidos."""
    embeddings = create_openai_embeddings()
    indexes = load_indexes(embeddings)
    documents_by_id = _documents_from_indexes(indexes)
    retriever = HybridRetriever(
        indexes,
        embeddings,
        list(documents_by_id.values()),
    )
    return RAGPipeline(retriever), documents_by_id


def _display_number(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


def build_summary_markdown(
    evaluations: Sequence[CaseEvaluation],
    *,
    generated_at: datetime,
) -> str:
    """Gera tabela legível sem persistir respostas ou quotations."""
    summary = calculate_summary(evaluations)
    rubric = summary["rubric_score"]
    lines = [
        "# Resumo do benchmark VendeFácil",
        "",
        f"Gerado em UTC: `{generated_at.isoformat()}`",
        "",
        "## Resultado geral",
        "",
        f"- Casos: {summary['total_cases']}",
        f"- Aprovados: {summary['passed_cases']}",
        f"- Falhos: {summary['failed_cases']}",
        f"- Erros: {summary['error_cases']}",
        (
            "- Pontuação da rubrica: "
            f"{rubric['points_earned']:.2f}/"
            f"{rubric['points_possible']}"
        ),
        f"- Casos sem score completo: {rubric['incomplete_cases']}",
        "",
        "## Resultado por categoria",
        "",
        "| Categoria | Casos | Aprovados | Falhos | Erros | Pontos | Máximo | Taxa |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category, category_summary in summary["by_category"].items():
        category_rubric = category_summary["rubric_score"]
        rate = category_rubric["rate"]
        lines.append(
            "| "
            f"{_escape_markdown_cell(category)} | "
            f"{category_summary['total_cases']} | "
            f"{category_summary['passed_cases']} | "
            f"{category_summary['failed_cases']} | "
            f"{category_summary['error_cases']} | "
            f"{category_rubric['points_earned']:.2f} | "
            f"{category_rubric['points_possible']} | "
            f"{_display_number(rate)} |"
        )

    triad = summary["rag_triad"]
    lines.extend(
        [
            "",
            "## RAG Triad",
            "",
            "| Métrica | Média | Casos avaliados |",
            "| --- | ---: | ---: |",
            (
                "| Context Relevance | "
                f"{_display_number(triad['context_relevance']['average'])} | "
                f"{triad['context_relevance']['evaluated']} |"
            ),
            (
                "| Answer Relevance | "
                f"{_display_number(triad['answer_relevance']['average'])} | "
                f"{triad['answer_relevance']['evaluated']} |"
            ),
            (
                "| Groundedness | "
                f"{_display_number(triad['groundedness']['average'])} | "
                f"{triad['groundedness']['evaluated']} |"
            ),
            "",
            "## Resultado por questão",
            "",
            "| ID | Categoria | Status | Pontos | Context | Answer | Groundedness |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for evaluation in evaluations:
        score = (
            evaluation.score.total_points
            if evaluation.score is not None
            else None
        )
        context = (
            evaluation.triad.context_relevance
            if evaluation.triad is not None
            else None
        )
        answer = (
            evaluation.triad.answer_relevance
            if evaluation.triad is not None
            else None
        )
        groundedness = (
            evaluation.triad.groundedness
            if evaluation.triad is not None
            else None
        )
        lines.append(
            f"| {evaluation.id} | "
            f"{_escape_markdown_cell(evaluation.category)} | "
            f"{evaluation.status} | {_display_number(score)} | "
            f"{_display_number(context)} | {_display_number(answer)} | "
            f"{_display_number(groundedness)} |"
        )
    return "\n".join(lines) + "\n"


def save_results(
    evaluations: Sequence[CaseEvaluation],
    output_directory: str | Path = DEFAULT_RESULTS_DIR,
) -> BenchmarkArtifacts:
    """Persiste JSON canônico, histórico e tabela Markdown segura."""
    generated_at = datetime.now(timezone.utc)
    summary = calculate_summary(evaluations)
    payload = {
        "timestamp": generated_at.isoformat(),
        "summary": summary,
        "results": [
            evaluation.model_dump(mode="json")
            for evaluation in evaluations
        ],
    }
    results_directory = Path(output_directory)
    results_directory.mkdir(parents=True, exist_ok=True)
    timestamped_path = results_directory / (
        f"benchmark_{generated_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    results_path = results_directory / "results.json"
    summary_path = results_directory / "benchmark_summary.md"
    serialized_payload = json.dumps(payload, ensure_ascii=False, indent=2)
    for output_path in (results_path, timestamped_path):
        output_path.write_text(serialized_payload, encoding="utf-8")
    summary_path.write_text(
        build_summary_markdown(evaluations, generated_at=generated_at),
        encoding="utf-8",
    )
    return BenchmarkArtifacts(
        results_json=results_path,
        summary_markdown=summary_path,
        timestamped_json=timestamped_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa o benchmark determinístico do RAG VendeFácil."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="Executa somente o ID informado; pode ser repetido.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Valida e resume os casos sem construir o pipeline.",
    )
    parser.add_argument(
        "--with-judge",
        action="store_true",
        help=(
            "Calcula Answer Relevance e Groundedness com uma chamada "
            "adicional de LLM por caso."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cases = load_cases(args.cases)
    try:
        cases = select_cases(
            cases,
            case_ids=args.case_ids,
            limit=args.limit,
        )
    except ValueError as error:
        parser.error(str(error))

    if args.validate_only:
        category_counts: dict[str, int] = {}
        for case in cases:
            category_counts[case.category] = (
                category_counts.get(case.category, 0) + 1
            )
        print(
            json.dumps(
                {
                    "cases_path": str(args.cases),
                    "total_cases": len(cases),
                    "categories": category_counts,
                    "expected_behaviors": {
                        behavior: sum(
                            case.expected_behavior == behavior
                            for case in cases
                        )
                        for behavior in (
                            "answer",
                            "masked_answer",
                            "refusal",
                        )
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    pipeline, documents_by_id = build_runtime_pipeline()
    judge_llm = create_judge_model() if args.with_judge else None
    evaluations = run_cases(
        pipeline,
        cases,
        documents_by_id=documents_by_id,
        judge_llm=judge_llm,
    )
    artifacts = save_results(evaluations, args.results_dir)
    print(json.dumps(calculate_summary(evaluations), indent=2))
    print(f"Resultado canônico: {artifacts.results_json}")
    print(f"Tabela resumo: {artifacts.summary_markdown}")
    print(f"Histórico da execução: {artifacts.timestamped_json}")


if __name__ == "__main__":
    main()
