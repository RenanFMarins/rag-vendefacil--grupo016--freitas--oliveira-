"""Executa o benchmark determinístico do pipeline RAG VendeFácil."""

import argparse
import json
import logging
import time
from collections.abc import Mapping, Sequence
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
    error: dict[str, str] | None = None


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
        )

    expected_refusal = case.expected_behavior == "refusal"
    checks = {
        "response_schema_valid": True,
        "refusal_expected": validated.is_refusal == expected_refusal,
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
    return CaseEvaluation(
        id=case.id,
        question=case.question,
        category=case.category,
        status=status,
        duration_ms=duration_ms,
        checks=checks,
        response_summary=response_summary,
        triad=triad,
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


def save_results(
    evaluations: Sequence[CaseEvaluation],
    output_directory: str | Path = DEFAULT_RESULTS_DIR,
) -> Path:
    """Persiste um resultado sem respostas, quotations ou dados sensíveis."""
    generated_at = datetime.now(timezone.utc)
    payload = {
        "timestamp": generated_at.isoformat(),
        "summary": calculate_summary(evaluations),
        "results": [
            evaluation.model_dump(mode="json")
            for evaluation in evaluations
        ],
    }
    results_directory = Path(output_directory)
    results_directory.mkdir(parents=True, exist_ok=True)
    output_path = results_directory / (
        f"benchmark_{generated_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


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
    output_path = save_results(evaluations, args.results_dir)
    print(json.dumps(calculate_summary(evaluations), indent=2))
    print(f"Resultado salvo em: {output_path}")


if __name__ == "__main__":
    main()

