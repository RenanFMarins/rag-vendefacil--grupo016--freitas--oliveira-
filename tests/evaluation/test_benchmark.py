import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from langchain_core.documents import Document

from ingestion.loaders.csv_loader import load_csv_documents
from ingestion.loaders.json_loader import load_json_documents
from ingestion.loaders.pdf_loader import load_pdf_documents
from eval.run_benchmark import BenchmarkCase
from eval.run_benchmark import DEFAULT_CASES_PATH
from eval.run_benchmark import build_summary_markdown
from eval.run_benchmark import calculate_category_summary
from eval.run_benchmark import calculate_summary
from eval.run_benchmark import evaluate_response
from eval.run_benchmark import load_cases
from eval.run_benchmark import run_cases
from eval.run_benchmark import save_results
from eval.run_benchmark import select_cases
from eval.triad import TriadJudgeDraft
from eval.triad import TriadEvaluation
from src.config import DATA_DIR
from src.generation.evidence import build_source_evidence
from src.pipeline import RetrievedChunk
from src.pipeline import RAGPipelineExecution
from starter.schema import RAGResponse, SourceEvidence


def make_case(**overrides: object) -> BenchmarkCase:
    data = {
        "id": "Q001",
        "question": "Como realizar uma sangria no VendeFácil PDV?",
        "category": "respondable",
        "expected_behavior": "answer",
        "expected_sources": ["manual.md"],
        "source_match_mode": "any",
    }
    data.update(overrides)
    return BenchmarkCase.model_validate(data)


def make_evidence() -> SourceEvidence:
    return SourceEvidence(
        filepath="manual.md",
        chunk_id="manual:001",
        quotation="Pressione F8 para iniciar a sangria.",
    )


def make_answer() -> RAGResponse:
    return RAGResponse(
        answer="Use a opção de sangria do PDV.",
        confidence_level="alta",
        sources_used=[make_evidence()],
        reasoning="O manual descreve o procedimento.",
        is_refusal=False,
        refusal_reason=None,
    )


def make_refusal(reason: str = "lgpd") -> dict[str, object]:
    return {
        "answer": "Não posso fornecer essa informação.",
        "confidence_level": "recusado",
        "sources_used": [],
        "reasoning": "A solicitação deve ser recusada.",
        "is_refusal": True,
        "refusal_reason": reason,
    }


class FakePipeline:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = iter(outcomes)

    def answer_with_trace(
        self, question: str, **kwargs: object
    ) -> RAGPipelineExecution:
        outcome = next(self._outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, RAGPipelineExecution):
            return outcome
        return RAGPipelineExecution(response=outcome)


class CountingJudgeModel:
    def __init__(self, result: TriadJudgeDraft) -> None:
        self.result = result
        self.calls = 0

    def with_structured_output(self, schema, *, method, strict):
        return self

    def invoke(self, messages):
        self.calls += 1
        return self.result


def test_loads_benchmark_cases(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps([make_case().model_dump(mode="json")]),
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert len(cases) == 1
    assert cases[0].id == "Q001"


def test_selects_benchmark_cases_by_id_in_requested_order() -> None:
    cases = [
        make_case(id="Q001"),
        make_case(id="Q002"),
        make_case(id="Q003"),
    ]

    selected = select_cases(
        cases,
        case_ids=["Q003", "Q001", "Q003"],
        limit=2,
    )

    assert [case.id for case in selected] == ["Q003", "Q001"]


def test_rejects_unknown_benchmark_case_id() -> None:
    with pytest.raises(ValueError, match="Q999"):
        select_cases([make_case(id="Q001")], case_ids=["Q999"])


def official_question(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "id": "Q01",
        "category": "Fácil (RAG Básico)",
        "question": "Quais produtos a VendeFácil oferece?",
        "expected_sources": ["data/structured/products.json"],
        "expected_metadata": {"format": "json"},
        "ground_truth_answer": "A VendeFácil oferece cinco produtos.",
        "key_points_for_evaluation": ["Mencionar cinco produtos"],
    }
    data.update(overrides)
    return data


def write_official_suite(
    path: Path,
    questions: list[dict[str, object]],
) -> None:
    path.write_text(
        json.dumps(
            {
                "benchmark_name": "Benchmark oficial",
                "version": "1.0",
                "description": "Casos oficiais para avaliação.",
                "questions": questions,
            }
        ),
        encoding="utf-8",
    )


def test_loads_official_benchmark_envelope(tmp_path: Path) -> None:
    path = tmp_path / "official.json"
    write_official_suite(
        path,
        [official_question(expected_chunk_ids=["products:001"])],
    )

    cases = load_cases(path)

    assert len(cases) == 1
    assert cases[0].expected_behavior == "answer"
    assert cases[0].expected_sources == ["data/structured/products.json"]
    assert cases[0].reference_sources == ["data/structured/products.json"]
    assert cases[0].expected_chunk_ids == ["products:001"]
    assert cases[0].expected_metadata == {"format": "json"}
    assert cases[0].ground_truth_answer == (
        "A VendeFácil oferece cinco produtos."
    )
    assert cases[0].key_points_for_evaluation == [
        "Mencionar cinco produtos"
    ]


def test_converts_official_guardrail_expectations(tmp_path: Path) -> None:
    path = tmp_path / "official.json"
    write_official_suite(
        path,
        [
            official_question(
                id="Q15",
                expected_metadata={"sensitive": True},
            ),
            official_question(
                id="Q20",
                expected_sources=[],
                expected_metadata={"out_of_domain": True},
            ),
        ],
    )

    lgpd_case, scope_case = load_cases(path)

    assert lgpd_case.expected_behavior == "refusal"
    assert lgpd_case.expected_refusal_reason == "lgpd"
    assert lgpd_case.expected_sources == []
    assert lgpd_case.reference_sources == [
        "data/structured/products.json"
    ]
    assert scope_case.expected_behavior == "refusal"
    assert scope_case.expected_refusal_reason == "fora_de_escopo"


def test_official_sources_are_alternatives_for_functional_check(
    tmp_path: Path,
) -> None:
    path = tmp_path / "official.json"
    write_official_suite(
        path,
        [
            official_question(
                id="Q03",
                expected_sources=[
                    "data/policies/reembolso.pdf",
                    "data/policies/reembolso.md",
                ],
            ),
            official_question(
                id="Q08",
                expected_sources=[
                    "data/emails/customer.txt",
                    "data/tickets.jsonl",
                ],
            ),
        ],
    )

    alternative_case, multiple_source_case = load_cases(path)

    assert alternative_case.source_match_mode == "any"
    assert multiple_source_case.source_match_mode == "any"


def test_default_official_benchmark_contains_all_distributed_cases() -> None:
    cases = load_cases(DEFAULT_CASES_PATH)

    assert len(cases) == 24
    assert cases[0].id == "Q01"
    assert cases[-1].id == "Q24"


def test_official_benchmark_covers_all_six_ingestion_formats() -> None:
    cases = load_cases(DEFAULT_CASES_PATH)
    suffixes = {
        Path(source).suffix
        for case in cases
        for source in case.reference_sources
    }

    assert {".csv", ".json", ".jsonl", ".md", ".pdf", ".txt"} <= suffixes
    assert next(case for case in cases if case.id == "Q15").expected_behavior == (
        "refusal"
    )
    assert next(case for case in cases if case.id == "Q19").expected_sources == [
        "data/structured/stores.json"
    ]


@pytest.mark.parametrize(
    ("document", "expected_source"),
    [
        (load_json_documents(DATA_DIR / "structured" / "products.json")[0], "products.json"),
        (load_csv_documents(DATA_DIR / "structured" / "customers.csv")[0], "customers.csv"),
        (load_pdf_documents(DATA_DIR / "unstructured" / "policies")[0], "reembolso.pdf"),
    ],
)
def test_evaluates_grounded_evidence_from_integrated_formats(
    document: Document,
    expected_source: str,
) -> None:
    evidence = build_source_evidence(document)
    response = RAGResponse(
        answer="Resposta sustentada pelo documento integrado.",
        confidence_level="alta",
        sources_used=[evidence],
        reasoning="A evidência foi construída diretamente do chunk.",
        is_refusal=False,
    )
    case = make_case(expected_sources=[expected_source])

    evaluation = evaluate_response(
        case,
        response,
        documents_by_id={evidence.chunk_id: document},
    )

    assert evaluation.status == "passed"
    assert evaluation.checks["source_match"] is True
    assert evaluation.checks["quotation_grounded"] is True


def test_evaluates_valid_answer_with_grounded_evidence() -> None:
    document = Document(
        page_content="Pressione F8 para iniciar a sangria.",
        metadata={"source_file": "manual.md", "chunk_id": "manual:001"},
    )

    result = evaluate_response(
        make_case(),
        make_answer(),
        documents_by_id={"manual:001": document},
    )

    assert result.status == "passed"
    assert result.checks["response_schema_valid"] is True
    assert result.checks["evidence_present"] is True
    assert result.checks["quotation_grounded"] is True


def test_evaluates_correct_refusal() -> None:
    case = make_case(
        category="lgpd_refuse",
        expected_behavior="refusal",
        expected_sources=[],
        expected_refusal_reason="lgpd",
    )

    result = evaluate_response(case, make_refusal())

    assert result.status == "passed"
    assert result.checks["expected_refusal_reason_match"] is True


def test_rejects_incorrect_refusal_reason() -> None:
    case = make_case(
        category="lgpd_refuse",
        expected_behavior="refusal",
        expected_sources=[],
        expected_refusal_reason="lgpd",
    )

    result = evaluate_response(case, make_refusal("fora_de_escopo"))

    assert result.status == "failed"
    assert result.checks["expected_refusal_reason_match"] is False


def test_detects_missing_evidence_for_expected_answer() -> None:
    result = evaluate_response(
        make_case(),
        make_refusal("sem_evidencia"),
    )

    assert result.status == "failed"
    assert result.checks["refusal_expected"] is False
    assert result.checks["evidence_present"] is False


def test_detects_missing_expected_source() -> None:
    case = make_case(expected_sources=["outra_fonte.md"])

    result = evaluate_response(case, make_answer())

    assert result.status == "failed"
    assert result.checks["source_match"] is False


def test_source_match_compares_file_name_with_official_relative_path() -> None:
    case = make_case(expected_sources=["data/documentation/manual.md"])

    result = evaluate_response(case, make_answer())

    assert result.status == "passed"
    assert result.checks["source_match"] is True


def test_calculates_summary() -> None:
    passed = evaluate_response(make_case(), make_answer())
    failed = evaluate_response(
        make_case(
            id="Q002",
            category="lgpd_refuse",
            expected_behavior="refusal",
            expected_sources=[],
            expected_refusal_reason="lgpd",
        ),
        make_refusal("fora_de_escopo"),
    )

    summary = calculate_summary([passed, failed])

    assert summary["total_cases"] == 2
    assert summary["passed_cases"] == 1
    assert summary["failed_cases"] == 1
    assert summary["error_cases"] == 0
    assert summary["metrics"]["response_schema_valid"]["rate"] == 1.0


def test_scores_valid_answer_using_official_weights() -> None:
    document = Document(
        page_content="Pressione F8 para iniciar a sangria.",
        metadata={"source_file": "manual.md", "chunk_id": "manual:001"},
    )
    triad = TriadEvaluation(
        context_relevance=1.0,
        context_relevance_basis="source_file",
        answer_relevance=0.9,
        groundedness=1.0,
        judge_status="passed",
    )

    evaluation = evaluate_response(
        make_case(),
        make_answer(),
        documents_by_id={"manual:001": document},
        triad=triad,
    )

    assert evaluation.score is not None
    assert evaluation.score.answer_points == 0.5
    assert evaluation.score.citation_points == 0.3
    assert evaluation.score.consistency_points == 0.2
    assert evaluation.score.total_points == 1.0
    assert evaluation.score.complete is True


def test_answer_score_remains_incomplete_without_judge() -> None:
    evaluation = evaluate_response(make_case(), make_answer())

    assert evaluation.score is not None
    assert evaluation.score.answer_correct is None
    assert evaluation.score.answer_points is None
    assert evaluation.score.total_points is None
    assert evaluation.score.complete is False


def test_low_answer_relevance_loses_answer_points() -> None:
    triad = TriadEvaluation(
        context_relevance=1.0,
        context_relevance_basis="source_file",
        answer_relevance=0.69,
        groundedness=1.0,
        judge_status="passed",
    )

    evaluation = evaluate_response(
        make_case(),
        make_answer(),
        triad=triad,
    )

    assert evaluation.status == "failed"
    assert evaluation.score is not None
    assert evaluation.score.answer_points == 0.0
    assert evaluation.score.total_points == 0.5


def test_correct_refusal_receives_full_score_without_judge() -> None:
    case = make_case(
        category="lgpd_refuse",
        expected_behavior="refusal",
        expected_sources=[],
        expected_refusal_reason="lgpd",
    )

    evaluation = evaluate_response(case, make_refusal())

    assert evaluation.score is not None
    assert evaluation.score.total_points == 1.0
    assert evaluation.score.complete is True


def test_calculates_summary_by_category() -> None:
    refusal_case = make_case(
        id="Q002",
        category="guardrail",
        expected_behavior="refusal",
        expected_sources=[],
        expected_refusal_reason="lgpd",
    )
    evaluations = [
        evaluate_response(make_case(category="factual"), make_answer()),
        evaluate_response(refusal_case, make_refusal()),
    ]

    categories = calculate_category_summary(evaluations)
    summary = calculate_summary(evaluations)

    assert list(categories) == ["factual", "guardrail"]
    assert categories["guardrail"]["rubric_score"]["points_earned"] == 1.0
    assert categories["factual"]["rubric_score"]["incomplete_cases"] == 1
    assert summary["rubric_score"]["scored_cases"] == 1
    assert summary["rubric_score"]["incomplete_cases"] == 1


def test_saves_canonical_json_history_and_markdown_summary(tmp_path: Path) -> None:
    case = make_case(
        category="guardrail",
        expected_behavior="refusal",
        expected_sources=[],
        expected_refusal_reason="lgpd",
    )
    evaluation = evaluate_response(case, make_refusal())

    artifacts = save_results([evaluation], tmp_path)

    assert artifacts.results_json == tmp_path / "results.json"
    assert artifacts.results_json.is_file()
    assert artifacts.summary_markdown == tmp_path / "benchmark_summary.md"
    assert artifacts.summary_markdown.is_file()
    assert artifacts.timestamped_json.is_file()
    payload = json.loads(artifacts.results_json.read_text(encoding="utf-8"))
    assert payload["summary"]["rubric_score"]["points_earned"] == 1.0
    markdown = artifacts.summary_markdown.read_text(encoding="utf-8")
    assert "## Resultado por categoria" in markdown
    assert "| Q001 | guardrail | passed | 1.000 |" in markdown


def test_summary_markdown_does_not_include_question_or_answer() -> None:
    case = make_case(
        category="guardrail",
        expected_behavior="refusal",
        expected_sources=[],
        expected_refusal_reason="lgpd",
    )
    evaluation = evaluate_response(case, make_refusal())

    markdown = build_summary_markdown(
        [evaluation],
        generated_at=datetime(2026, 9, 3, tzinfo=timezone.utc),
    )

    assert case.question not in markdown
    assert "Não posso fornecer essa informação." not in markdown


def test_run_cases_calculates_triad_from_pipeline_trace() -> None:
    case = make_case(
        reference_sources=["data/documentation/manual.md"],
        ground_truth_answer="O manual descreve o procedimento.",
    )
    execution = RAGPipelineExecution(
        response=make_answer(),
        retrieved_chunks=(
            RetrievedChunk(
                chunk_id="manual:001",
                source_file="manual.md",
                rank=1,
            ),
        ),
    )
    judge = CountingJudgeModel(
        TriadJudgeDraft(
            answer_relevance=0.8,
            answer_relevance_reason="Resposta relevante.",
            groundedness=1.0,
            groundedness_reason="Resposta sustentada.",
        )
    )

    evaluation = run_cases(
        FakePipeline([execution]),
        [case],
        judge_llm=judge,
    )[0]
    summary = calculate_summary([evaluation])

    assert evaluation.triad is not None
    assert evaluation.triad.context_relevance == 1.0
    assert evaluation.triad.context_relevance_basis == "source_file"
    assert evaluation.triad.answer_relevance == 0.8
    assert evaluation.triad.groundedness == 1.0
    assert evaluation.triad.retrieved_chunk_ids == ["manual:001"]
    assert summary["rag_triad"]["context_relevance"] == {
        "average": 1.0,
        "evaluated": 1,
    }
    assert summary["rag_triad"]["answer_relevance"] == {
        "average": 0.8,
        "evaluated": 1,
    }


def test_run_cases_never_sends_expected_refusal_to_judge() -> None:
    case = make_case(
        category="lgpd_refuse",
        expected_behavior="refusal",
        expected_sources=[],
        expected_refusal_reason="lgpd",
    )
    judge = CountingJudgeModel(
        TriadJudgeDraft(
            answer_relevance=1.0,
            answer_relevance_reason="Recusa adequada.",
            groundedness=1.0,
            groundedness_reason="Sem afirmações factuais.",
        )
    )

    evaluation = run_cases(
        FakePipeline([make_refusal()]),
        [case],
        judge_llm=judge,
    )[0]

    assert judge.calls == 0
    assert evaluation.triad is not None
    assert evaluation.triad.judge_status == "not_applicable"
    assert evaluation.triad.answer_relevance is None


def test_run_cases_continues_after_controlled_error() -> None:
    cases = [make_case(), make_case(id="Q002")]
    pipeline = FakePipeline([RuntimeError("falha simulada"), make_answer()])

    evaluations = run_cases(pipeline, cases)

    assert [evaluation.status for evaluation in evaluations] == [
        "error",
        "passed",
    ]
    assert evaluations[0].error == {
        "type": "RuntimeError",
        "message": "Falha controlada durante a execução do caso.",
    }
