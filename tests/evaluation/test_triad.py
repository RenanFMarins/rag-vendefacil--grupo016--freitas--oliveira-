from eval.triad import JudgeConfig
from eval.triad import TriadJudgeDraft
from eval.triad import build_triad_evaluation
from eval.triad import calculate_context_relevance
from eval.triad import invoke_triad_judge
from src.pipeline import RetrievedChunk
from starter.schema import RAGResponse, SourceEvidence


def make_response(*, refusal: bool = False) -> RAGResponse:
    if refusal:
        return RAGResponse(
            answer="Não posso fornecer essa informação.",
            confidence_level="recusado",
            sources_used=[],
            reasoning="A solicitação deve ser recusada.",
            is_refusal=True,
            refusal_reason="lgpd",
        )
    return RAGResponse(
        answer="O procedimento está descrito no manual.",
        confidence_level="alta",
        sources_used=[
            SourceEvidence(
                filepath="manual.md",
                chunk_id="manual:001",
                quotation="Use a opção indicada no menu principal.",
            )
        ],
        reasoning="A evidência descreve o procedimento.",
        is_refusal=False,
        refusal_reason=None,
    )


def retrieved(
    chunk_id: str,
    source_file: str,
    rank: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_file=source_file,
        rank=rank,
    )


class FakeStructuredModel:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = iter(outcomes)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        outcome = next(self._outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeChatModel:
    def __init__(self, outcomes: list[object]) -> None:
        self.structured_model = FakeStructuredModel(outcomes)
        self.schema = None
        self.method = None
        self.strict = None

    def with_structured_output(self, schema, *, method, strict):
        self.schema = schema
        self.method = method
        self.strict = strict
        return self.structured_model


def valid_judge_result() -> TriadJudgeDraft:
    return TriadJudgeDraft(
        answer_relevance=0.9,
        answer_relevance_reason="A resposta atende diretamente à pergunta.",
        groundedness=1.0,
        groundedness_reason="As afirmações aparecem na evidência.",
    )


def test_context_relevance_prefers_expected_chunk_ids() -> None:
    score, basis = calculate_context_relevance(
        [
            retrieved("manual:001", "manual.md"),
            retrieved("ticket:999", "tickets.jsonl", rank=2),
        ],
        expected_chunk_ids=["manual:001", "ticket:001"],
        expected_sources=["data/manual.md"],
    )

    assert score == 0.5
    assert basis == "chunk_id"


def test_context_relevance_uses_sources_as_documented_fallback() -> None:
    score, basis = calculate_context_relevance(
        [retrieved("policy:001", "reembolso.md")],
        expected_sources=[
            "data/policies/reembolso.pdf",
            "data/policies/reembolso.md",
        ],
    )

    assert score == 1.0
    assert basis == "source_file"


def test_context_relevance_counts_distinct_expected_documents() -> None:
    score, basis = calculate_context_relevance(
        [retrieved("email:001", "customer_001.txt")],
        expected_sources=[
            "data/emails/customer_001.txt",
            "data/tickets.jsonl",
        ],
    )

    assert score == 0.5
    assert basis == "source_file"


def test_refusal_has_no_context_or_judge_metric() -> None:
    evaluation = build_triad_evaluation(
        response=make_response(refusal=True),
        retrieved_chunks=[],
        expected_behavior="refusal",
        expected_sources=[],
        judge_requested=True,
        judge_applicable=False,
    )

    assert evaluation.context_relevance is None
    assert evaluation.context_relevance_basis == "not_applicable"
    assert evaluation.answer_relevance is None
    assert evaluation.groundedness is None
    assert evaluation.judge_status == "not_applicable"


def test_judge_uses_strict_pydantic_structured_output() -> None:
    llm = FakeChatModel([valid_judge_result()])

    result = invoke_triad_judge(
        question="Como realizar o procedimento?",
        response=make_response(),
        expected_behavior="answer",
        ground_truth_answer="O manual explica o procedimento.",
        key_points=["Mencionar o menu principal"],
        llm=llm,
    )

    assert result == valid_judge_result()
    assert llm.schema is TriadJudgeDraft
    assert llm.method == "json_schema"
    assert llm.strict is True


def test_judge_retries_invalid_structured_output_once() -> None:
    llm = FakeChatModel(
        [
            {"answer_relevance": 2},
            valid_judge_result(),
        ]
    )

    result = invoke_triad_judge(
        question="Como realizar o procedimento?",
        response=make_response(),
        expected_behavior="answer",
        ground_truth_answer="O manual explica o procedimento.",
        key_points=[],
        llm=llm,
        config=JudgeConfig(max_attempts=2),
    )

    assert result == valid_judge_result()
    assert llm.structured_model.calls == 2


def test_judge_returns_controlled_failure_after_retry_limit() -> None:
    llm = FakeChatModel(
        [
            {"answer_relevance": 2},
            {"groundedness": -1},
        ]
    )

    result = invoke_triad_judge(
        question="Como realizar o procedimento?",
        response=make_response(),
        expected_behavior="answer",
        ground_truth_answer="O manual explica o procedimento.",
        key_points=[],
        llm=llm,
        config=JudgeConfig(max_attempts=2),
    )

    assert result is None
    assert llm.structured_model.calls == 2


def test_triad_persists_scores_but_not_judge_reasons() -> None:
    evaluation = build_triad_evaluation(
        response=make_response(),
        retrieved_chunks=[retrieved("manual:001", "manual.md")],
        expected_behavior="answer",
        expected_chunk_ids=["manual:001"],
        judge_result=valid_judge_result(),
        judge_requested=True,
    )

    assert evaluation.context_relevance == 1.0
    assert evaluation.answer_relevance == 0.9
    assert evaluation.groundedness == 1.0
    assert evaluation.judge_status == "passed"
    assert "reason" not in evaluation.model_dump_json()

