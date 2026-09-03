import logging
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.interface_rag import FRIENDLY_ERROR
from app.interface_rag import ask_pipeline
from app.interface_rag import response_view_model
from starter.schema import RAGResponse
from starter.schema import SourceEvidence


class FakePipeline:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[tuple[str, bool]] = []

    def answer(self, question: str, *, debug: bool = False):
        self.calls.append((question, debug))
        if self.error:
            raise self.error
        return self.response


def normal_response() -> RAGResponse:
    return RAGResponse(
        answer="A sangria retira valores do caixa.",
        confidence_level="alta",
        sources_used=[
            SourceEvidence(
                filepath="manual_pdv.md",
                chunk_id="manual:pdv:001",
                quotation="A sangria retira valores do caixa.",
            )
        ],
        reasoning="A resposta está descrita no manual.",
        is_refusal=False,
    )


def refusal(reason: str) -> RAGResponse:
    return RAGResponse(
        answer="Não posso responder a esta solicitação.",
        confidence_level="recusado",
        sources_used=[],
        reasoning="A solicitação foi recusada pelo guardrail.",
        is_refusal=True,
        refusal_reason=reason,
    )


def test_normal_response_preserves_contract_and_evidence() -> None:
    view = response_view_model(normal_response())

    assert view["answer"] == "A sangria retira valores do caixa."
    assert view["confidence_level"] == "alta"
    assert view["is_refusal"] is False
    assert view["refusal_reason"] is None
    assert view["sources_used"][0]["chunk_id"] == "manual:pdv:001"


def test_refusal_labels_cover_all_typed_reasons() -> None:
    assert "privacidade" in response_view_model(refusal("lgpd"))[
        "refusal_label"
    ]
    assert "fora do escopo" in response_view_model(
        refusal("fora_de_escopo")
    )["refusal_label"]
    assert "evidência suficiente" in response_view_model(
        refusal("sem_evidencia")
    )["refusal_label"]


def test_ask_pipeline_uses_public_answer_api_with_debug() -> None:
    expected = normal_response()
    pipeline = FakePipeline(response=expected)

    response, error = ask_pipeline(
        pipeline,
        "Como fazer sangria?",
        debug=True,
    )

    assert response == expected
    assert error is None
    assert pipeline.calls == [("Como fazer sangria?", True)]


def test_technical_error_is_controlled_without_sensitive_details(
    caplog,
) -> None:
    sensitive_value = "CPF 123.456.789-00"
    pipeline = FakePipeline(error=RuntimeError(sensitive_value))

    with caplog.at_level(logging.ERROR):
        response, error = ask_pipeline(pipeline, "Pergunta corporativa")

    assert response is None
    assert error == FRIENDLY_ERROR
    assert sensitive_value not in error
    assert sensitive_value not in caplog.text
    assert "RuntimeError" in caplog.text


def test_streamlit_interface_starts_without_loading_pipeline() -> None:
    project_root = Path(__file__).resolve().parents[2]
    app = AppTest.from_file(project_root / "app" / "interface_rag.py")

    app.run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "💬 Assistente corporativo VendeFácil"
    assert "CSV, JSON, JSONL" in app.caption[0].value
