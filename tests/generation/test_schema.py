import pytest
from pydantic import ValidationError

from starter.schema import RAGResponse
from starter.schema import SourceEvidence


def make_response(**changes: object) -> RAGResponse:
    data = {
        "answer": "Resposta fundamentada nos documentos recuperados.",
        "confidence_level": "alta",
        "sources_used": [
            SourceEvidence(
                filepath="data/semi_structured/tickets.jsonl",
                chunk_id="tickets:TCK-1001:000",
                quotation="O ticket relata uma falha de sincronização.",
            )
        ],
        "reasoning": "A resposta utiliza a evidência recuperada.",
        "is_refusal": False,
        "refusal_reason": None,
    }
    data.update(changes)
    return RAGResponse.model_validate(data)


def test_accepts_valid_source_evidence() -> None:
    evidence = SourceEvidence(
        filepath="data/unstructured/manual.md",
        chunk_id="manual:manual:section-001:part-000",
        quotation="Trecho utilizado como evidência.",
    )

    assert evidence.filepath == "data/unstructured/manual.md"
    assert evidence.chunk_id == "manual:manual:section-001:part-000"
    assert evidence.quotation == "Trecho utilizado como evidência."


def test_accepts_quotation_with_exactly_500_characters() -> None:
    evidence = SourceEvidence(
        filepath="tickets.jsonl",
        chunk_id="tickets:TCK-1001:000",
        quotation="a" * 500,
    )

    assert len(evidence.quotation) == 500


def test_rejects_quotation_above_500_characters() -> None:
    with pytest.raises(ValidationError):
        SourceEvidence(
            filepath="tickets.jsonl",
            chunk_id="tickets:TCK-1001:000",
            quotation="a" * 501,
        )


@pytest.mark.parametrize("confidence", ["alta", "media", "baixa", "recusado"])
def test_accepts_valid_confidence_level(confidence: str) -> None:
    response = make_response(confidence_level=confidence)

    assert response.confidence_level == confidence


@pytest.mark.parametrize("confidence", ["Alta", "ALTA", "muito alta"])
def test_rejects_invalid_confidence_level(confidence: str) -> None:
    with pytest.raises(ValidationError):
        make_response(confidence_level=confidence)


@pytest.mark.parametrize(
    "reason",
    ["lgpd", "fora_de_escopo", "sem_evidencia", None],
)
def test_accepts_valid_refusal_reason(reason: str | None) -> None:
    if reason is None:
        response = make_response(refusal_reason=None)
    else:
        response = make_response(
            confidence_level="recusado",
            sources_used=[],
            is_refusal=True,
            refusal_reason=reason,
        )

    assert response.refusal_reason == reason


def test_rejects_invalid_refusal_reason() -> None:
    with pytest.raises(ValidationError):
        make_response(refusal_reason="motivo_desconhecido")


def test_accepts_valid_refusal() -> None:
    response = make_response(
        confidence_level="recusado",
        sources_used=[],
        is_refusal=True,
        refusal_reason="lgpd",
    )

    assert response.is_refusal is True
    assert response.confidence_level == "recusado"
    assert response.sources_used == []
    assert response.refusal_reason == "lgpd"


def test_rejects_refusal_with_sources() -> None:
    with pytest.raises(ValidationError, match="sources_used"):
        make_response(
            confidence_level="recusado",
            is_refusal=True,
            refusal_reason="lgpd",
        )


def test_rejects_refusal_without_refusal_reason() -> None:
    with pytest.raises(ValidationError, match="refusal_reason"):
        make_response(
            confidence_level="recusado",
            sources_used=[],
            is_refusal=True,
            refusal_reason=None,
        )


def test_rejects_refusal_with_non_refusal_confidence() -> None:
    with pytest.raises(ValidationError, match="confidence_level"):
        make_response(
            confidence_level="alta",
            sources_used=[],
            is_refusal=True,
            refusal_reason="lgpd",
        )


def test_accepts_normal_response_with_source() -> None:
    response = make_response()

    assert response.is_refusal is False
    assert len(response.sources_used) == 1
    assert response.refusal_reason is None


def test_rejects_normal_response_without_source() -> None:
    with pytest.raises(ValidationError, match="ao menos uma evidência"):
        make_response(sources_used=[])


def test_rejects_normal_response_with_refusal_reason() -> None:
    with pytest.raises(ValidationError, match="refusal_reason"):
        make_response(refusal_reason="fora_de_escopo")

