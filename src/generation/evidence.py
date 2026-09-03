"""Constrói evidências determinísticas a partir de Documents recuperados."""

from langchain_core.documents import Document

from starter.schema import SourceEvidence


MAX_QUOTATION_LENGTH = 500


class EvidenceBuildError(ValueError):
    """Indica que um Document não possui dados para formar uma evidência."""


def _required_metadata(document: Document, field: str) -> str:
    value = document.metadata.get(field)
    if not isinstance(value, str) or not value.strip():
        raise EvidenceBuildError(
            f"Document sem metadata obrigatório: {field}."
        )
    return value


def build_source_evidence(document: Document) -> SourceEvidence:
    if not document.page_content.strip():
        raise EvidenceBuildError("Document possui page_content vazio.")

    return SourceEvidence(
        filepath=_required_metadata(document, "source_file"),
        chunk_id=_required_metadata(document, "chunk_id"),
        quotation=document.page_content[:MAX_QUOTATION_LENGTH],
    )

