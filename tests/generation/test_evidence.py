from copy import deepcopy

import pytest
from langchain_core.documents import Document

from src.generation.evidence import EvidenceBuildError
from src.generation.evidence import MAX_QUOTATION_LENGTH
from src.generation.evidence import build_source_evidence


def make_document(content: str = "Trecho literal do documento.") -> Document:
    return Document(
        page_content=content,
        metadata={
            "source_file": "tickets.jsonl",
            "chunk_id": "tickets:TCK-1001:000",
            "doc_type": "ticket",
            "sensitivity": "interno",
        },
    )


def test_builds_evidence_from_valid_document_without_modifying_it() -> None:
    document = make_document()
    original_content = document.page_content
    original_metadata = deepcopy(document.metadata)

    evidence = build_source_evidence(document)

    assert evidence.filepath == document.metadata["source_file"]
    assert evidence.chunk_id == document.metadata["chunk_id"]
    assert evidence.quotation == document.page_content
    assert document.page_content == original_content
    assert document.metadata == original_metadata


def test_rejects_document_without_source_file() -> None:
    document = make_document()
    del document.metadata["source_file"]

    with pytest.raises(EvidenceBuildError, match="source_file"):
        build_source_evidence(document)


def test_rejects_document_without_chunk_id() -> None:
    document = make_document()
    del document.metadata["chunk_id"]

    with pytest.raises(EvidenceBuildError, match="chunk_id"):
        build_source_evidence(document)


def test_rejects_document_with_empty_page_content() -> None:
    document = make_document("   \n\t")

    with pytest.raises(EvidenceBuildError, match="page_content vazio"):
        build_source_evidence(document)


def test_limits_long_content_to_literal_500_character_substring() -> None:
    content = "0123456789" * 60
    document = make_document(content)

    evidence = build_source_evidence(document)

    assert len(evidence.quotation) == MAX_QUOTATION_LENGTH
    assert evidence.quotation == content[:MAX_QUOTATION_LENGTH]
    assert evidence.quotation in document.page_content

