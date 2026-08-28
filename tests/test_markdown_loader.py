from collections import Counter
from pathlib import Path

import pytest

from ingestion.loaders.markdown_loader import load_markdown_documents


MARKDOWN_DIRECTORY = Path(__file__).parents[1] / "data" / "unstructured"

REQUIRED_METADATA = {
    "source_file",
    "doc_type",
    "chunk_id",
    "sensitivity",
}


def test_loads_all_markdown_files_with_unique_chunks() -> None:
    documents = load_markdown_documents(MARKDOWN_DIRECTORY)

    assert len(documents) == 81
    assert len({doc.metadata["chunk_id"] for doc in documents}) == 81
    assert all(doc.page_content.strip() for doc in documents)
    assert all(REQUIRED_METADATA <= doc.metadata.keys() for doc in documents)
    assert all(doc.metadata["section"] for doc in documents)


def test_distributes_documents_by_source_nature() -> None:
    documents = load_markdown_documents(MARKDOWN_DIRECTORY)

    distribution = Counter(doc.metadata["doc_type"] for doc in documents)

    assert distribution == {
        "ata": 38,
        "manual": 24,
        "policy": 19,
    }


def test_preserves_manual_hierarchy_and_module() -> None:
    file_path = (
        MARKDOWN_DIRECTORY
        / "documentation"
        / "pdv"
        / "manual_pdv.md"
    )

    documents = load_markdown_documents(file_path)
    first_document = documents[0]

    expected_section = (
        "Manual do Operador - VendeFácil PDV (v3.4)"
        " > 1. Operações de Frente de Caixa"
        " > 1.1 Sangria e Suprimento"
    )
    assert len(documents) == 4
    assert first_document.metadata["doc_type"] == "manual"
    assert first_document.metadata["sensitivity"] == "publico"
    assert first_document.metadata["module"] == "pdv"
    assert first_document.metadata["section"] == expected_section
    assert first_document.page_content.startswith(f"Seção: {expected_section}")


def test_extracts_normalized_date_from_every_meeting() -> None:
    meeting_directory = MARKDOWN_DIRECTORY / "meetings"

    documents = load_markdown_documents(meeting_directory)

    assert documents
    assert all(doc.metadata["doc_type"] == "ata" for doc in documents)
    assert all(
        len(doc.metadata["date"]) == 10
        and doc.metadata["date"][4] == "-"
        and doc.metadata["date"][7] == "-"
        for doc in documents
    )


def test_classifies_sensitivity_by_document_nature() -> None:
    documents = load_markdown_documents(MARKDOWN_DIRECTORY)
    distribution = Counter(
        doc.metadata["sensitivity"] for doc in documents
    )

    assert distribution == {
        "publico": 37,
        "interno": 42,
        "restrito": 2,
    }

    hr_documents = [
        doc
        for doc in documents
        if doc.metadata["source_file"]
        == "2026-03-hr_performance_review_q1.md"
    ]
    assert hr_documents
    assert all(
        doc.metadata["sensitivity"] == "restrito"
        for doc in hr_documents
    )

    internal_policy_documents = [
        doc
        for doc in documents
        if doc.metadata["source_file"] == "home_office.md"
    ]
    assert all(
        doc.metadata["sensitivity"] == "interno"
        for doc in internal_policy_documents
    )

    public_policy_documents = [
        doc
        for doc in documents
        if doc.metadata["source_file"] == "reembolso.md"
    ]
    assert all(
        doc.metadata["sensitivity"] == "publico"
        for doc in public_policy_documents
    )


def test_applies_size_fallback_only_inside_long_section(tmp_path: Path) -> None:
    documentation_directory = tmp_path / "documentation" / "estoque"
    documentation_directory.mkdir(parents=True)
    file_path = documentation_directory / "long_manual.md"
    long_section = " ".join(
        f"Orientação de inventário {number}." for number in range(40)
    )
    file_path.write_text(
        "# Manual de Estoque\n\n"
        "## Inventário\n\n"
        f"{long_section}",
        encoding="utf-8",
    )

    documents = load_markdown_documents(
        file_path,
        chunk_size=140,
        chunk_overlap=20,
    )

    assert len(documents) > 1
    assert all(
        doc.page_content.startswith(
            "Seção: Manual de Estoque > Inventário"
        )
        for doc in documents
    )
    assert all(doc.metadata["module"] == "estoque" for doc in documents)
    assert [doc.metadata["chunk_id"] for doc in documents] == [
        f"manual:long_manual:section-000:part-{part_number:03d}"
        for part_number in range(len(documents))
    ]


def test_generates_stable_chunk_ids() -> None:
    file_path = (
        MARKDOWN_DIRECTORY
        / "documentation"
        / "estoque"
        / "sincronizacao_estoque.md"
    )

    first_ids = [
        doc.metadata["chunk_id"]
        for doc in load_markdown_documents(file_path)
    ]
    second_ids = [
        doc.metadata["chunk_id"]
        for doc in load_markdown_documents(file_path)
    ]

    assert first_ids == second_ids


def test_reports_empty_markdown_file(tmp_path: Path) -> None:
    documentation_directory = tmp_path / "documentation" / "pdv"
    documentation_directory.mkdir(parents=True)
    file_path = documentation_directory / "empty.md"
    file_path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        load_markdown_documents(file_path)

    assert str(file_path) in str(error.value)
    assert "vazio" in str(error.value)
