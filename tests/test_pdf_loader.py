from pathlib import Path

import pytest
from pypdf import PdfWriter

import ingestion.loaders.pdf_loader as pdf_loader
from ingestion.loaders.pdf_loader import load_pdf_documents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

POLICIES = DATA_DIR / "unstructured" / "policies"
REEMBOLSO = POLICIES / "reembolso.pdf"


def test_loads_real_policy_pdfs_with_required_metadata() -> None:
    documents = load_pdf_documents(POLICIES)

    assert documents
    assert {document.metadata["source_file"] for document in documents} == {
        "reembolso.pdf",
        "seguranca_lgpd.pdf",
    }
    assert all(document.page_content.strip() for document in documents)
    assert all(
        document.metadata["doc_type"] == "policy"
        for document in documents
    )
    assert all(
        document.metadata["sensitivity"] == "publico"
        for document in documents
    )
    assert all(document.metadata["page"] >= 1 for document in documents)


def test_pdf_chunk_ids_are_unique_and_deterministic() -> None:
    first = load_pdf_documents(POLICIES)
    second = load_pdf_documents(POLICIES)
    first_ids = [document.metadata["chunk_id"] for document in first]

    assert len(first_ids) == len(set(first_ids))
    assert first_ids == [
        document.metadata["chunk_id"] for document in second
    ]


def test_splits_long_page_without_losing_page_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "policy.pdf"
    path.write_bytes(b"arquivo simulado")

    class FakePage:
        def extract_text(self) -> str:
            return "Primeiro parágrafo. " * 20 + "\n\n" + "Segundo. " * 20

    class FakeReader:
        pages = [FakePage()]

    monkeypatch.setattr(pdf_loader, "PdfReader", lambda _path: FakeReader())

    documents = load_pdf_documents(path, chunk_size=100, chunk_overlap=20)

    assert len(documents) > 1
    assert all(document.metadata["page"] == 1 for document in documents)
    assert documents[0].metadata["chunk_id"].endswith("part-000")


def test_rejects_pdf_without_extractable_text(tmp_path: Path) -> None:
    path = tmp_path / "empty.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with path.open("wb") as output:
        writer.write(output)

    with pytest.raises(ValueError, match="PDF sem texto extraível"):
        load_pdf_documents(path)


def test_rejects_non_pdf_file(tmp_path: Path) -> None:
    path = tmp_path / "policy.txt"
    path.write_text("texto", encoding="utf-8")

    with pytest.raises(ValueError, match="não é PDF"):
        load_pdf_documents(path)


def test_reports_missing_pdf() -> None:
    with pytest.raises(FileNotFoundError, match="não encontrado"):
        load_pdf_documents(POLICIES / "missing.pdf")


def test_validates_chunk_configuration() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        load_pdf_documents(REEMBOLSO, chunk_size=100, chunk_overlap=100)
