import csv
from pathlib import Path

import pytest

from ingestion.loaders.csv_loader import load_csv_documents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

CUSTOMERS = DATA_DIR / "structured" / "customers.csv"
EMPLOYEES = DATA_DIR / "structured" / "employees.csv"
LOGS = DATA_DIR / "semi_structured" / "system_logs.csv"
SALES = DATA_DIR / "structured" / "sales.csv"


def test_loads_one_document_per_csv_record() -> None:
    assert len(load_csv_documents(CUSTOMERS)) == 2000
    assert len(load_csv_documents(EMPLOYEES)) == 10
    assert len(load_csv_documents(LOGS)) == 450
    assert len(load_csv_documents(SALES)) == 3000


def test_serializes_customer_in_natural_language() -> None:
    document = load_csv_documents(CUSTOMERS)[0]

    assert document.page_content.startswith("Cliente CUST001:")
    assert "Plano" in document.page_content
    assert "MRR de R$" in document.page_content
    assert document.metadata["doc_type"] == "customer"
    assert document.metadata["sensitivity"] == "interno"
    assert document.metadata["chunk_id"] == "customers:CUST001"
    assert document.metadata["module"] in {
        "analytics",
        "ecommerce",
        "estoque",
        "pay",
        "pdv",
    }


def test_normalizes_vendefacil_loja_as_ecommerce() -> None:
    documents = load_csv_documents(CUSTOMERS)
    customer = next(
        document
        for document in documents
        if document.metadata["chunk_id"] == "customers:CUST004"
    )

    assert customer.metadata["module"] == "ecommerce"


def test_employee_salary_is_restricted() -> None:
    document = load_csv_documents(EMPLOYEES)[0]

    assert "salário individual" in document.page_content
    assert document.metadata["doc_type"] == "employee"
    assert document.metadata["sensitivity"] == "restrito"


def test_log_chunk_ids_are_unique_and_deterministic() -> None:
    first = load_csv_documents(LOGS)
    second = load_csv_documents(LOGS)
    first_ids = [document.metadata["chunk_id"] for document in first]

    assert len(first_ids) == len(set(first_ids)) == 450
    assert first_ids == [
        document.metadata["chunk_id"] for document in second
    ]
    assert first_ids[0] == "system_logs:row-000001"


def test_marks_credential_log_as_restricted(tmp_path: Path) -> None:
    path = tmp_path / "system_logs.csv"
    fieldnames = [
        "timestamp",
        "level",
        "service",
        "module",
        "customer_id",
        "event",
        "error_code",
        "message",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "2026-01-01T10:00:00",
                "level": "ERROR",
                "service": "auth",
                "module": "pdv",
                "customer_id": "CUST001",
                "event": "credential_exposure",
                "error_code": "AUTH-401",
                "message": "Senha registrada indevidamente no log.",
            }
        )

    document = load_csv_documents(path)[0]

    assert document.metadata["sensitivity"] == "restrito"


def test_loads_directory_in_deterministic_order(tmp_path: Path) -> None:
    for source in (EMPLOYEES, CUSTOMERS):
        destination = tmp_path / source.name
        destination.write_bytes(source.read_bytes())

    documents = load_csv_documents(tmp_path)

    assert documents[0].metadata["source_file"] == "customers.csv"
    assert documents[-1].metadata["source_file"] == "employees.csv"


def test_rejects_unknown_csv(tmp_path: Path) -> None:
    path = tmp_path / "unknown.csv"
    path.write_text("id,name\n1,Teste\n", encoding="utf-8")

    with pytest.raises(ValueError, match="CSV não reconhecido"):
        load_csv_documents(path)


def test_reports_missing_required_column(tmp_path: Path) -> None:
    path = tmp_path / "customers.csv"
    path.write_text("customer_id,company_name\nCUST001,Teste\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Colunas ausentes"):
        load_csv_documents(path)
