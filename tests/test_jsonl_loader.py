import json
from pathlib import Path

import pytest

from ingestion.loaders.jsonl_loader import load_jsonl_documents


DATA_FILE = (
    Path(__file__).parents[1]
    / "data"
    / "semi_structured"
    / "tickets.jsonl"
)

REQUIRED_METADATA = {
    "source_file",
    "doc_type",
    "chunk_id",
    "sensitivity",
}


def _ticket_fixture(**overrides: object) -> dict[str, object]:
    ticket: dict[str, object] = {
        "ticket_id": "TCK-TEST",
        "customer_id": "CUST-TEST",
        "customer_name": "Cliente de Teste",
        "state": "MG",
        "module": "estoque",
        "title": "Falha de sincronização",
        "priority": "Alta",
        "status": "Aberto",
        "created_at": "2026-02-10T09:30:00Z",
        "category": "Sincronização / API",
        "description": "O saldo da filial não foi atualizado.",
        "resolution": None,
        "sentiment": "Insatisfeito",
    }
    ticket.update(overrides)
    return ticket


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    content = "\n".join(
        json.dumps(record, ensure_ascii=False) for record in records
    )
    path.write_text(content, encoding="utf-8")


def test_loads_one_document_per_ticket_from_current_dataset() -> None:
    documents = load_jsonl_documents(DATA_FILE)

    assert len(documents) == 75
    assert len({doc.metadata["chunk_id"] for doc in documents}) == 75
    assert all(doc.metadata["chunk_id"].endswith(":000") for doc in documents)


def test_applies_required_and_ticket_metadata() -> None:
    document = load_jsonl_documents(DATA_FILE)[0]

    assert REQUIRED_METADATA <= document.metadata.keys()
    assert document.metadata == {
        "source_file": "tickets.jsonl",
        "doc_type": "ticket",
        "sensitivity": "interno",
        "ticket_id": "TCK-1001",
        "customer_id": "CUST001",
        "state": "MG",
        "module": "estoque",
        "priority": "Alta",
        "status": "Aberto",
        "category": "Sincronização / API",
        "sentiment": "Insatisfeito",
        "date": "2026-02-10",
        "chunk_id": "tickets:TCK-1001:000",
    }


def test_serializes_ticket_as_natural_language() -> None:
    content = load_jsonl_documents(DATA_FILE)[0].page_content

    assert "Ticket TCK-1001" in content
    assert "Cliente: Supermercado Boa Compra (CUST001), estado MG." in content
    assert "Módulo: estoque. Prioridade: Alta. Status: Aberto." in content
    assert "Descrição:" in content
    assert "Resolução: Ainda não informada." in content


def test_generates_stable_chunk_ids() -> None:
    first_load = load_jsonl_documents(DATA_FILE)
    second_load = load_jsonl_documents(DATA_FILE)

    first_ids = [doc.metadata["chunk_id"] for doc in first_load]
    second_ids = [doc.metadata["chunk_id"] for doc in second_load]

    assert first_ids == second_ids


def test_splits_only_long_body_and_repeats_header(tmp_path: Path) -> None:
    file_path = tmp_path / "long_ticket.jsonl"
    long_description = " ".join(
        f"Informação relevante {number}." for number in range(30)
    )
    _write_jsonl(
        file_path,
        [_ticket_fixture(description=long_description)],
    )

    documents = load_jsonl_documents(
        file_path,
        chunk_size=120,
        chunk_overlap=20,
    )

    assert len(documents) > 1
    assert all(
        doc.page_content.startswith("Ticket TCK-TEST — Falha de sincronização")
        for doc in documents
    )
    assert [doc.metadata["chunk_id"] for doc in documents] == [
        f"long_ticket:TCK-TEST:{part_number:03d}"
        for part_number in range(len(documents))
    ]
    assert all(doc.metadata["ticket_id"] == "TCK-TEST" for doc in documents)


def test_reports_file_and_line_for_invalid_json(tmp_path: Path) -> None:
    file_path = tmp_path / "invalid.jsonl"
    valid_line = json.dumps(_ticket_fixture(), ensure_ascii=False)
    file_path.write_text(f"{valid_line}\n{{json inválido}}", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        load_jsonl_documents(file_path)

    assert str(file_path) in str(error.value)
    assert "linha 2" in str(error.value)
