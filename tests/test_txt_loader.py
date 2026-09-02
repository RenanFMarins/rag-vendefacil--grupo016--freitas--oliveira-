from collections import Counter
from pathlib import Path

import pytest

from ingestion.loaders.txt_loader import load_txt_documents


EMAIL_DIRECTORY = (
    Path(__file__).parents[1]
    / "data"
    / "unstructured"
    / "emails"
)

REQUIRED_METADATA = {
    "source_file",
    "doc_type",
    "chunk_id",
    "sensitivity",
}

RESTRICTED_FILES = {
    "customer_027_envio_credenciais_acesso_admin.txt",
    "customer_028_envio_senha_certificado_digital.txt",
    "internal_013_compartilhamento_chave_api_producao.txt",
    "internal_014_envio_credenciais_banco_dados_prod.txt",
    "internal_015_senha_root_servidores_tef.txt",
}


def _email_fixture(
    *,
    sender: str = "cliente@empresa.com.br",
    recipients: str = "suporte@vendefacil.com.br",
    date: str = "10 de Fevereiro de 2026 09:15",
    subject: str = "Falha de sincronização - TCK-9999",
    body: str = "Cliente CUST999 / MG relatou falha no estoque.",
) -> str:
    return (
        f"De: {sender}\n"
        f"Para: {recipients}\n"
        f"Data: {date}\n"
        f"Assunto: {subject}\n\n"
        f"{body}"
    )


def test_loads_one_document_per_current_email() -> None:
    documents = load_txt_documents(EMAIL_DIRECTORY)

    assert len(documents) == 43
    assert len({doc.metadata["chunk_id"] for doc in documents}) == 43
    assert all(doc.page_content.strip() for doc in documents)
    assert all(REQUIRED_METADATA <= doc.metadata.keys() for doc in documents)


def test_extracts_relationships_and_normalizes_date() -> None:
    file_path = EMAIL_DIRECTORY / "customer_001_sincronizacao.txt"

    document = load_txt_documents(file_path)[0]

    assert document.metadata["source_file"] == file_path.name
    assert document.metadata["doc_type"] == "email"
    assert document.metadata["sensitivity"] == "interno"
    assert document.metadata["customer_id"] == "CUST001"
    assert document.metadata["ticket_id"] == "TCK-1001"
    assert document.metadata["state"] == "MG"
    assert document.metadata["module"] == "estoque"
    assert document.metadata["date"] == "2026-02-10"
    assert document.metadata["chunk_id"] == (
        "customer_001_sincronizacao:msg-000:part-000"
    )


def test_omits_ticket_id_when_email_does_not_have_one() -> None:
    file_path = EMAIL_DIRECTORY / "customer_002_reembolso_cancelamento.txt"

    document = load_txt_documents(file_path)[0]

    assert document.metadata["customer_id"] == "CUST0010"
    assert document.metadata["state"] == "RS"
    assert "ticket_id" not in document.metadata


def test_classifies_only_credential_emails_as_restricted() -> None:
    documents = load_txt_documents(EMAIL_DIRECTORY)
    distribution = Counter(
        document.metadata["sensitivity"] for document in documents
    )
    restricted_files = {
        document.metadata["source_file"]
        for document in documents
        if document.metadata["sensitivity"] == "restrito"
    }

    assert distribution == {"interno": 38, "restrito": 5}
    assert restricted_files == RESTRICTED_FILES


def test_splits_thread_into_independent_messages(tmp_path: Path) -> None:
    file_path = tmp_path / "thread.txt"
    first_message = _email_fixture(
        subject="Primeira mensagem - TCK-1001",
        body="Cliente CUST001 / MG relatou problema no estoque.",
    )
    second_message = _email_fixture(
        sender="suporte@vendefacil.com.br",
        recipients="cliente@empresa.com.br",
        subject="Resposta ao ticket TCK-1001",
        body="A equipe corrigiu a sincronização do estoque.",
    )
    file_path.write_text(
        f"{first_message}\n\n{second_message}",
        encoding="utf-8",
    )

    documents = load_txt_documents(file_path)

    assert len(documents) == 2
    assert [doc.metadata["chunk_id"] for doc in documents] == [
        "thread:msg-000:part-000",
        "thread:msg-001:part-000",
    ]
    assert documents[0].metadata["sender"] == "cliente@empresa.com.br"
    assert documents[1].metadata["sender"] == "suporte@vendefacil.com.br"


def test_splits_only_long_body_and_repeats_header(tmp_path: Path) -> None:
    file_path = tmp_path / "long_email.txt"
    long_body = " ".join(
        f"Informação relevante sobre o estoque {number}."
        for number in range(30)
    )
    file_path.write_text(
        _email_fixture(body=long_body),
        encoding="utf-8",
    )

    documents = load_txt_documents(
        file_path,
        chunk_size=120,
        chunk_overlap=20,
    )

    assert len(documents) > 1
    assert all(
        doc.page_content.startswith("E-mail de: cliente@empresa.com.br")
        for doc in documents
    )
    assert [doc.metadata["chunk_id"] for doc in documents] == [
        f"long_email:msg-000:part-{part_number:03d}"
        for part_number in range(len(documents))
    ]


def test_marks_every_part_of_sensitive_message_as_restricted(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "sensitive_email.txt"
    body = (
        " ".join("Contexto interno do atendimento." for _ in range(20))
        + "\nSenha Administrador: MinhaSenha123!"
    )
    file_path.write_text(_email_fixture(body=body), encoding="utf-8")

    documents = load_txt_documents(
        file_path,
        chunk_size=100,
        chunk_overlap=10,
    )

    assert len(documents) > 1
    assert all(
        doc.metadata["sensitivity"] == "restrito"
        for doc in documents
    )


def test_reports_missing_required_header(tmp_path: Path) -> None:
    file_path = tmp_path / "invalid_email.txt"
    file_path.write_text(
        "De: cliente@empresa.com.br\n"
        "Para: suporte@vendefacil.com.br\n"
        "Data: 10 de Fevereiro de 2026 09:15\n\n"
        "Mensagem sem assunto.",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        load_txt_documents(file_path)

    assert str(file_path) in str(error.value)
    assert "assunto" in str(error.value)
