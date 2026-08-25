import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _serialize_header(ticket: dict[str, Any]) -> str:
    return (
        f"Ticket {ticket['ticket_id']} — {ticket['title']}\n"
        f"Cliente: {ticket['customer_name']} "
        f"({ticket['customer_id']}), estado {ticket['state']}.\n"
        f"Módulo: {ticket['module']}. "
        f"Prioridade: {ticket['priority']}. "
        f"Status: {ticket['status']}.\n"
        f"Categoria: {ticket['category']}.\n"
        f"Sentimento do cliente: {ticket['sentiment']}."
    )


def _serialize_body(ticket: dict[str, Any]) -> str:
    description = ticket.get("description") or "Não informada."
    resolution = ticket.get("resolution") or "Ainda não informada."

    return (
        f"Descrição: {description}\n"
        f"Resolução: {resolution}"
    )


def load_jsonl_documents(
    file_path: str | Path,
    chunk_size: int = 1200,
    chunk_overlap: int = 120,
) -> list[Document]:
    path = Path(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: list[Document] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                ticket = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON inválido em {path}, linha {line_number}"
                ) from error

            ticket_id = ticket["ticket_id"]
            header = _serialize_header(ticket)
            body = _serialize_body(ticket)

            # Apenas o corpo pode ser dividido.
            body_parts = splitter.split_text(body) or [body]

            base_metadata = {
                "source_file": path.name,
                "doc_type": "ticket",
                "sensitivity": "interno",
                "ticket_id": ticket_id,
                "customer_id": ticket["customer_id"],
                "state": ticket["state"],
                "module": ticket["module"],
                "priority": ticket["priority"],
                "status": ticket["status"],
                "category": ticket["category"],
                "sentiment": ticket["sentiment"],
                "date": ticket["created_at"][:10],
            }

            for part_number, body_part in enumerate(body_parts):
                metadata = {
                    **base_metadata,
                    "chunk_id": (
                        f"{path.stem}:{ticket_id}:{part_number:03d}"
                    ),
                }

                documents.append(
                    Document(
                        page_content=f"{header}\n{body_part}",
                        metadata=metadata,
                    )
                )

    return documents
