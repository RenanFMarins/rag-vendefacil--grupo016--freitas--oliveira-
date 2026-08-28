"""Validações compartilhadas pelos construtores de documentos."""

from collections import Counter

from langchain_core.documents import Document


REQUIRED_METADATA = {
    "source_file",
    "doc_type",
    "chunk_id",
    "sensitivity",
}
VALID_SENSITIVITIES = {"publico", "interno", "restrito"}


def validate_documents(
    documents: list[Document],
    format_name: str,
) -> list[Document]:
    """Confirma o contrato comum antes de entregar chunks à indexação."""
    if not documents:
        raise ValueError(f"O construtor {format_name} não gerou documentos.")

    chunk_ids: set[str] = set()
    for position, document in enumerate(documents, start=1):
        if not document.page_content.strip():
            raise ValueError(
                f"O documento {position} de {format_name} está vazio."
            )

        missing = REQUIRED_METADATA - document.metadata.keys()
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(
                f"O documento {position} de {format_name} não possui: {fields}."
            )

        sensitivity = document.metadata["sensitivity"]
        if sensitivity not in VALID_SENSITIVITIES:
            raise ValueError(
                f"Sensitivity inválida em {format_name}: {sensitivity}."
            )

        chunk_id = document.metadata["chunk_id"]
        if chunk_id in chunk_ids:
            raise ValueError(
                f"chunk_id duplicado no construtor {format_name}: {chunk_id}."
            )
        chunk_ids.add(chunk_id)

    return documents


def print_summary(documents: list[Document], format_name: str) -> None:
    """Mostra uma sanidade curta quando o construtor roda pelo terminal."""
    by_doc_type = Counter(doc.metadata["doc_type"] for doc in documents)
    by_sensitivity = Counter(
        doc.metadata["sensitivity"] for doc in documents
    )

    print(f"Formato: {format_name}")
    print(f"Total de chunks: {len(documents)}")
    print(f"Distribuição por doc_type: {dict(by_doc_type)}")
    print(f"Distribuição por sensitivity: {dict(by_sensitivity)}")
