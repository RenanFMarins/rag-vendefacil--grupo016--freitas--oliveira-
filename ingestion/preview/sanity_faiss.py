"""Executa verificações de sanidade nos índices FAISS persistidos.

O script recarrega os índices existentes, sem reler os arquivos de origem e
sem gerar um novo índice. Cada pergunta é vetorizada uma única vez e pesquisada
nos seis índices; os resultados são reunidos em um top global.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ingestion.build._faiss import create_openai_embeddings
from ingestion.build._faiss import load_faiss_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATHS = {
    "csv": PROJECT_ROOT / "storage" / "faiss_csv",
    "json": PROJECT_ROOT / "storage" / "faiss_json",
    "jsonl": PROJECT_ROOT / "storage" / "faiss_jsonl",
    "markdown": PROJECT_ROOT / "storage" / "faiss_markdown",
    "pdf": PROJECT_ROOT / "storage" / "faiss_pdf",
    "txt": PROJECT_ROOT / "storage" / "faiss_txt",
}

TEST_QUESTIONS = (
    "Como realizar uma sangria no VendeFácil PDV?",
    (
        "Qual cliente relatou problema de sincronização de estoque "
        "entre matriz e filial?"
    ),
    "Qual é o prazo de atendimento para chamados críticos?",
)


@dataclass(frozen=True)
class SearchResult:
    """Representa um chunk encontrado em um dos índices."""

    index_name: str
    document: Document
    distance: float


def load_indexes(
    embeddings: Embeddings,
    index_paths: Mapping[str, Path] = INDEX_PATHS,
) -> dict[str, FAISS]:
    """Recarrega todos os índices informados sem reindexar os documentos."""
    return {
        name: load_faiss_index(path, embeddings=embeddings)
        for name, path in index_paths.items()
    }


def index_statistics(
    indexes: Mapping[str, FAISS],
) -> tuple[int, Counter[str]]:
    """Calcula o total de chunks e a distribuição global por doc_type."""
    doc_types: Counter[str] = Counter()
    total_chunks = 0

    for vectorstore in indexes.values():
        documents = vectorstore.docstore._dict.values()
        total_chunks += len(vectorstore.docstore._dict)
        doc_types.update(
            document.metadata.get("doc_type", "ausente")
            for document in documents
        )

    return total_chunks, doc_types


def search_all_indexes(
    question: str,
    indexes: Mapping[str, FAISS],
    embeddings: Embeddings,
    top_k: int = 5,
) -> list[SearchResult]:
    """Retorna os chunks globalmente mais próximos para uma pergunta."""
    query_vector = embeddings.embed_query(question)
    candidates: list[SearchResult] = []

    for index_name, vectorstore in indexes.items():
        local_k = min(top_k, vectorstore.index.ntotal)
        matches = vectorstore.similarity_search_with_score_by_vector(
            query_vector,
            k=local_k,
        )
        candidates.extend(
            SearchResult(
                index_name=index_name,
                document=document,
                distance=float(distance),
            )
            for document, distance in matches
        )

    # O FAISS usa distância: quanto menor o valor, maior a proximidade.
    candidates.sort(key=lambda result: result.distance)
    return candidates[:top_k]


def shorten_content(content: str, limit: int) -> str:
    """Limita o texto apenas para manter a saída do terminal legível."""
    normalized = " ".join(content.split())
    if limit == 0 or len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def print_result(
    position: int,
    result: SearchResult,
    content_limit: int,
) -> None:
    """Exibe um resultado e seus metadados mais importantes."""
    metadata = result.document.metadata
    print(f"\n  {position}. Índice: {result.index_name}")
    print(f"     Distância FAISS: {result.distance:.6f} (menor = mais similar)")
    print(f"     Arquivo: {metadata.get('source_file', 'ausente')}")
    print(f"     doc_type: {metadata.get('doc_type', 'ausente')}")
    print(f"     sensitivity: {metadata.get('sensitivity', 'ausente')}")
    print(f"     chunk_id: {metadata.get('chunk_id', 'ausente')}")

    optional_fields = (
        "customer_id",
        "ticket_id",
        "section",
        "priority",
        "status",
        "date",
    )
    for field in optional_fields:
        if value := metadata.get(field):
            print(f"     {field}: {value}")

    content = shorten_content(result.document.page_content, content_limit)
    print(f"     Texto: {content}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recarrega os índices FAISS e executa três buscas de sanidade."
        )
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Quantidade de chunks exibidos por pergunta (padrão: 5).",
    )
    parser.add_argument(
        "--content-limit",
        type=int,
        default=600,
        help=(
            "Máximo de caracteres do texto de cada chunk; use 0 para exibir "
            "o texto completo (padrão: 600)."
        ),
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k deve ser maior que zero.")
    if args.content_limit < 0:
        parser.error("--content-limit não pode ser negativo.")

    embeddings = create_openai_embeddings()
    indexes = load_indexes(embeddings)
    total_chunks, doc_types = index_statistics(indexes)

    print("SANIDADE DOS ÍNDICES FAISS")
    print("=" * 80)
    for name, vectorstore in indexes.items():
        print(f"- {name}: {vectorstore.index.ntotal} chunks")
    print(f"\nTotal de chunks: {total_chunks}")

    print("\nDistribuição por doc_type:")
    for doc_type, quantity in sorted(doc_types.items()):
        print(f"- {doc_type}: {quantity}")

    print("\nPerguntas de teste:")
    for number, question in enumerate(TEST_QUESTIONS, start=1):
        print(f"{number}. {question}")

    for number, question in enumerate(TEST_QUESTIONS, start=1):
        print("\n" + "=" * 80)
        print(f"PERGUNTA {number}: {question}")
        print(f"TOP {args.top_k} CHUNKS MAIS SIMILARES")
        results = search_all_indexes(
            question,
            indexes,
            embeddings,
            top_k=args.top_k,
        )
        for position, result in enumerate(results, start=1):
            print_result(position, result, args.content_limit)


if __name__ == "__main__":
    main()
