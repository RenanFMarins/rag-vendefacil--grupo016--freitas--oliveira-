"""Constrói os Documents de tickets provenientes de JSONL."""

import argparse
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ingestion.build._common import print_summary, validate_documents
from ingestion.build._faiss import save_faiss_index
from ingestion.loaders.jsonl_loader import load_jsonl_documents


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "semi_structured" / "tickets.jsonl"
DEFAULT_INDEX = PROJECT_ROOT / "storage" / "faiss_jsonl"


def build_jsonl_documents(
    source: str | Path = DEFAULT_SOURCE,
) -> list[Document]:
    """Carrega tickets JSONL e valida chunks prontos para indexação."""
    documents = load_jsonl_documents(source)
    return validate_documents(documents, "jsonl")


def build_jsonl_index(
    source: str | Path = DEFAULT_SOURCE,
    index_path: str | Path = DEFAULT_INDEX,
    embeddings: Embeddings | None = None,
):
    """Constrói e persiste o índice FAISS dos tickets JSONL."""
    documents = build_jsonl_documents(source)
    return save_faiss_index(documents, index_path, embeddings)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Constrói e valida os Documents do arquivo JSONL."
    )
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--save-index",
        action="store_true",
        help="Gera embeddings e salva o índice FAISS do formato.",
    )
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args()

    documents = build_jsonl_documents(args.source)
    print_summary(documents, "jsonl")
    if args.save_index:
        vectorstore = save_faiss_index(documents, args.index_path)
        print(f"Vetores indexados: {vectorstore.index.ntotal}")
        print(f"Índice salvo em: {args.index_path}")


if __name__ == "__main__":
    main()
