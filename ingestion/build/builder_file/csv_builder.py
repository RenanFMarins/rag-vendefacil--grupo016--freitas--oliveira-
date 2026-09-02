"""Constrói os Documents e o índice dos arquivos CSV."""

import argparse
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ingestion.build._common import print_summary, validate_documents
from ingestion.build._faiss import save_faiss_index
from ingestion.loaders.csv_loader import load_csv_documents


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = PROJECT_ROOT / "data"
DEFAULT_INDEX = PROJECT_ROOT / "storage" / "faiss_csv"


def build_csv_documents(
    source: str | Path = DEFAULT_SOURCE,
) -> list[Document]:
    documents = load_csv_documents(source)
    return validate_documents(documents, "csv")


def build_csv_index(
    source: str | Path = DEFAULT_SOURCE,
    index_path: str | Path = DEFAULT_INDEX,
    embeddings: Embeddings | None = None,
):
    documents = build_csv_documents(source)
    return save_faiss_index(documents, index_path, embeddings)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Constrói e valida os Documents dos arquivos CSV."
    )
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--save-index", action="store_true")
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args()

    documents = build_csv_documents(args.source)
    print_summary(documents, "csv")
    if args.save_index:
        vectorstore = save_faiss_index(documents, args.index_path)
        print(f"Vetores indexados: {vectorstore.index.ntotal}")
        print(f"Índice salvo em: {args.index_path}")


if __name__ == "__main__":
    main()
