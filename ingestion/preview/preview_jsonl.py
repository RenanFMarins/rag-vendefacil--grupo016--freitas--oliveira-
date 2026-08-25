import argparse
from pathlib import Path

from ingestion.loaders.jsonl_loader import load_jsonl_documents


DEFAULT_DATA_FILE = (
    Path(__file__).parents[2]
    / "data"
    / "semi_structured"
    / "tickets.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exibe os Documents gerados pelo loader JSONL."
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_FILE,
        help="Caminho do arquivo JSONL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Quantidade de Documents que serão exibidos.",
    )
    args = parser.parse_args()

    documents = load_jsonl_documents(args.file_path)
    chunk_ids = {document.metadata["chunk_id"] for document in documents}

    print(f"Arquivo: {args.file_path}")
    print(f"Total de Documents: {len(documents)}")
    print(f"Chunk IDs únicos: {len(chunk_ids)}")

    for position, document in enumerate(documents[: args.limit], start=1):
        print("\n" + "=" * 80)
        print(f"DOCUMENTO {position}")
        print("=" * 80)
        print("\nTEXTO VETORIZÁVEL:\n")
        print(document.page_content)
        print("\nMETADADOS:\n")
        for key, value in document.metadata.items():
            print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
