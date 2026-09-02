"""Mostra os Documents produzidos pelo loader JSON."""

import argparse
from pathlib import Path

from ingestion.loaders.json_loader import load_json_documents


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "structured"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exibe os Documents gerados pelo loader JSON."
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_PATH,
    )
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit deve ser maior que zero.")

    documents = load_json_documents(args.file_path)
    chunk_ids = {document.metadata["chunk_id"] for document in documents}
    print(f"Origem: {args.file_path}")
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
