import argparse
from collections import Counter
from pathlib import Path

from ingestion.loaders.txt_loader import load_txt_documents


DEFAULT_EMAIL_DIRECTORY = (
    Path(__file__).parents[2]
    / "data"
    / "unstructured"
    / "emails"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exibe os Documents gerados pelo loader TXT."
    )
    parser.add_argument(
        "file_or_directory",
        nargs="?",
        type=Path,
        default=DEFAULT_EMAIL_DIRECTORY,
        help="Caminho de um arquivo TXT ou de um diretório com e-mails.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Quantidade de Documents que serão exibidos.",
    )
    parser.add_argument(
        "--only-restricted",
        action="store_true",
        help="Exibe somente Documents classificados como restritos.",
    )
    args = parser.parse_args()

    documents = load_txt_documents(args.file_or_directory)
    chunk_ids = {document.metadata["chunk_id"] for document in documents}
    sensitivity_distribution = Counter(
        document.metadata["sensitivity"] for document in documents
    )

    selected_documents = documents
    if args.only_restricted:
        selected_documents = [
            document
            for document in documents
            if document.metadata["sensitivity"] == "restrito"
        ]

    print(f"Origem: {args.file_or_directory}")
    print(f"Total de Documents: {len(documents)}")
    print(f"Chunk IDs únicos: {len(chunk_ids)}")
    print("Distribuição por sensibilidade:")
    for sensitivity, total in sorted(sensitivity_distribution.items()):
        print(f"- {sensitivity}: {total}")

    if args.only_restricted:
        print(f"Documents restritos selecionados: {len(selected_documents)}")

    for position, document in enumerate(
        selected_documents[: args.limit],
        start=1,
    ):
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
