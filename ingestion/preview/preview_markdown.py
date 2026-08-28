import argparse
from collections import Counter
from pathlib import Path

from ingestion.loaders.markdown_loader import load_markdown_documents


DEFAULT_MARKDOWN_DIRECTORY = (
    Path(__file__).parents[2]
    / "data"
    / "unstructured"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exibe os Documents gerados pelo loader Markdown."
    )
    parser.add_argument(
        "file_or_directory",
        nargs="?",
        type=Path,
        default=DEFAULT_MARKDOWN_DIRECTORY,
        help="Caminho de um arquivo Markdown ou de um diretório.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Quantidade de Documents que serão exibidos.",
    )
    parser.add_argument(
        "--doc-type",
        choices=("manual", "ata", "policy"),
        help="Exibe somente um tipo de documento.",
    )
    parser.add_argument(
        "--sensitivity",
        choices=("publico", "interno", "restrito"),
        help="Exibe somente uma classificação de sensibilidade.",
    )
    args = parser.parse_args()

    documents = load_markdown_documents(args.file_or_directory)
    chunk_ids = {document.metadata["chunk_id"] for document in documents}
    doc_type_distribution = Counter(
        document.metadata["doc_type"] for document in documents
    )
    sensitivity_distribution = Counter(
        document.metadata["sensitivity"] for document in documents
    )

    selected_documents = documents
    if args.doc_type:
        selected_documents = [
            document
            for document in selected_documents
            if document.metadata["doc_type"] == args.doc_type
        ]
    if args.sensitivity:
        selected_documents = [
            document
            for document in selected_documents
            if document.metadata["sensitivity"] == args.sensitivity
        ]

    print(f"Origem: {args.file_or_directory}")
    print(f"Total de Documents: {len(documents)}")
    print(f"Chunk IDs únicos: {len(chunk_ids)}")
    print("Distribuição por doc_type:")
    for doc_type, total in sorted(doc_type_distribution.items()):
        print(f"- {doc_type}: {total}")
    print("Distribuição por sensibilidade:")
    for sensitivity, total in sorted(sensitivity_distribution.items()):
        print(f"- {sensitivity}: {total}")

    if args.doc_type or args.sensitivity:
        print(f"Documents selecionados pelos filtros: {len(selected_documents)}")

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
