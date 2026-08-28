"""Exibe os textos, metadados e vetores persistidos em um índice FAISS."""

import argparse
from pathlib import Path

import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import DeterministicFakeEmbedding


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATHS = {
    "jsonl": PROJECT_ROOT / "storage" / "faiss_jsonl",
    "txt": PROJECT_ROOT / "storage" / "faiss_txt",
    "markdown": PROJECT_ROOT / "storage" / "faiss_markdown",
}


def load_index_for_preview(index_path: Path):
    """Carrega um índice local sem gerar embeddings ou chamar uma API."""
    faiss_path = index_path / "index.faiss"
    pickle_path = index_path / "index.pkl"
    if not faiss_path.is_file() or not pickle_path.is_file():
        raise FileNotFoundError(
            f"Índice incompleto em {index_path}. "
            "Esperado: index.faiss e index.pkl."
        )

    raw_index = faiss.read_index(str(faiss_path))
    preview_embeddings = DeterministicFakeEmbedding(size=raw_index.d)
    vectorstore = FAISS.load_local(
        str(index_path),
        preview_embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mostra o conteúdo persistido em um índice FAISS local."
    )
    parser.add_argument("format", choices=sorted(INDEX_PATHS))
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Quantidade de chunks exibidos.",
    )
    parser.add_argument(
        "--show-vectors",
        action="store_true",
        help="Mostra os 10 primeiros valores numéricos de cada vetor.",
    )
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit deve ser maior que zero.")

    index_path = INDEX_PATHS[args.format]
    vectorstore = load_index_for_preview(index_path)
    stored_documents = list(vectorstore.docstore._dict.items())

    print(f"Índice: {index_path}")
    print(f"Total de vetores: {vectorstore.index.ntotal}")
    print(f"Dimensões por vetor: {vectorstore.index.d}")
    print(f"Documents no index.pkl: {len(stored_documents)}")

    for position, (document_id, document) in enumerate(
        stored_documents[: args.limit]
    ):
        print("\n" + "=" * 80)
        print(f"POSIÇÃO NO FAISS: {position}")
        print(f"DOCUMENT ID: {document_id}")
        print("=" * 80)
        print("\nMETADADOS:\n")
        for key, value in document.metadata.items():
            print(f"- {key}: {value}")
        print("\nTEXTO:\n")
        print(document.page_content)

        if args.show_vectors:
            vector = vectorstore.index.reconstruct(position)
            values = ", ".join(f"{value:.6f}" for value in vector[:10])
            print("\nPRIMEIROS 10 VALORES DO VETOR:\n")
            print(f"[{values}, ...]")


if __name__ == "__main__":
    main()
