"""Cria uma base FAISS de demonstração a partir de um arquivo Markdown."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "data"
    / "unstructured"
    / "documentation"
    / "pdv"
    / "manual_pdv.md"
)
DEFAULT_INDEX = PROJECT_ROOT / "storage" / "faiss_markdown_demo"


def cosine_relevance_score(squared_l2_distance: float) -> float:
    """Converte a distância L2 quadrática de vetores unitários em cosseno."""
    return max(0.0, min(1.0, 1.0 - squared_l2_distance / 2.0))


def split_markdown(source_path: Path):
    """Divide o Markdown por cabeçalhos e limita seções muito extensas."""
    markdown = source_path.read_text(encoding="utf-8")
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "title"),
            ("##", "section"),
            ("###", "subsection"),
        ],
        strip_headers=False,
    )
    section_documents = header_splitter.split_text(markdown)

    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = size_splitter.split_documents(section_documents)

    relative_source = source_path.relative_to(PROJECT_ROOT).as_posix()
    for position, chunk in enumerate(chunks):
        stable_value = f"{relative_source}:{position}:{chunk.page_content}"
        chunk.metadata.update(
            {
                "source_file": source_path.name,
                "source_path": relative_source,
                "doc_type": "manual",
                "chunk_id": hashlib.sha256(
                    stable_value.encode("utf-8")
                ).hexdigest()[:16],
                "sensitivity": "publico",
                "module": "pdv",
            }
        )

    return chunks


def build_index(source_path: Path, index_path: Path) -> int:
    """Vetoriza os chunks e persiste o índice FAISS em disco."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("A variável OPENAI_API_KEY não foi configurada.")

    embedding_model = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    chunks = split_markdown(source_path)
    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=api_key)
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings,
        relevance_score_fn=cosine_relevance_score,
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    return len(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cria um índice FAISS de teste usando um Markdown."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    args = parse_args()
    source_path = args.source.resolve()
    index_path = args.index.resolve()

    if not source_path.is_file():
        raise FileNotFoundError(f"Markdown não encontrado: {source_path}")

    total = build_index(source_path, index_path)
    print(f"Índice criado com {total} chunks.")
    print(f"Fonte: {source_path}")
    print(f"Salvo em: {index_path}")


if __name__ == "__main__":
    main()
