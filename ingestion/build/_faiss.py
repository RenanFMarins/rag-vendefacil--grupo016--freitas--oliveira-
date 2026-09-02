"""Cria, persiste e recarrega os índices FAISS dos construtores."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_openai_embeddings() -> OpenAIEmbeddings:
    """Cria o cliente de embeddings usando somente variáveis de ambiente."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("A variável OPENAI_API_KEY não foi configurada.")

    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model, api_key=api_key)


def save_faiss_index(
    documents: list[Document],
    index_path: str | Path,
    embeddings: Embeddings | None = None,
) -> FAISS:
    """Vetoriza Documents e salva index.faiss e index.pkl em disco."""
    path = Path(index_path)
    embedding_client = embeddings or create_openai_embeddings()
    chunk_ids = [document.metadata["chunk_id"] for document in documents]

    vectorstore = FAISS.from_documents(
        documents,
        embedding_client,
        ids=chunk_ids,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(path))
    return vectorstore


def load_faiss_index(
    index_path: str | Path,
    embeddings: Embeddings | None = None,
) -> FAISS:
    """Recarrega um índice existente sem reler ou reindexar as fontes."""
    path = Path(index_path)
    if not (path / "index.faiss").is_file():
        raise FileNotFoundError(f"index.faiss não encontrado em: {path}")
    if not (path / "index.pkl").is_file():
        raise FileNotFoundError(f"index.pkl não encontrado em: {path}")

    embedding_client = embeddings or create_openai_embeddings()
    return FAISS.load_local(
        str(path),
        embedding_client,
        allow_dangerous_deserialization=True,
    )
