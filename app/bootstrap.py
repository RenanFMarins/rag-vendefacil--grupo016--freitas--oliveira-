"""Monta o pipeline RAG estruturado usado pela aplicação Streamlit."""

from collections.abc import Mapping
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ingestion.build._faiss import create_openai_embeddings
from ingestion.build._faiss import load_faiss_index
from src.config import INDEX_PATHS
from src.config import load_project_environment
from src.generation.generator import create_generation_model
from src.guardrails.scope import create_scope_model
from src.pipeline import RAGPipeline
from src.retrieval.pipeline import HybridRetrievalConfig
from src.retrieval.pipeline import HybridRetriever
from src.retrieval.query_analyzer import create_chat_model


def _validate_index_paths(index_paths: Mapping[str, Path]) -> None:
    if not index_paths:
        raise ValueError("Ao menos um índice deve ser configurado.")

    missing = [
        name
        for name, path in index_paths.items()
        if not (path / "index.faiss").is_file()
        or not (path / "index.pkl").is_file()
    ]
    if missing:
        names = ", ".join(sorted(missing))
        raise FileNotFoundError(
            f"Índices obrigatórios ausentes ou incompletos: {names}."
        )


def load_persisted_indexes(
    embeddings: Embeddings,
    index_paths: Mapping[str, Path] = INDEX_PATHS,
) -> dict[str, FAISS]:
    """Carrega todos os índices configurados sem reindexar o corpus."""
    _validate_index_paths(index_paths)
    return {
        name: load_faiss_index(path, embeddings=embeddings)
        for name, path in sorted(index_paths.items())
    }


def documents_from_indexes(
    indexes: Mapping[str, FAISS],
) -> list[Document]:
    """Reúne Documents persistidos e exige chunk_ids globais únicos."""
    documents: list[Document] = []
    known_chunk_ids: set[str] = set()

    for index_name, vectorstore in sorted(indexes.items()):
        for position, document_id in sorted(
            vectorstore.index_to_docstore_id.items()
        ):
            document = vectorstore.docstore.search(document_id)
            if not isinstance(document, Document):
                raise ValueError(
                    "Document ausente no índice "
                    f"{index_name}, posição {position}."
                )

            chunk_id = str(document.metadata.get("chunk_id", "")).strip()
            if not chunk_id:
                raise ValueError(
                    f"Document sem chunk_id no índice {index_name}."
                )
            if chunk_id in known_chunk_ids:
                raise ValueError(f"chunk_id duplicado nos índices: {chunk_id}")

            known_chunk_ids.add(chunk_id)
            documents.append(document)

    if not documents:
        raise ValueError("Nenhum Document foi encontrado nos índices.")
    return documents


def build_rag_pipeline(
    index_paths: Mapping[str, Path] = INDEX_PATHS,
    *,
    top_k: int = 5,
    candidate_k: int = 20,
) -> RAGPipeline:
    """Carrega recursos existentes e devolve o pipeline oficial pronto."""
    if candidate_k < top_k:
        raise ValueError("candidate_k deve ser maior ou igual a top_k.")

    load_project_environment()
    _validate_index_paths(index_paths)
    embeddings = create_openai_embeddings()
    indexes = load_persisted_indexes(embeddings, index_paths)
    documents = documents_from_indexes(indexes)
    retriever = HybridRetriever(
        indexes,
        embeddings,
        documents,
        llm=create_chat_model(),
        config=HybridRetrievalConfig(
            top_k=top_k,
            candidate_k=candidate_k,
            dense_fetch_k=max(40, candidate_k),
        ),
    )
    return RAGPipeline(
        retriever,
        scope_llm=create_scope_model(),
        generation_llm=create_generation_model(),
    )
