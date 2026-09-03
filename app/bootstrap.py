"""Carrega os recursos persistidos e monta o retrieval da Etapa 2."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from ingestion.build._faiss import create_openai_embeddings, load_faiss_index
from src.config import INDEX_PATHS
from src.retrieval.pipeline import HybridRetrievalConfig, HybridRetriever
from src.retrieval.query_analyzer import create_chat_model


@dataclass(frozen=True)
class RetrievalRuntime:
    retriever: HybridRetriever
    document_count: int
    index_sizes: dict[str, int]


def load_persisted_indexes(
    embeddings: Embeddings,
    index_paths: Mapping[str, Path] = INDEX_PATHS,
) -> dict[str, FAISS]:
    return {
        name: load_faiss_index(path, embeddings=embeddings)
        for name, path in sorted(index_paths.items())
    }


def collect_indexed_documents(
    indexes: Mapping[str, FAISS],
) -> list[Document]:
    documents: list[Document] = []
    known_chunk_ids: set[str] = set()

    for index_name, vectorstore in sorted(indexes.items()):
        for position, document_id in sorted(
            vectorstore.index_to_docstore_id.items()
        ):
            document = vectorstore.docstore.search(document_id)
            if not isinstance(document, Document):
                raise ValueError(
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


def build_retrieval_runtime(
    *,
    top_k: int = 5,
    candidate_k: int = 20,
    embeddings: Embeddings | None = None,
    llm: BaseChatModel | None = None,
    index_paths: Mapping[str, Path] = INDEX_PATHS,
) -> RetrievalRuntime:
    if candidate_k < top_k:
        raise ValueError("candidate_k deve ser maior ou igual a top_k.")

    embedding_client = embeddings or create_openai_embeddings()
    chat_model = llm or create_chat_model()
    indexes = load_persisted_indexes(embedding_client, index_paths)
    documents = collect_indexed_documents(indexes)
    retriever = HybridRetriever(
        indexes,
        embedding_client,
        documents,
        llm=chat_model,
        config=HybridRetrievalConfig(
            top_k=top_k,
            candidate_k=candidate_k,
            dense_fetch_k=max(40, candidate_k),
        ),
    )
    return RetrievalRuntime(
        retriever=retriever,
        document_count=len(documents),
        index_sizes={
            name: vectorstore.index.ntotal
            for name, vectorstore in sorted(indexes.items())
        },
    )
