"""Busca densa nos índices FAISS com filtragem explícita e adaptativa."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.query_analyzer_models import QueryFilters


@dataclass(frozen=True)
class DenseSearchConfig:
    k: int = 5
    fetch_k: int = 20
    oversampling_factor: float = 1.5
    prefilter_selectivity_threshold: float = 0.15
    max_fetch_k: int | None = None

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k deve ser maior que zero.")
        if self.fetch_k < self.k:
            raise ValueError("fetch_k deve ser maior ou igual a k.")
        if self.oversampling_factor < 1:
            raise ValueError("oversampling_factor deve ser pelo menos 1.")
        if not 0 <= self.prefilter_selectivity_threshold <= 1:
            raise ValueError(
                "prefilter_selectivity_threshold deve estar entre 0 e 1."
            )
        if self.max_fetch_k is not None and self.max_fetch_k < self.k:
            raise ValueError("max_fetch_k deve ser maior ou igual a k.")


@dataclass(frozen=True)
class DenseSearchResult:

    index_name: str
    document: Document
    distance: float

    @property
    def chunk_id(self) -> str:
        return str(self.document.metadata["chunk_id"])


@dataclass(frozen=True)
class DenseSearchDiagnostic:

    index_name: str
    total_documents: int
    eligible_documents: int
    selectivity: float
    strategy: str
    fetch_k: int


@dataclass(frozen=True)
class DenseSearchResponse:
    results: list[DenseSearchResult]
    diagnostics: list[DenseSearchDiagnostic]


def _matches_filters(document: Document, filters: Mapping[str, object]) -> bool:
    return all(
        document.metadata.get(field) == value
        for field, value in filters.items()
    )


def _eligible_documents(
    vectorstore: FAISS,
    filters: Mapping[str, object],
) -> list[tuple[int, Document]]:
    eligible: list[tuple[int, Document]] = []
    for position, document_id in sorted(
        vectorstore.index_to_docstore_id.items()
    ):
        document = vectorstore.docstore.search(document_id)
        if not isinstance(document, Document):
            raise ValueError(
                f"Document não encontrado no docstore: {document_id}"
            )
        if _matches_filters(document, filters):
            eligible.append((position, document))
    return eligible


def _exact_prefilter_search(
    vectorstore: FAISS,
    query_vector: list[float],
    eligible: list[tuple[int, Document]],
    k: int,
) -> list[tuple[Document, float]]:
    if vectorstore.index.metric_type != faiss.METRIC_L2:
        raise ValueError("A pré-filtragem exata requer um índice FAISS L2.")

    query = np.asarray(query_vector, dtype=np.float32)
    matches: list[tuple[Document, float]] = []
    for position, document in eligible:
        vector = np.asarray(
            vectorstore.index.reconstruct(position),
            dtype=np.float32,
        )
        difference = query - vector
        distance = float(np.dot(difference, difference))
        matches.append((document, distance))

    matches.sort(key=lambda item: item[1])
    return matches[:k]


def _adaptive_fetch_k(
    *,
    target_k: int,
    selectivity: float,
    total_documents: int,
    config: DenseSearchConfig,
) -> int:
    estimated = math.ceil(
        (target_k / selectivity) * config.oversampling_factor
    )
    fetch_k = max(config.fetch_k, estimated)
    limit = config.max_fetch_k or total_documents
    return min(fetch_k, limit, total_documents)


def dense_search_by_vector(
    query_vector: list[float],
    indexes: Mapping[str, FAISS],
    validated_filters: QueryFilters | None = None,
    config: DenseSearchConfig | None = None,
) -> DenseSearchResponse:
    search_config = config or DenseSearchConfig()
    filters = (
        validated_filters.model_dump(exclude_none=True)
        if validated_filters
        else {}
    )
    candidates: list[DenseSearchResult] = []
    diagnostics: list[DenseSearchDiagnostic] = []

    for index_name, vectorstore in indexes.items():
        total_documents = vectorstore.index.ntotal
        if total_documents == 0:
            continue

        if not filters:
            local_k = min(search_config.k, total_documents)
            matches = vectorstore.similarity_search_with_score_by_vector(
                query_vector,
                k=local_k,
            )
            diagnostics.append(
                DenseSearchDiagnostic(
                    index_name=index_name,
                    total_documents=total_documents,
                    eligible_documents=total_documents,
                    selectivity=1.0,
                    strategy="unfiltered",
                    fetch_k=local_k,
                )
            )
        else:
            eligible = _eligible_documents(vectorstore, filters)
            eligible_count = len(eligible)
            selectivity = eligible_count / total_documents
            target_k = min(search_config.k, eligible_count)

            if eligible_count == 0:
                diagnostics.append(
                    DenseSearchDiagnostic(
                        index_name=index_name,
                        total_documents=total_documents,
                        eligible_documents=0,
                        selectivity=0.0,
                        strategy="no_matches",
                        fetch_k=0,
                    )
                )
                continue

            if selectivity <= search_config.prefilter_selectivity_threshold:
                matches = _exact_prefilter_search(
                    vectorstore,
                    query_vector,
                    eligible,
                    target_k,
                )
                strategy = "exact_prefilter"
                effective_fetch_k = 0
            else:
                effective_fetch_k = _adaptive_fetch_k(
                    target_k=target_k,
                    selectivity=selectivity,
                    total_documents=total_documents,
                    config=search_config,
                )
                matches = vectorstore.similarity_search_with_score_by_vector(
                    query_vector,
                    k=target_k,
                    filter=filters,
                    fetch_k=effective_fetch_k,
                )
                strategy = "adaptive_postfilter"

                if len(matches) < target_k:
                    matches = _exact_prefilter_search(
                        vectorstore,
                        query_vector,
                        eligible,
                        target_k,
                    )
                    strategy = "postfilter_fallback_exact"

            diagnostics.append(
                DenseSearchDiagnostic(
                    index_name=index_name,
                    total_documents=total_documents,
                    eligible_documents=eligible_count,
                    selectivity=selectivity,
                    strategy=strategy,
                    fetch_k=effective_fetch_k,
                )
            )

        candidates.extend(
            DenseSearchResult(
                index_name=index_name,
                document=document,
                distance=float(distance),
            )
            for document, distance in matches
        )

    candidates.sort(key=lambda result: result.distance)
    return DenseSearchResponse(
        results=candidates[: search_config.k],
        diagnostics=diagnostics,
    )


def dense_search(
    query: str,
    indexes: Mapping[str, FAISS],
    embeddings: Embeddings,
    validated_filters: QueryFilters | None = None,
    config: DenseSearchConfig | None = None,
) -> DenseSearchResponse:
    query_vector = embeddings.embed_query(query)
    return dense_search_by_vector(
        query_vector,
        indexes,
        validated_filters=validated_filters,
        config=config,
    )
