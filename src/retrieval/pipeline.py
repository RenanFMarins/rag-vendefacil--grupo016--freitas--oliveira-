"""Orquestra Query Analyzer, filtros, FAISS, BM25, RRF e Top-K."""

import logging
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from src.retrieval.dense import DenseSearchConfig
from src.retrieval.dense import DenseSearchResponse
from src.retrieval.dense import dense_search
from src.retrieval.fusion import FusedSearchResult
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.metadata_catalog import build_metadata_catalog
from src.retrieval.models import QueryAnalysis
from src.retrieval.query_analyzer import analyze_question
from src.retrieval.query_analyzer import requested_document_types
from src.retrieval.sparse import BM25SparseRetriever
from src.retrieval.sparse import SparseSearchResult
from src.retrieval.sparse import tokenize_for_bm25


LOGGER = logging.getLogger(__name__)
EXACT_PINNED_METADATA_FIELDS = (
    "customer_id",
    "ticket_id",
    "employee_id",
    "store_id",
    "product_id",
    "sale_id",
    "error_code",
    "priority",
)
EXACT_PINNED_METADATA_LIMITS = {
    "error_code": 1,
}


@dataclass(frozen=True)
class HybridRetrievalConfig:

    top_k: int = 5
    multi_document_top_k: int | None = None
    candidate_k: int = 20
    dense_fetch_k: int = 40
    rrf_rank_constant: int = 60
    dense_prefilter_selectivity_threshold: float = 0.15
    dense_max_fetch_k: int | None = None
    pin_exact_identifier_matches: bool = True

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k deve ser maior que zero.")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k deve ser maior ou igual a top_k.")
        if (
            self.multi_document_top_k is not None
            and self.multi_document_top_k < self.top_k
        ):
            raise ValueError(
                "multi_document_top_k deve ser maior ou igual a top_k."
            )
        if (
            self.multi_document_top_k is not None
            and self.candidate_k < self.multi_document_top_k
        ):
            raise ValueError(
                "candidate_k deve ser maior ou igual a multi_document_top_k."
            )
        if self.dense_fetch_k < self.candidate_k:
            raise ValueError(
                "dense_fetch_k deve ser maior ou igual a candidate_k."
            )
        if self.rrf_rank_constant < 1:
            raise ValueError("rrf_rank_constant deve ser maior que zero.")
        if not 0 <= self.dense_prefilter_selectivity_threshold <= 1:
            raise ValueError(
                "dense_prefilter_selectivity_threshold deve estar "
                "entre 0 e 1."
            )
        if (
            self.dense_max_fetch_k is not None
            and self.dense_max_fetch_k < self.candidate_k
        ):
            raise ValueError(
                "dense_max_fetch_k deve ser maior ou igual a candidate_k."
            )

    @property
    def effective_multi_document_top_k(self) -> int:
        if self.multi_document_top_k is not None:
            return self.multi_document_top_k
        return max(self.top_k, min(8, self.candidate_k))


@dataclass(frozen=True)
class HybridRetrievalResponse:

    question: str
    analysis: QueryAnalysis
    dense_response: DenseSearchResponse
    sparse_results: list[SparseSearchResult]
    results: list[FusedSearchResult]


def _exact_metadata_chunk_ids(
    query: str,
    sparse_results: Sequence[SparseSearchResult],
) -> list[str]:
    query_tokens = set(tokenize_for_bm25(query))
    matches: list[str] = []
    matches_by_field: dict[str, int] = {}
    for result in sparse_results:
        for field in EXACT_PINNED_METADATA_FIELDS:
            value = result.document.metadata.get(field)
            if value is None:
                continue
            if not any(
                token in query_tokens
                for token in tokenize_for_bm25(str(value))
            ):
                continue

            limit = EXACT_PINNED_METADATA_LIMITS.get(field)
            current_count = matches_by_field.get(field, 0)
            if limit is not None and current_count >= limit:
                continue

            matches.append(result.chunk_id)
            matches_by_field[field] = current_count + 1
            break
    return matches


def _prioritize_exact_identifier_matches(
    results: Sequence[FusedSearchResult],
    pinned_chunk_ids: Sequence[str],
    *,
    top_k: int,
) -> list[FusedSearchResult]:
    by_chunk_id = {result.chunk_id: result for result in results}
    pinned = [
        chunk_id
        for chunk_id in pinned_chunk_ids
        if chunk_id in by_chunk_id
    ]
    pinned_set = set(pinned)
    ordered_ids = [
        *pinned,
        *(
            result.chunk_id
            for result in results
            if result.chunk_id not in pinned_set
        ),
    ]
    return [
        replace(by_chunk_id[chunk_id], rank=rank)
        for rank, chunk_id in enumerate(ordered_ids[:top_k], start=1)
    ]


class HybridRetriever:
    def __init__(
        self,
        indexes: Mapping[str, FAISS],
        embeddings: Embeddings,
        documents: Sequence[Document],
        *,
        llm: BaseChatModel | None = None,
        config: HybridRetrievalConfig | None = None,
    ) -> None:
        if not indexes:
            raise ValueError("Ao menos um índice FAISS deve ser informado.")
        if not documents:
            raise ValueError("Ao menos um Document deve ser informado.")

        self._indexes = dict(indexes)
        self._embeddings = embeddings
        self._metadata_catalog = build_metadata_catalog(documents)
        self._sparse_retriever = BM25SparseRetriever(documents)
        self._llm = llm
        self._config = config or HybridRetrievalConfig()

    @property
    def metadata_catalog(self) -> dict[str, list[object]]:
        return {
            field: list(values)
            for field, values in self._metadata_catalog.items()
        }

    def retrieve(
        self,
        question: str,
        *,
        debug: bool = False,
        allowed_sensitivities: Collection[str] | None = None,
    ) -> HybridRetrievalResponse:
        analysis = analyze_question(
            question,
            self._metadata_catalog,
            llm=self._llm,
            debug=debug,
        )
        filters = analysis.filters
        dense_response = dense_search(
            analysis.query,
            self._indexes,
            self._embeddings,
            validated_filters=filters,
            allowed_sensitivities=allowed_sensitivities,
            config=DenseSearchConfig(
                k=self._config.candidate_k,
                fetch_k=self._config.dense_fetch_k,
                prefilter_selectivity_threshold=(
                    self._config.dense_prefilter_selectivity_threshold
                ),
                max_fetch_k=self._config.dense_max_fetch_k,
            ),
        )
        sparse_results = self._sparse_retriever.search(
            analysis.query,
            k=self._config.candidate_k,
            validated_filters=filters,
            allowed_sensitivities=allowed_sensitivities,
        )
        requested_types = requested_document_types(question)
        effective_top_k = (
            self._config.effective_multi_document_top_k
            if len(requested_types) >= 2
            else self._config.top_k
        )
        all_fused_results = reciprocal_rank_fusion(
            dense_response.results,
            sparse_results,
            top_k=max(
                1,
                len(dense_response.results) + len(sparse_results),
            ),
            rank_constant=self._config.rrf_rank_constant,
        )
        pinned_chunk_ids = (
            _exact_metadata_chunk_ids(analysis.query, sparse_results)
            if self._config.pin_exact_identifier_matches
            else []
        )
        fused_results = _prioritize_exact_identifier_matches(
            all_fused_results,
            pinned_chunk_ids,
            top_k=effective_top_k,
        )

        if debug:
            LOGGER.info("Candidatos Dense: %d", len(dense_response.results))
            LOGGER.info("Candidatos BM25: %d", len(sparse_results))
            LOGGER.info("Resultados Top-K: %d", len(fused_results))

        return HybridRetrievalResponse(
            question=question.strip(),
            analysis=analysis,
            dense_response=dense_response,
            sparse_results=sparse_results,
            results=fused_results,
        )

