"""Orquestra Query Analyzer, filtros, FAISS, BM25, RRF e Top-K."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from src.dense_search import DenseSearchConfig
from src.dense_search import DenseSearchResponse
from src.dense_search import dense_search
from src.metadata_catalog import build_metadata_catalog
from src.query_analyzer import analyze_question
from src.query_analyzer_models import QueryAnalysis
from src.rrf import FusedSearchResult
from src.rrf import reciprocal_rank_fusion
from src.sparse_search import BM25SparseRetriever
from src.sparse_search import SparseSearchResult


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HybridRetrievalConfig:

    top_k: int = 5
    candidate_k: int = 20
    dense_fetch_k: int = 40
    rrf_rank_constant: int = 60
    dense_prefilter_selectivity_threshold: float = 0.15
    dense_max_fetch_k: int | None = None

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k deve ser maior que zero.")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k deve ser maior ou igual a top_k.")
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


@dataclass(frozen=True)
class HybridRetrievalResponse:

    question: str
    analysis: QueryAnalysis
    dense_response: DenseSearchResponse
    sparse_results: list[SparseSearchResult]
    results: list[FusedSearchResult]


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
        )
        fused_results = reciprocal_rank_fusion(
            dense_response.results,
            sparse_results,
            top_k=self._config.top_k,
            rank_constant=self._config.rrf_rank_constant,
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
