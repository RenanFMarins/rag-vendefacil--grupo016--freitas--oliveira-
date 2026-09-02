"""Fusão de rankings Dense e BM25 com Reciprocal Rank Fusion."""

from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.documents import Document

from src.retrieval.dense import DenseSearchResult
from src.retrieval.sparse import SparseSearchResult


@dataclass(frozen=True)
class FusedSearchResult:

    document: Document
    score: float
    rank: int
    dense_rank: int | None
    sparse_rank: int | None
    dense_distance: float | None
    sparse_score: float | None

    @property
    def chunk_id(self) -> str:
        return str(self.document.metadata["chunk_id"])

    @property
    def matched_retrievers(self) -> tuple[str, ...]:
        retrievers: list[str] = []
        if self.dense_rank is not None:
            retrievers.append("dense")
        if self.sparse_rank is not None:
            retrievers.append("bm25")
        return tuple(retrievers)


@dataclass
class _FusionCandidate:
    document: Document
    score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_distance: float | None = None
    sparse_score: float | None = None


def _add_dense_ranking(
    candidates: dict[str, _FusionCandidate],
    results: Sequence[DenseSearchResult],
    rank_constant: int,
) -> None:
    seen: set[str] = set()
    for rank, result in enumerate(results, start=1):
        if result.chunk_id in seen:
            continue
        seen.add(result.chunk_id)
        candidate = candidates.setdefault(
            result.chunk_id,
            _FusionCandidate(document=result.document),
        )
        candidate.score += 1 / (rank_constant + rank)
        candidate.dense_rank = rank
        candidate.dense_distance = result.distance


def _add_sparse_ranking(
    candidates: dict[str, _FusionCandidate],
    results: Sequence[SparseSearchResult],
    rank_constant: int,
) -> None:
    seen: set[str] = set()
    for rank, result in enumerate(results, start=1):
        if result.chunk_id in seen:
            continue
        seen.add(result.chunk_id)
        candidate = candidates.setdefault(
            result.chunk_id,
            _FusionCandidate(document=result.document),
        )
        candidate.score += 1 / (rank_constant + rank)
        candidate.sparse_rank = rank
        candidate.sparse_score = result.score


def reciprocal_rank_fusion(
    dense_results: Sequence[DenseSearchResult],
    sparse_results: Sequence[SparseSearchResult],
    *,
    top_k: int = 5,
    rank_constant: int = 60,
) -> list[FusedSearchResult]:
    if top_k < 1:
        raise ValueError("top_k deve ser maior que zero.")
    if rank_constant < 1:
        raise ValueError("rank_constant deve ser maior que zero.")

    candidates: dict[str, _FusionCandidate] = {}
    _add_dense_ranking(candidates, dense_results, rank_constant)
    _add_sparse_ranking(candidates, sparse_results, rank_constant)

    ordered = sorted(
        candidates.items(),
        key=lambda item: (
            -item[1].score,
            min(
                rank
                for rank in (item[1].dense_rank, item[1].sparse_rank)
                if rank is not None
            ),
            item[0],
        ),
    )

    return [
        FusedSearchResult(
            document=candidate.document,
            score=candidate.score,
            rank=rank,
            dense_rank=candidate.dense_rank,
            sparse_rank=candidate.sparse_rank,
            dense_distance=candidate.dense_distance,
            sparse_score=candidate.sparse_score,
        )
        for rank, (_chunk_id, candidate) in enumerate(
            ordered[:top_k],
            start=1,
        )
    ]

