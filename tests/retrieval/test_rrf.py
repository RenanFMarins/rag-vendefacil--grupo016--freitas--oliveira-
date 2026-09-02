import pytest
from langchain_core.documents import Document

from src.retrieval.dense import DenseSearchResult
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.sparse import SparseSearchResult


def make_document(chunk_id: str) -> Document:
    return Document(
        page_content=f"Conteúdo de {chunk_id}",
        metadata={"chunk_id": chunk_id},
    )


def dense_result(chunk_id: str, distance: float) -> DenseSearchResult:
    return DenseSearchResult(
        index_name="test",
        document=make_document(chunk_id),
        distance=distance,
    )


def sparse_result(
    chunk_id: str,
    score: float,
    rank: int,
) -> SparseSearchResult:
    return SparseSearchResult(
        document=make_document(chunk_id),
        score=score,
        rank=rank,
    )


def test_fuses_by_chunk_id_and_rewards_presence_in_both_rankings() -> None:
    dense = [dense_result("dense-only", 0.1), dense_result("shared", 0.2)]
    sparse = [sparse_result("shared", 8.0, 1), sparse_result("bm25-only", 7.0, 2)]

    results = reciprocal_rank_fusion(
        dense,
        sparse,
        top_k=3,
        rank_constant=60,
    )

    assert [result.chunk_id for result in results] == [
        "shared",
        "dense-only",
        "bm25-only",
    ]
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert results[0].dense_rank == 2
    assert results[0].sparse_rank == 1
    assert results[0].matched_retrievers == ("dense", "bm25")


def test_applies_top_k_and_preserves_original_diagnostics() -> None:
    dense = [dense_result("a", 0.12), dense_result("b", 0.34)]
    sparse = [sparse_result("b", 9.5, 1)]

    results = reciprocal_rank_fusion(dense, sparse, top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "b"
    assert results[0].dense_distance == 0.34
    assert results[0].sparse_score == 9.5


def test_handles_one_empty_retriever_and_uses_deterministic_tie_break() -> None:
    sparse = [
        sparse_result("chunk-b", 2.0, 1),
        sparse_result("chunk-a", 2.0, 2),
    ]

    results = reciprocal_rank_fusion([], sparse, top_k=2)

    assert [result.chunk_id for result in results] == ["chunk-b", "chunk-a"]
    assert all(result.dense_rank is None for result in results)


def test_breaks_equal_rrf_scores_deterministically_by_chunk_id() -> None:
    dense = [dense_result("chunk-b", 0.1)]
    sparse = [sparse_result("chunk-a", 9.0, 1)]

    results = reciprocal_rank_fusion(dense, sparse, top_k=2)

    assert results[0].score == results[1].score
    assert [result.chunk_id for result in results] == ["chunk-a", "chunk-b"]


def test_deduplicates_repeated_chunk_inside_the_same_ranking() -> None:
    dense = [dense_result("same", 0.1), dense_result("same", 0.2)]

    results = reciprocal_rank_fusion(dense, [], top_k=5)

    assert len(results) == 1
    assert results[0].score == pytest.approx(1 / 61)


@pytest.mark.parametrize(
    ("top_k", "rank_constant"),
    [(0, 60), (5, 0)],
)
def test_rejects_invalid_configuration(
    top_k: int,
    rank_constant: int,
) -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            [],
            [],
            top_k=top_k,
            rank_constant=rank_constant,
        )

