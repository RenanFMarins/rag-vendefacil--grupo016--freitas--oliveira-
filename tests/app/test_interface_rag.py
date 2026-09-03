from pathlib import Path

from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

from app.interface_rag import serialize_response
from src.retrieval.dense import DenseSearchDiagnostic, DenseSearchResponse
from src.retrieval.fusion import FusedSearchResult
from src.retrieval.models import QueryAnalysis, QueryFilters
from src.retrieval.pipeline import HybridRetrievalResponse
from src.retrieval.sparse import SparseSearchResult


def test_serialize_response_preserves_retrieval_evidence():
    document = Document(
        page_content="Trecho literal recuperado.",
        metadata={
            "chunk_id": "manual-001",
            "source_file": "manual.md",
            "doc_type": "manual",
            "sensitivity": "interno",
            "section": "Caixa",
        },
    )
    response = HybridRetrievalResponse(
        question="Como operar o caixa?",
        analysis=QueryAnalysis(
            query="operação do caixa",
            filters=QueryFilters(doc_type="manual"),
        ),
        dense_response=DenseSearchResponse(
            results=[],
            diagnostics=[
                DenseSearchDiagnostic(
                    index_name="markdown",
                    total_documents=10,
                    eligible_documents=4,
                    selectivity=0.4,
                    strategy="adaptive_postfilter",
                    fetch_k=10,
                )
            ],
        ),
        sparse_results=[
            SparseSearchResult(document=document, score=2.0, rank=1)
        ],
        results=[
            FusedSearchResult(
                document=document,
                score=0.02,
                rank=1,
                dense_rank=None,
                sparse_rank=1,
                dense_distance=None,
                sparse_score=2.0,
            )
        ],
    )

    record = serialize_response(response)

    assert record["question"] == "Como operar o caixa?"
    assert record["semantic_query"] == "operação do caixa"
    assert record["filters"] == {"doc_type": "manual"}
    assert record["results"][0]["chunk_id"] == "manual-001"
    assert record["results"][0]["content"] == document.page_content
    assert record["results"][0]["retrievers"] == ("bm25",)


def test_streamlit_interface_starts_without_loading_external_resources():
    project_root = Path(__file__).resolve().parents[2]
    app = AppTest.from_file(project_root / "app" / "interface_rag.py")

    app.run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "🔎 Retrieval híbrido VendeFácil"
    assert "Etapa 2" in app.caption[0].value
