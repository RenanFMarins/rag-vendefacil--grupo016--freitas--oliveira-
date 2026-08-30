"""Executa e exibe o pipeline completo de recuperação da Etapa 2."""

import argparse
import logging

from ingestion.build._faiss import create_openai_embeddings
from ingestion.preview.sanity_faiss import load_indexes
from src.hybrid_retrieval import HybridRetrievalConfig
from src.hybrid_retrieval import HybridRetriever
from src.inspect_metadata_catalog import load_indexed_documents


TEST_QUESTIONS = (
    "Quais tickets de Minas Gerais estão relacionados ao módulo de estoque?",
    "Quais informações existem sobre o ticket TCK-1005?",
    "Como recolher periodicamente o dinheiro acumulado pelo atendente?",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspeciona Query Analyzer, FAISS, BM25, RRF e Top-K."
    )
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.INFO)

    documents = load_indexed_documents()
    embeddings = create_openai_embeddings()
    indexes = load_indexes(embeddings)
    retriever = HybridRetriever(
        indexes,
        embeddings,
        documents,
        config=HybridRetrievalConfig(
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            dense_fetch_k=max(40, args.candidate_k),
        ),
    )

    print(f"Documents carregados: {len(documents)}")
    for question in args.queries or TEST_QUESTIONS:
        response = retriever.retrieve(question, debug=args.debug)
        filters = response.analysis.filters.model_dump(exclude_none=True)

        print("\n" + "=" * 100)
        print(f"Pergunta: {response.question}")
        print(f"Query semântica: {response.analysis.query}")
        print(f"Filtros válidos: {filters}")
        if response.analysis.rejected_filters:
            rejected_filters = [
                item.model_dump()
                for item in response.analysis.rejected_filters
            ]
            print(
                "Filtros rejeitados: "
                f"{rejected_filters}"
            )
        print(
            f"Candidatos: dense={len(response.dense_response.results)}, "
            f"bm25={len(response.sparse_results)}"
        )
        print("Top-K após RRF:")
        for result in response.results:
            print(
                f"  {result.rank}. chunk_id={result.chunk_id} "
                f"rrf={result.score:.6f} "
                f"dense_rank={result.dense_rank} "
                f"bm25_rank={result.sparse_rank} "
                f"retrievers={result.matched_retrievers} "
                f"source={result.document.metadata.get('source_file')}"
            )


if __name__ == "__main__":
    main()
