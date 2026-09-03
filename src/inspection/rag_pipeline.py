"""Executa o pipeline integrado das Etapas 2 e 3 pelo terminal."""

import argparse
import logging

from ingestion.build._faiss import create_openai_embeddings
from ingestion.preview.sanity_faiss import load_indexes
from src.inspection.metadata_catalog import load_indexed_documents
from src.pipeline import RAGPipeline
from src.retrieval.pipeline import HybridRetrievalConfig
from src.retrieval.pipeline import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa escopo, LGPD, retrieval, geração e validação."
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--aggregate-group-size", type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.WARNING)
        logging.getLogger("src.pipeline").setLevel(logging.INFO)
        logging.getLogger("src.generation.generator").setLevel(logging.INFO)

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
    pipeline = RAGPipeline(retriever)
    response = pipeline.answer(
        args.query,
        aggregate_group_size=args.aggregate_group_size,
        debug=args.debug,
    )
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

