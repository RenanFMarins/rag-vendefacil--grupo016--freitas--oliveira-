"""Compara FAISS e BM25 sem realizar fusão entre os resultados."""

import argparse

from ingestion.build._faiss import create_openai_embeddings
from ingestion.preview.sanity_faiss import load_indexes
from src.dense_search import DenseSearchConfig
from src.dense_search import dense_search
from src.inspect_metadata_catalog import load_indexed_documents
from src.sparse_search import BM25SparseRetriever


TEST_QUESTIONS = (
    "TCK-1005",
    "Como recolher periodicamente o dinheiro acumulado pelo atendente?",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara resultados independentes de FAISS e BM25."
    )
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    documents = load_indexed_documents()
    sparse_retriever = BM25SparseRetriever(documents)
    embeddings = create_openai_embeddings()
    indexes = load_indexes(embeddings)
    questions = args.queries or TEST_QUESTIONS

    print(f"Documents no BM25: {sparse_retriever.document_count}")
    for question in questions:
        dense_results = dense_search(
            question,
            indexes,
            embeddings,
            config=DenseSearchConfig(k=args.k, fetch_k=max(20, args.k)),
        ).results
        sparse_results = sparse_retriever.search(question, k=args.k)

        print("\n" + "=" * 100)
        print(f"Pergunta enviada aos dois retrievers: {question}")
        print("\nDENSE / FAISS (menor distância é melhor)")
        for position, result in enumerate(dense_results, start=1):
            print(
                f"  {position}. chunk_id={result.chunk_id} "
                f"distance={result.distance:.6f} "
                f"source={result.document.metadata.get('source_file')}"
            )

        print("\nSPARSE / BM25 (maior score é melhor)")
        for result in sparse_results:
            print(
                f"  {result.rank}. chunk_id={result.chunk_id} "
                f"score={result.score:.6f} "
                f"source={result.document.metadata.get('source_file')}"
            )


if __name__ == "__main__":
    main()
