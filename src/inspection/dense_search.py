"""Compara a busca densa sem filtro e com filtros validados."""

import argparse

from ingestion.build._faiss import create_openai_embeddings
from ingestion.preview.sanity_faiss import load_indexes
from src.inspection.metadata_catalog import load_indexed_documents
from src.retrieval.dense import DenseSearchConfig
from src.retrieval.dense import DenseSearchResponse
from src.retrieval.dense import dense_search_by_vector
from src.retrieval.metadata_catalog import build_metadata_catalog
from src.retrieval.models import QueryFilters
from src.retrieval.query_analyzer import analyze_question


TEST_QUESTIONS = (
    (
        "Quais tickets de clientes de Minas Gerais estão relacionados "
        "ao módulo de estoque?"
    ),
    "Quais tickets de São Paulo estão relacionados ao módulo Pay?",
    "Quais informações existem sobre o módulo de estoque?",
)


def print_response(title: str, response: DenseSearchResponse) -> None:
    print(f"\n{title}")
    for diagnostic in response.diagnostics:
        print(
            f"- {diagnostic.index_name}: strategy={diagnostic.strategy}, "
            f"eligible={diagnostic.eligible_documents}/"
            f"{diagnostic.total_documents}, "
            f"selectivity={diagnostic.selectivity:.3f}, "
            f"fetch_k={diagnostic.fetch_k}"
        )

    for position, result in enumerate(response.results, start=1):
        metadata = result.document.metadata
        print(
            f"  {position}. [{result.index_name}] "
            f"distance={result.distance:.6f} "
            f"source={metadata.get('source_file')} "
            f"chunk_id={metadata.get('chunk_id')} "
            f"state={metadata.get('state')} "
            f"module={metadata.get('module')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara Dense Search sem e com filtros validados."
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--fetch-k", type=int, default=20)
    parser.add_argument("--prefilter-threshold", type=float, default=0.15)
    parser.add_argument("--max-fetch-k", type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = DenseSearchConfig(
        k=args.k,
        fetch_k=args.fetch_k,
        prefilter_selectivity_threshold=args.prefilter_threshold,
        max_fetch_k=args.max_fetch_k,
    )
    documents = load_indexed_documents()
    metadata_catalog = build_metadata_catalog(documents)
    embeddings = create_openai_embeddings()
    indexes = load_indexes(embeddings)

    for question in TEST_QUESTIONS:
        analysis = analyze_question(
            question,
            metadata_catalog,
            debug=args.debug,
        )
        query_vector = embeddings.embed_query(analysis.query)
        unfiltered = dense_search_by_vector(
            query_vector,
            indexes,
            validated_filters=QueryFilters(),
            config=config,
        )
        filtered = dense_search_by_vector(
            query_vector,
            indexes,
            validated_filters=analysis.filters,
            config=config,
        )

        print("\n" + "=" * 100)
        print(f"Pergunta: {question}")
        print(f"Query semântica: {analysis.query}")
        print(
            "Filtros validados: "
            f"{analysis.filters.model_dump(exclude_none=True)}"
        )
        print_response("SEM FILTRO", unfiltered)
        print_response("COM FILTRO", filtered)


if __name__ == "__main__":
    main()
