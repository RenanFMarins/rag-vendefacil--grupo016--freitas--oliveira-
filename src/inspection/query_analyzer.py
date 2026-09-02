
import argparse
import json
import logging

from src.inspection.metadata_catalog import load_indexed_documents
from src.retrieval.metadata_catalog import build_metadata_catalog
from src.retrieval.query_analyzer import analyze_question


TEST_QUESTIONS = (
    (
        "Quais tickets de clientes de Minas Gerais estão relacionados "
        "ao módulo de estoque?"
    ),
    "Mostre tickets de alta prioridade.",
    "Quais informações existem sobre problemas de pagamento?",
    "Quero documentos internos.",
    "Quais tickets de MG estão abertos?",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interpreta perguntas sem executar retrieval."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Pergunta única; se omitida, executa as cinco perguntas de teste.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Exibe pergunta, query semântica e filtros nos logs.",
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    documents = load_indexed_documents()
    metadata_catalog = build_metadata_catalog(documents)
    questions = (args.question,) if args.question else TEST_QUESTIONS

    for question in questions:
        analysis = analyze_question(
            question,
            metadata_catalog,
            debug=args.debug,
        )
        print("\n" + "=" * 80)
        print(f"Pergunta: {question}")
        print(
            json.dumps(
                analysis.model_dump(exclude_none=True),
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
