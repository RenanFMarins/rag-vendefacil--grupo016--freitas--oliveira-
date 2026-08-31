from retrieval.hybrid import busca_hibrida
from retrieval.filters import extract_metadata, extrair_filtros_manuais, extrair_filtros_automatico, combinar_filtros, filtrar_documento
from retrieval.bm25 import busca_bm25
from retrieval.densa import busca_densa
from retrieval.rrf import fusao_rrf
from ingest import abrir_banco


def imprimir_resultados(titulo, resultados):

    print(f"\n{'=' * 60}")
    print(titulo)
    print(f"{'=' * 60}")

    for posicao, (documento, score) in enumerate(resultados, start=1):

        print(f"\n--- Resultado {posicao} ---")
        print(f"Score: {score}")
        print(f"Metadata: {documento.metadata}")
        print(f"Conteúdo: {documento.page_content[:300]}...")


def comparar_com_sem_filtro(pergunta, documentos):
    print("\n")
    print("#" * 70)
    print("COMPARATIVO: COM FILTRO x SEM FILTRO")
    print("#" * 70)

    resultados_sem_filtro = busca_hibrida(
        pergunta,
        documentos
    )

    filtros_manuais = extrair_filtros_manuais(pergunta)

    dados_extraidos = extract_metadata(documentos)

    filtros_automaticos = extrair_filtros_automatico(
        pergunta,
        dados_extraidos
    )

    filtros = combinar_filtros(
        filtros_manuais,
        filtros_automaticos
    )

    documentos_filtrados = filtrar_documento(documentos, filtros)

    resultados_com_filtro = busca_hibrida(
        pergunta,
        documentos_filtrados
    )

    print("\nFILTROS EXTRAÍDOS:")
    print(filtros)

    print("\nDOCUMENTOS:")
    print(f"Total sem filtro: {len(documentos)}")
    print(f"Total com filtro: {len(documentos_filtrados)}")

    imprimir_resultados(
        "SEM FILTRO",
        resultados_sem_filtro
    )

    imprimir_resultados(
        "COM FILTRO",
        resultados_com_filtro
    )


db = abrir_banco()

documentos = list(
    db.docstore._dict.values()
)
"""

perguntas = [
    "Quais lojas de Minas Gerais possuem o módulo de estoque?",

    "Quais lojas do Rio de Janeiro utilizam o módulo PDV?",

    "Quais tickets de clientes de Minas Gerais estão relacionados ao módulo de estoque?"
]

for pergunta in perguntas:

    comparar_com_sem_filtro(
        pergunta,
        documentos
    )

"""


def comparar_bm25_dense_rrf(pergunta, documentos):

    print("\n")
    print("#" * 70)
    print("COMPARATIVO: BM25 x DENSE x RRF")
    print("#" * 70)

    filtros_manuais = extrair_filtros_manuais(pergunta)

    dados_extraidos = extract_metadata(documentos)

    filtros_automaticos = extrair_filtros_automatico(
        pergunta,
        dados_extraidos
    )

    filtros = combinar_filtros(
        filtros_manuais,
        filtros_automaticos
    )

    documentos_filtrados = filtrar_documento(documentos, filtros)

    resultados_bm25 = busca_bm25(
        pergunta, documentos_filtrados
    )

    documentos_bm25 = [documento for documento, score in resultados_bm25]

    resultados_densa = busca_densa(
        pergunta, documentos_filtrados
    )

    documentos_densa = [documento for documento, score in resultados_densa]

    ranking_rrf = fusao_rrf(
        documentos_bm25,
        documentos_densa
    )

    ranking_rrf = ranking_rrf[:5]

    imprimir_resultados(
        "BM25",
        resultados_bm25
    )

    imprimir_resultados(
        "DENSE",
        resultados_densa
    )

    imprimir_resultados(
        "RRF",
        ranking_rrf
    )


perguntas = "Quais lojas de Minas Gerais possuem o módulo de estoque?"


comparar_bm25_dense_rrf(perguntas, documentos)
