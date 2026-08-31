from query_analyzer import busca_bm25, busca_densa


def fusao_rrf(resultados_bm25, resultados_densa, k_rrf=60):

    ranking = {}

    # BM25
    for rank, documento in enumerate(resultados_bm25, start=1):

        if documento not in ranking:
            ranking[documento] = 0

        ranking[documento] += 1 / (k_rrf+rank)

    # DENSA
    for rank, documento in enumerate(resultados_densa, start=1):

        if documento not in ranking:
            ranking[documento] = 0

        ranking[documento] += 1 / (k_rrf+rank)

    resultados = sorted(
        ranking.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return resultados


def busca_hibrida(pergunta, documentos_filtrados, top_k=5):

    resultados_bm25 = busca_bm25(pergunta, documentos_filtrados)

    resultados_densa = busca_densa(pergunta, documentos_filtrados)

    documentos_bm25 = [
        doc for doc, score in resultados_bm25
    ]

    documentos_densa = [doc for doc, score in resultados_densa]

    ranking_final = fusao_rrf(
        documentos_bm25,
        documentos_densa
    )

    return ranking_final[:top_k]
