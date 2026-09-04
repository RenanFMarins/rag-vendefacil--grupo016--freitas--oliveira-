def fusao_rrf(resultados_bm25, resultados_densa, k_rrf=60):

    ranking = {}

    # BM25
    for rank, (documento, _) in enumerate(resultados_bm25, start=1):
        doc_id = documento.metadata.get("chunk_id")

        if doc_id not in ranking:
            ranking[doc_id] = {
                "documento": documento,
                "score": 0
            }

        ranking[doc_id]["score"] += 1 / (k_rrf + rank)

    # DENSA
    for rank, (documento, _) in enumerate(resultados_densa, start=1):

        doc_id = documento.metadata.get("chunk_id")

        if doc_id not in ranking:
            ranking[doc_id] = {
                "documento": documento,
                "score": 0
            }

        ranking[doc_id]["score"] += 1 / (k_rrf + rank)

    resultados = sorted(
        ranking.values(),
        key=lambda x: x['score'],
        reverse=True
    )
    print('SCORE', [
        (item["score"])
        for item in resultados
    ])

    return [
        (item["documento"], item["score"])
        for item in resultados
    ]
