import bm25s


def busca_bm25(pergunta, documentos_filtrados, top_k=5):

    textos = [documento.page_content for documento in documentos_filtrados]

    mapa_documentos = {
        texto: documento
        for texto, documento in zip(
            textos,
            documentos_filtrados
        )
    }

    retriever = bm25s.BM25()

    corpus_tokens = bm25s.tokenize(textos, stopwords="portuguese")

    retriever.index(corpus_tokens)

    pergunta_tokens = bm25s.tokenize(pergunta, stopwords="portuguese")

    total_documentos = len(textos)
    # k=5
    resultados, scores = retriever.retrieve(
        pergunta_tokens, k=min(top_k, len(textos)))

    print("RESULTADOS:", resultados)
    print("TIPO RESULTADOS:", type(resultados))

    print("PRIMEIRO:", resultados[0])
    print("TIPO PRIMEIRO:", type(resultados[0]))

    resultados_bm25 = []

    for indice, score in zip(resultados[0], scores[0]):

        documento = documentos_filtrados[int(indice)]

        resultados_bm25.append((documento, score))

    return resultados_bm25
