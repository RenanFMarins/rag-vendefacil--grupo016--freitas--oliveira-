from .bm25 import busca_bm25
from .densa import busca_densa
from .rrf import fusao_rrf


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
