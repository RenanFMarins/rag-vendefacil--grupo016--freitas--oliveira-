from src.query.query_analyzer import analisar_pergunta
from src.query.lgpd import verificar_politica_lgpd
from src.query.escopo import verificar_escopo
from src.retrieval.filters import filtrar_documento
from src.retrieval.bm25 import busca_bm25
from src.retrieval.densa import busca_densa
from src.retrieval.rrf import fusao_rrf


def pipeline_rag(pergunta, vocabulario, documentos):

    analise = analisar_pergunta(pergunta, vocabulario)
    print("pipepline: ", analise["filtros"])

    politica_previa = verificar_politica_lgpd(analise)
    if politica_previa["comportamento"] == "recusar":
        return politica_previa

    documentos_filtrados = filtrar_documento(documentos, analise['filtros'])
    print(
        f"\n🔍 [DEBUG] Filtros enviados: {analise['filtros']} | Total após filtro: {len(documentos_filtrados)}")

    if not documentos_filtrados:
        return {
            "is_refusal": True,
            "refusal_reason": "Nenhum documento localizado para os filtros aplicados."
        }

    resultados_bm25 = busca_bm25(pergunta, documentos_filtrados)
    resultados_densos = busca_densa(pergunta, documentos_filtrados)

    resultados_fundidos = fusao_rrf(resultados_bm25, resultados_densos)
    print("MELHOR RESULTADO RRF:", resultados_fundidos[0])

    escopo = verificar_escopo(analise, resultados_fundidos)
    if not escopo["dentro_do_escopo"]:
        print(
            f"🛑 [DEBUG BLOC] BLOQUEADO NO ESCOPO para a pergunta: '{pergunta}' | Motivo: {escopo['refusal_reason']}")
        return {"is_refusal": True, "refusal_reason": escopo["refusal_reason"]}

    chunks_top_k = [doc for doc, score in resultados_fundidos[:5]]
    politica_final = verificar_politica_lgpd(analise, chunks_top_k)
    if politica_final["comportamento"] == "recusar":
        print(
            f"🛑 [DEBUG BLOC] BLOQUEADO NA LGPD FINAL para a pergunta: '{pergunta}'")
        return politica_final

    return {
        "pergunta": pergunta,
        "top_chunks": chunks_top_k,
        "politica_final": politica_final,
        "analise": analise
    }
