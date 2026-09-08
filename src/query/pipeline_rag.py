from src.query.query_analyzer import analisar_pergunta
from src.query.lgpd import verificar_politica_lgpd, aplicar_mascaramento_chunks
from src.query.escopo import verificar_escopo
from src.retrieval.filters import filtrar_documento
from src.retrieval.bm25 import busca_bm25
from src.retrieval.densa import busca_densa
from src.retrieval.rrf import fusao_rrf
from src.query.confidence_level import calcular_confidence_level
from src.query.schema import RAGResponse
from src.query.llm import gerar_resposta_llm


def pipeline_rag(pergunta, vocabulario, documentos):

    analise = analisar_pergunta(pergunta, vocabulario)
    politica_previa = verificar_politica_lgpd(analise)
    if politica_previa["comportamento"] == "recusar":
        return RAGResponse(
            answer="Não posso processar sua solicitação devido a restrições de privacidade.",
            confidence_level="Recusado",
            sources_used=[],
            reasoning="Recusado preventivamente por política de LGPD.",
            is_refusal=True,
            refusal_reason=politica_previa.get(
                "reason", "Violação de LGPD detectada na pergunta.")
        )

    documentos_filtrados = filtrar_documento(documentos, analise['filtros'])

    if not documentos_filtrados:
        return RAGResponse(
            answer="Não encontrei informações específicas com os filtros aplicados.",
            confidence_level="Baixo",
            sources_used=[],
            reasoning="Nenhum documento localizado para os filtros aplicados.",
            is_refusal=True,
            refusal_reason="Nenhum documento localizado para os filtros aplicados."
        )

    resultados_bm25 = busca_bm25(pergunta, documentos_filtrados)
    resultados_densos = busca_densa(pergunta, documentos_filtrados)
    resultados_fundidos = fusao_rrf(resultados_bm25, resultados_densos)

    escopo = verificar_escopo(analise, resultados_fundidos,  k_rrf=60)
    if not escopo["dentro_do_escopo"]:
        return RAGResponse(
            answer="Não posso responder a essa pergunta.",
            confidence_level="Recusado",
            sources_used=[],
            reasoning="Recusado por: " + escopo["refusal_reason"],
            is_refusal=True,
            refusal_reason=escopo["refusal_reason"],
        )

    top_k_pares = resultados_fundidos[:5]

    chunks_top_k = [doc for doc, score in top_k_pares]
    print("======", chunks_top_k)
    nivel_confianca = calcular_confidence_level(
        melhor_score=escopo["melhor_score"],
        quantidade_fontes=len(resultados_fundidos),
        k_rrf=60,
    )

    politica_final = verificar_politica_lgpd(analise, chunks_top_k)

    if politica_final["comportamento"] == "recusar":
        return RAGResponse(
            answer="Os documentos encontrados contêm informações confidenciais que não podem ser exibidas.",
            confidence_level="Recusado",
            sources_used=[],
            reasoning="Recusado reativamente por conter dados sensíveis expostos.",
            is_refusal=True,
            refusal_reason=politica_final.get(
                "reason", "Dados sensíveis detectados nos documentos recuperados.")
        )

    if politica_final["comportamento"] == "mascarar":
        chunks_top_k = aplicar_mascaramento_chunks(
            chunks_top_k, politica_final["atributos_detectados"])

    resposta = gerar_resposta_llm(
        pergunta,
        chunks_top_k,
        nivel_confianca
    )

    return resposta
