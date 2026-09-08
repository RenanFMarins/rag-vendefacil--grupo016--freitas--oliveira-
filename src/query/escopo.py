FORA_DE_ESCOPO = "fora_de_escopo"
SEM_EVIDENCIA = "sem_evidencia"


def sinal_fraco_fora_de_escopo(analise):
    sem_assunto = analise['assunto'] is None
    sem_filtros = not analise['filtros']
    sem_entidades = not analise['entidades']

    return sem_assunto and sem_filtros and sem_entidades


def verificar_escopo(analise, resultadados_busca, k_rrf=60, rank_maximo=5):
    """
    Decisão real de escopo, chamada DEPOIS da busca híbrida (RRF)
    """
    sinal_fraco = sinal_fraco_fora_de_escopo(analise)

    limiar_score = 1 / (k_rrf + rank_maximo)

    if not resultadados_busca:
        motivo = FORA_DE_ESCOPO if sinal_fraco else SEM_EVIDENCIA
        return {
            'dentro_do_escopo':  False,
            "refusal_reason": motivo,
            "melhor_score": None,
            "limiar_usado": limiar_score
        }

    _, melhore_score = resultadados_busca[0]

    if melhore_score < limiar_score:
        motivo = FORA_DE_ESCOPO if sinal_fraco else SEM_EVIDENCIA
        return {
            'dentro_do_escopo':  False,
            "refusal_reason": motivo,
            "melhor_score": melhore_score,
            "limiar_usado": limiar_score,
        }

    return {
        'dentro_do_escopo':  True,
        "refusal_reason": None,
        "melhor_score": melhore_score,
        "limiar_usado": limiar_score
    }
