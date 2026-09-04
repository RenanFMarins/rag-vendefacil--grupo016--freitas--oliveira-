"""
    Calcula o nível de confiança da resposta ('Alta', 'Média', 'Baixa'),
    com base em dois sinais:
"""


def calcular_confidence_level(melhor_score, quantidade_fontes, k_rrf=60, rank_maximo_aceitavel=5):

    limiar_baixo = 1/(k_rrf + rank_maximo_aceitavel)
    limiar_medio = 1.3 * limiar_baixo
    limiar_alto = 2 / (k_rrf + 3)

    if melhor_score >= limiar_alto:
        nivel = "Alta"
    elif melhor_score >= limiar_medio:
        nivel = "Média"
    else:
        nivel = "Baixa"

    if quantidade_fontes <= 1 and nivel == "Alta":
        nivel = "Média"

    return nivel
