"""Classifica perguntas por regras determinísticas da política LGPD."""

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict


LGPDAction = Literal["RECUSAR", "MASCARAR", "RESPONDER"]
LGPDCategory = Literal[
    "salario_individual",
    "remuneracao_individual",
    "risco_reidentificacao",
    "remuneracao_agregada",
    "cpf",
    "dados_bancarios",
    "chave_pix",
    "credenciais",
    "tokens",
    "senhas_em_logs",
    "dados_saude",
    "email_pessoal",
    "telefone",
    "endereco_residencial",
    "numero_cartao",
    "customer_id",
    "dados_produto",
    "dados_loja",
    "politica",
    "manual",
    "sem_dado_sensivel",
]

MINIMUM_AGGREGATE_GROUP_SIZE = 5


class LGPDDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: LGPDAction
    category: LGPDCategory
    reason: str


SALARY_PATTERN = re.compile(
    r"\b(salario|salarial|remuneracao|contracheque|holerite)\b"
)
SALARY_POLICY_PATTERN = re.compile(
    r"\b(politica|manual|regra|diretriz)(?:\s+\w+){0,3}\s+"
    r"(?:salarial|salario|remuneracao)\b"
)
AGGREGATE_PATTERN = re.compile(
    r"\b(media|mediana|estatistica|agregado|agregada|total|distribuicao|"
    r"faixa salarial|percentual)\b"
)

REFUSAL_PATTERNS: tuple[tuple[LGPDCategory, re.Pattern[str], str], ...] = (
    ("cpf", re.compile(r"\bcpf\b"), "CPF é um identificador pessoal protegido."),
    (
        "dados_bancarios",
        re.compile(
            r"\b(dados bancarios|conta bancaria|numero da conta|"
            r"agencia bancaria|extrato bancario)\b"
        ),
        "Dados bancários individuais não podem ser divulgados.",
    ),
    (
        "chave_pix",
        re.compile(r"\bchave pix\b"),
        "Chaves PIX não podem ser divulgadas.",
    ),
    (
        "credenciais",
        re.compile(r"\b(credencial|credenciais|login|acesso ssh|chave api)\b"),
        "Credenciais de acesso não podem ser divulgadas.",
    ),
    (
        "tokens",
        re.compile(r"\b(token|tokens|jwt|bearer token|secret)\b"),
        "Tokens e segredos de autenticação não podem ser divulgados.",
    ),
    (
        "senhas_em_logs",
        re.compile(r"\b(senha|senhas|password)\b"),
        "Senhas, inclusive as registradas em logs, não podem ser divulgadas.",
    ),
    (
        "dados_saude",
        re.compile(
            r"\b(dados de saude|diagnostico|doenca|atestado medico|"
            r"prontuario|condicao medica)\b"
        ),
        "Dados individuais de saúde são dados pessoais sensíveis.",
    ),
)

MASKING_PATTERNS: tuple[tuple[LGPDCategory, re.Pattern[str], str], ...] = (
    (
        "email_pessoal",
        re.compile(r"\b(e-mail pessoal|email pessoal|endereco de e-mail pessoal)\b"),
        "E-mail pessoal exige mascaramento antes da exibição.",
    ),
    (
        "telefone",
        re.compile(r"\b(telefone|numero de celular|celular de)\b"),
        "Telefone exige mascaramento antes da exibição.",
    ),
    (
        "endereco_residencial",
        re.compile(r"\b(endereco residencial|endereco da casa|onde mora)\b"),
        "Endereço residencial exige mascaramento antes da exibição.",
    ),
    (
        "numero_cartao",
        re.compile(r"\b(numero do cartao|dados do cartao|cartao de credito)\b"),
        "Número de cartão exige mascaramento antes da exibição.",
    ),
)

ALLOWED_PATTERNS: tuple[tuple[LGPDCategory, re.Pattern[str], str], ...] = (
    (
        "customer_id",
        re.compile(r"\b(customer_id|cust\d+)\b"),
        "customer_id é uma referência interna permitida.",
    ),
    (
        "dados_produto",
        re.compile(r"\b(produto|produtos|sku|catalogo)\b"),
        "Dados de produto são normalmente permitidos.",
    ),
    (
        "dados_loja",
        re.compile(r"\b(loja|lojas|filial|filiais)\b"),
        "Dados de loja são normalmente permitidos.",
    ),
    (
        "politica",
        re.compile(r"\b(politica|politicas|diretriz|diretrizes)\b"),
        "Consulta sobre política é normalmente permitida.",
    ),
    (
        "manual",
        re.compile(r"\b(manual|procedimento|documentacao)\b"),
        "Consulta sobre manual ou procedimento é normalmente permitida.",
    ),
)


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return " ".join(without_accents.split())


def _decision(
    action: LGPDAction,
    category: LGPDCategory,
    reason: str,
) -> LGPDDecision:
    return LGPDDecision(action=action, category=category, reason=reason)


def classify_lgpd_question(
    question: str,
    *,
    aggregate_group_size: int | None = None,
) -> LGPDDecision:
    """Classifica a pergunta sem executar LLM, mascaramento ou retrieval.

    Remuneração agregada só é permitida quando o grupo conhecido possui
    ao menos cinco pessoas. Sem essa informação, a política recusa por risco
    de reidentificação de indivíduos em grupos pequenos.
    """
    normalized_question = _normalize(question)
    if not normalized_question:
        raise ValueError("A pergunta não pode estar vazia.")
    if aggregate_group_size is not None and aggregate_group_size < 1:
        raise ValueError("aggregate_group_size deve ser maior que zero.")

    if SALARY_PATTERN.search(normalized_question):
        if SALARY_POLICY_PATTERN.search(normalized_question):
            return _decision(
                "RESPONDER",
                "politica",
                "Consulta sobre política de remuneração, sem pedido de valor individual.",
            )
        if AGGREGATE_PATTERN.search(normalized_question):
            if (
                aggregate_group_size is not None
                and aggregate_group_size >= MINIMUM_AGGREGATE_GROUP_SIZE
            ):
                return _decision(
                    "RESPONDER",
                    "remuneracao_agregada",
                    "Estatística agregada permitida para grupo com pelo menos cinco pessoas.",
                )
            return _decision(
                "RECUSAR",
                "risco_reidentificacao",
                "Agregado salarial exige grupo conhecido com pelo menos cinco pessoas.",
            )

        category: LGPDCategory = (
            "remuneracao_individual"
            if "remuneracao" in normalized_question
            else "salario_individual"
        )
        return _decision(
            "RECUSAR",
            category,
            "Salário ou remuneração individual não pode ser divulgado.",
        )

    for category, pattern, reason in REFUSAL_PATTERNS:
        if pattern.search(normalized_question):
            return _decision("RECUSAR", category, reason)

    for category, pattern, reason in MASKING_PATTERNS:
        if pattern.search(normalized_question):
            return _decision("MASCARAR", category, reason)

    for category, pattern, reason in ALLOWED_PATTERNS:
        if pattern.search(normalized_question):
            return _decision("RESPONDER", category, reason)

    return _decision(
        "RESPONDER",
        "sem_dado_sensivel",
        "A pergunta não solicita categoria sensível reconhecida pela política.",
    )

