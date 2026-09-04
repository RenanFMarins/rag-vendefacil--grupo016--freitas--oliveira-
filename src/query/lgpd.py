import re
from src.query.query_analyzer import identificar_atributos

DADOS_RECUSAR = [
    "salario",
    "remuneracao",
    "cpf",
    "dados_bancarios",
    "chave_pix",
    "credencial",
    "token",
    "senha",
    "dados_saude"
]

DADOS_MASCARAR = [
    "email_pessoal",
    "email",
    "telefone",
    "endereco_residencial",
    "endereco",
    "numero_cartao",
    "salario",
    "remuneracao",
    "cpf",
    "dados_bancarios",
    "chave_pix",
    "credencial",
    "token",
    "senha",
    "dados_saude"

]


def verificar_sensibilidade_chunks(chunks):

    chunk_restrito = False
    atributos_no_conteudo = set()

    for chunk in chunks:
        metadata = getattr(chunk, "metadata", {}) or {}
        conteudo = getattr(chunk, "page_content", "") or ""

        if metadata.get("sensitivity") == "restrito":
            chunk_restrito = True

        atributos_no_conteudo.update(identificar_atributos(conteudo))

    return {
        "tem_chunk_restrito": chunk_restrito,
        "atributos_no_conteudo": list(atributos_no_conteudo),
    }


def verificar_politica_lgpd(analise, chunks=None):

    RECUSAR = "recusar"
    MASCARAR = "mascarar"
    RESPONDER = "responder"

    atributos = analise["atributos"]

    chunks = chunks or []

    atributos_pergunta = set(analise["atributos"])
    sesibilidade = verificar_sensibilidade_chunks(chunks)
    atributos_chunks = set(sesibilidade["atributos_no_conteudo"])

    atributos_totais = atributos_pergunta | atributos_chunks

    if sesibilidade["tem_chunk_restrito"] or any(
        atributo in DADOS_RECUSAR for atributo in atributos_totais
    ):
        return {

            "comportamento": RECUSAR,
            "is_refusal": True,
            "refusal_reason": "lgpd",
            "atributos_detectados": list(atributos_totais),
        }

    if any(atributo in DADOS_MASCARAR for atributo in atributos):
        return {
            "comportamento": MASCARAR,
            "is_refusal": False,
            "refusal_reason": None,
            "atributos_detectados": list(atributos_totais),
        }

    return {
        "comportamento": RESPONDER,
        "is_refusal": False,
        "refusal_reason": None,
        "atributos_detectados": list(atributos_totais),
    }


def mascarar_email(email):
    if "@" not in email:
        return email

    usuario, dominio = email.split("@")
    if len(usuario) > 2:
        email_mascarado = usuario[:2] + "***"
    else:
        email_mascarado = usuario[0] + "***" if usuario else "***"

    return f"{email_mascarado}@{dominio}"


def mascarar_telefone(telefone):

    digitos = re.sub(r'\D', '', telefone)

    if len(digitos) == 11:
        return f"({digitos[:2]}) {digitos[2]}****-**{digitos[-2:]}"

    elif len(digitos) == 10:
        return f"({digitos[:2]}) ****-**{digitos[-2:]}"

    return digitos


def mascarar_cartao(cartao):
    digitos = re.sub(r'\D', '', cartao)

    if len(digitos) >= 12:
        return f"**** **** **** {digitos[-4:]}"

    return cartao


def mascarar_endereco(endereco):
    if len(endereco) <= 10:
        return "Rua *********"
    return f"{endereco[:5]}********* n° ***"


def aplicar_mascaramento(atributo, valor):
    MASCARADORES = {
        "email": mascarar_email,
        "telefone": mascarar_telefone,
        "numero_cartao": mascarar_cartao,
        "endereco": mascarar_endereco
    }

    mascarador = MASCARADORES.get(atributo)

    if mascarador:
        return mascarador(valor)

    return valor
