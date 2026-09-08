import re
from src.query.query_analyzer import identificar_atributos

# remover esse atributos, passar para mascarar
DADOS_RECUSAR_DIRETO = [
    # Citados explicitamente na seção 1.4 (guardrail para sistemas de IA/RAG)
    "salario",
    "remuneracao",
    "senha",
    "cpf",
    # Tratados como incidente de segurança na seção 2 (vazamento de
    # credenciais/chaves de API aciona procedimento de emergência)
    "credencial",
    "token",
    "chave_pix",
    "chave_api",
    # "Nível 1 (Restrito ao RH e Diretoria)" na seção 1.3, mesmo grupo de
    # confidencialidade do salário/CPF -- não citados nominalmente na 1.4,
    # mas tratados aqui com o mesmo rigor por consistência de nível.
    # Se isso for rigor demais pro seu caso, mova para DADOS_MASCARAR.
    "bonus",
    "endereco_residencial",
    "dados_bancarios",
]

DADOS_MASCARAR = [
    # Não mencionados na política -- boa prática geral de PII
    "email_pessoal",
    "email",
    "telefone",
    "endereco",
    "numero_cartao",
    # Proibidos de sequer existir no banco (seção 1.2) -- tratamento
    # defensivo caso apareçam "de carona" em algum documento
    "dados_saude",
    "biometria",
    "religiao",
    "cvv",
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

    chunks = chunks or []

    atributos_pergunta = set(analise["atributos"])
    sesibilidade = verificar_sensibilidade_chunks(chunks)
    atributos_chunks = set(sesibilidade["atributos_no_conteudo"])
    atributos_totais = atributos_pergunta | atributos_chunks

    pediu_credencial_diretamente = bool(
        atributos_pergunta & set(DADOS_RECUSAR_DIRETO))

    if sesibilidade["tem_chunk_restrito"] or pediu_credencial_diretamente:
        return {

            "comportamento": RECUSAR,
            "is_refusal": True,
            "refusal_reason": "lgpd",
            "atributos_detectados": list(atributos_totais),
        }

    todos_atributos_sensiveis = set(DADOS_RECUSAR_DIRETO) | set(DADOS_MASCARAR)
    if atributos_totais & todos_atributos_sensiveis:
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
    return "[TELEFONE PROTEGIDO]"


def mascarar_cartao(cartao):
    digitos = re.sub(r'\D', '', cartao)
    if len(digitos) >= 12:
        return f"**** **** **** {digitos[-4:]}"
    return "[CARTÃO PROTEGIDO]"


def mascarar_endereco(endereco):
    if len(endereco) <= 10:
        return "Rua *********"
    return f"{endereco[:5]}********* n° ***"


def mascarar_cpf(cpf):
    digitos = re.sub(r'\D', '', cpf)
    if len(digitos) == 11:
        return f"***.***.{digitos[6:9]}-**"
    return "[CPF PROTEGIDO]"


def mascarar_generico(valor):
    return "[DADO PROTEGIDO]"


MASCARADORES = {
    "email": mascarar_email,
    "email_pessoal": mascarar_email,
    "telefone": mascarar_telefone,
    "numero_cartao": mascarar_cartao,
    "endereco": mascarar_endereco,
    "endereco_residencial": mascarar_endereco,
    "cpf": mascarar_cpf,
}


def aplicar_mascaramento(atributo, valor):
    mascarador = MASCARADORES.get(atributo, mascarar_generico)
    return mascarador(valor)


PADROES_CONTEUDO = {
    "salario": re.compile(r'R\$\s?[\d\.,]+'),
    "remuneracao": re.compile(r'R\$\s?[\d\.,]+'),
    "cpf": re.compile(r'\d{3}\.?\d{3}\.?\d{3}-?\d{2}'),
    "email": re.compile(r'[\w\.-]+@[\w\.-]+\.\w+'),
    "telefone": re.compile(r'\(?\d{2}\)?\s?\d{4,5}-?\d{4}'),
    "numero_cartao": re.compile(r'(?:\d[ -]*?){13,19}'),
}


def mascarar_texto(atributo, texto):

    padrao = PADROES_CONTEUDO.get(atributo)
    if not padrao:
        return texto

    def substituir(match):
        return aplicar_mascaramento(atributo, match.group())

    return padrao.sub(substituir, texto)


def aplicar_mascaramento_chunks(chunks, atributos_a_mascarar):

    atributos_a_mascarar = set(atributos_a_mascarar) & set(DADOS_MASCARAR)
    if not atributos_a_mascarar:
        return chunks

    chunks_mascarados = []
    for chunk in chunks:
        metadata_nova = dict(getattr(chunk, "metadata", {}) or {})
        conteudo_novo = getattr(chunk, "page_content", "") or ""

        for atributo in atributos_a_mascarar:

            if atributo in metadata_nova and metadata_nova[atributo]:
                metadata_nova[atributo] = aplicar_mascaramento(
                    atributo, str(metadata_nova[atributo]))

            conteudo_novo = mascarar_texto(atributo, conteudo_novo)

        chunk_mascarado = type(chunk)(page_content=conteudo_novo, metadata=metadata_nova) \
            if hasattr(chunk, "page_content") else chunk

        chunks_mascarados.append(chunk_mascarado)

    return chunks_mascarados
