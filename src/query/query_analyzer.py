from src.retrieval.filters import extrair_filtros_automatico, extrair_filtros_manuais
from src.query.doc_type_via_llm import classificar_doc_type_via_llm
from rapidfuzz import fuzz
import unicodedata
import re
from groq import Groq
import os
import getpass


if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for GROQ: ")

cliente = Groq()


PALAVRAS_AGREGADO = [
    "media",
    "medio",
    "total",
    "quantos",
    "quantas",
    "soma",
    "media geral",
    "no geral",
    "em geral",
    "todos os",
    "todas as",
    "por equipe",
    "por setor",
    "por departamento",
]

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
    "numero_cartao"
]


def normalizar_texto(texto):

    texto = texto.lower()

    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(
        caractere
        for caractere in texto
        if unicodedata.category(caractere) != "Mn"
    )
    return texto


def identificar_assunto(pergunta, corte_similaridade=80):

    SINONIMOS = {
        "funcionário": [
            "funcionário",
            "funcionario",
            "colaborador",
            "empregado"
        ],

        "produto": [
            "produto",
            "item",
            "mercadoria"
        ],

        "loja": [
            "loja",
            "filial",
            "estabelecimento"
        ],

        "cliente": [
            "cliente",
            "consumidor"
        ],

        "venda": [
            "venda",
            "vendas",
            "pedido",
            "faturamento"
        ],

        "log": [
            "log",
            "logs",
            "servidor",
            "erro do sistema"
        ],

        "ticket": [
            "ticket",
            "chamado",
            "suporte"
        ],

        "documentacao": [
            "manual",
            "documentação",
            "documentacao",
            "módulo",
            "modulo",
            "como funciona"
        ],

        "reuniao": [
            "reunião",
            "reuniao",
            "ata",
            "atas"
        ],

        "email": [
            "email",
            "e-mail",
            "correio eletronico"
        ],

        "politica": [
            "política",
            "politica",
            "norma",
            "regra interna"
        ],
    }

    pergunta = pergunta.lower()

    for assunto, palavras in SINONIMOS.items():
        if any(palavra in pergunta for palavra in palavras):
            return assunto

    for assunto, palavras in SINONIMOS.items():
        for palavra in palavras:
            if fuzz.partial_ratio(palavra, pergunta) >= corte_similaridade:
                return assunto

    return None


def identificar_intencao(pergunta):
    INTENCOES = {
        "consulta": [
            "qual",
            "quais",
            "quem",
            "onde",
            "quando",
            "quanto",
            "informar",
            "mostrar",
            "consultar"
        ],

        "listar": [
            "listar",
            "liste",
            "quais são",
            "todos",
            "todas"
        ],

        "buscar": [
            "buscar",
            "procure",
            "procurar",
            "encontrar",
            "encontre"
        ]
    }

    pergunta = normalizar_texto(pergunta)

    for intencao, palavras in INTENCOES.items():
        if any(palavra in pergunta for palavra in palavras):
            return intencao

    return None


def identificar_atributos(pergunta):

    ATRIBUTOS = {
        "salario": [
            "salari",       # cobre salario/salário/salarial/salariais
            "remuneraca",   # cobre remuneração/remunerações
            "vencimento",
            "holerite",
            "folha de pagamento"
        ],

        "cpf": [
            "cpf",
        ],

        "dados_bancarios": [
            "conta bancaria",
            "conta corrente",
            "agencia bancaria",
            "dados bancarios",
            "banco",
            "iban"
        ],

        "chave_pix": [
            "chave pix"
        ],

        "credencial": [
            "senha",
            "credencial",
            "token",
            "chave de api",
            "api key",
            "login e senha"
        ],

        "dados_saude": [
            "saude",
            "atestado",
            "exame medico",
            "doenca",
            "cid ",
            "laudo medico"
        ],

        "telefone": [
            "telefone",
            "celular",
            "contato"
        ],

        "endereco": [
            "endereco",
            "residencia",
            "logradouro"
        ],

        "email": [
            "email",
            "e-mail",
            "correio eletronico"
        ],

        "numero_cartao": [
            "numero do cartao",
            "cartao de credito",
            "cartao de debito"
        ]
    }

    pergunta = normalizar_texto(pergunta)

    atributos_encontrados = []

    for atributo, palavras in ATRIBUTOS.items():
        if any(palavra in pergunta for palavra in palavras):
            atributos_encontrados.append(atributo)

    return atributos_encontrados


def contem_nome(pergunta):

    padrao_nome = r'\b[A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+)+\b'
    return re.search(padrao_nome, pergunta) is not None


def contem_identificador_especifico(pergunta):

    padroes = [
        r'\bcliente\s+n?[ºo°]?\s*\d+\b',
        r'\bticket\s*#?\s*\d+\b',
        r'\bchamado\s*#?\s*\d+\b',
        r'\bfuncion[aá]rio\s+n?[ºo°]?\s*\d+\b',
        r'\bcpf\s*[\d.\-]{5,}\b',
    ]
    return any(re.search(p, pergunta, re.IGNORECASE) for p in padroes)


def identificar_dado_individual(pergunta, analise):

    pergunta_normalizada = normalizar_texto(pergunta)

    if any(palavra in pergunta_normalizada for palavra in PALAVRAS_AGREGADO):
        return False

    if contem_nome(pergunta):
        return True

    if contem_identificador_especifico(pergunta):
        return True

    return False


def identificar_dado_sensivel(atributos):
    DADOS_SENSIVEIS = [
        "salario",
        "cpf",
        "telefone",
        "endereco_residencial",
        "endereco",
        "email_pessoal",
        "email",
        "numero_cartao",
        "dados_bancarios",
        "chave_pix",
        "credencial",
        "token",
        "senha",
        "dados_saude"
    ]
    return any(
        atributo in DADOS_SENSIVEIS
        for atributo in atributos
    )


"""
perguntas = [
    "Qual o salário do funcionario João Pereira?",
    "Quais produtos a VendeFácil oferece?",
    "Liste todos os funcionários do financeiro.",
    "Quero buscar os produtos ativos.",
    "Onde fica a loja de Juiz de Fora?"
]

for pergunta in perguntas:
    print(analisar_pergunta(pergunta))
    print()

"""

"""
a política de LGPD deveria proteger o conteúdo dos chunks recuperados, 
não só interpretar a intenção da pergunta.
"""


"""
perguntas = [
    "Qual o salário do funcionario João Pereira?",
    "Quais produtos a VendeFácil oferece?",
    "Liste todos os funcionários do financeiro.",
    "Quero buscar os produtos ativos.",
    "Onde fica a loja de Juiz de Fora?"
]


for pergunta in perguntas:
    analise = analisar_pergunta(pergunta)
    politicas = verificar_politica_lgpd(analise)
    print(politicas)
    print()

"""


"""
email_teste = "mariasilva@gmail.com"
telefone_teste = "(31) 98765-4321"
cartao_teste = "4532 1182 9381 1234"
endereco_teste = "Av. Paulista, 1000 - Bela Vista"

print("E-mail:", mascarar_email(email_teste))
print("Telefone:", mascarar_telefone(telefone_teste))
print("Cartão de Crédito:", mascarar_cartao(cartao_teste))
print("Endereço:", mascarar_endereco(endereco_teste))

"""


def identificar_filtros(pergunta, vocabulario):

    doc_types_identificados = classificar_doc_type_via_llm(
        pergunta, cliente)

    filtros_manuais = extrair_filtros_manuais(pergunta)

    filtros_automaticos = extrair_filtros_automatico(
        pergunta,
        vocabulario, doc_types_permitidos=doc_types_identificados
    )

    filtros_combinados = filtros_automaticos | filtros_manuais
    filtros_combinados["doc_type"] = doc_types_identificados

    return filtros_combinados


def extrair_entidades(pergunta):
    entidades = []

    nome = re.search(
        r'\b[A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+)+\b',
        pergunta
    )
    if nome:
        entidades.append({"tipo": "pessoa", "valor": nome.group()})

    padroes_id = {
        "cliente": r'\bcliente\s+n?[ºo°]?\s*\d+\b',
        "ticket": r'\bticket\s*#?\s*\d+\b',
        "chamado": r'\bchamado\s*#?\s*\d+\b',
        "funcionario": r'\bfuncion[aá]rio\s+n?[ºo°]?\s*\d+\b',
        "cpf": r'\bcpf\s*[\d.\-]{5,}\b',
    }
    for tipo, padrao in padroes_id.items():
        id = re.search(padrao, pergunta, re.IGNORECASE)
        if id:
            entidades.append({"tipo": tipo, "valor": id.group()})

    return entidades


def analisar_pergunta(pergunta, vocabulario):
    analise = {
        "assunto": identificar_assunto(pergunta),
        "intencao": identificar_intencao(pergunta),
        "atributos": identificar_atributos(pergunta),
        # depois fazer a parte de entidades e filtros
        "entidades": extrair_entidades(pergunta),
        "filtros": identificar_filtros(pergunta, vocabulario),
        "dado_individual": False,
        "dado_sensivel": False
    }

    analise["dado_individual"] = identificar_dado_individual(pergunta, analise)

    analise["dado_sensivel"] = any(
        atributo in DADOS_RECUSAR + DADOS_MASCARAR
        for atributo in analise["atributos"]
    )
    print("analise", analise["filtros"])
    return analise


"""
def processar_pergunta(pergunta, vocabulario):

    analise = analisar_pergunta(pergunta, vocabulario)

    politica = verificar_politica_lgpd(analise)

    return {
        "pergunta": pergunta,
        "analise": analise,
        "politica": politica

    }

"""
"""
perguntas = [
    "Qual o salário do funcionario João Pereira?",
    "Qual a média salarial da equipe de suporte?",
    "Quais produtos a VendeFácil oferece?",
    "Liste todos os funcionários do financeiro.",
    "Quero buscar os produtos ativos.",
    "Onde fica a loja de Juiz de Fora?"
]

for pergunta in perguntas:
    resultado = analisar_pergunta(pergunta, vocabulario={})
    print("Pergunta:", pergunta)
    print("Análise:", resultado["filtros"])

    print()



pergunta = "Qual o salário do funcionário João Pereira?"

resultado = processar_pergunta(pergunta)

print("Pergunta:", resultado["pergunta"])
print("Análise:", resultado["analise"])
print("Política:", resultado["politica"])
"""
