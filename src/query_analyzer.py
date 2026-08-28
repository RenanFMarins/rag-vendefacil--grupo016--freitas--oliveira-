from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np


def query_analyzer(pergunta):

    pergunta = pergunta.lower()

    filtros = {}

    tipos_documento = {"tickets": "ticket", "ticket": "ticket", "clientes": "customer", "cliente": "customer", "lojas": "store", "loja": "store",
                       "vendas": "sale", "venda": "sale", "produtos": "product", "produto": "product", "funcionários": "employee", "funcionários": "employee"}

    estados = {"minas gerais": "MG", "minas": "MG",
               "rio de janeiro": "RJ", "são paulo": "SP", "espírito santo": "ES"}

    modulos = {"pdv": "pdv", "ponto de venda": "pdv", "estoque": "estoque",
               "inventário": "estoque", "pay": "pay", "pagamento": "pay"}

    # documentos
    for termo, valor in tipos_documento.items():
        if termo in pergunta:
            filtros["doc_type"] = valor
            break

    # estado
    for termo, valor in estados.items():
        if termo in pergunta:
            filtros["state"] = valor
            break

    # modulo
    for termo, valor in modulos.items():
        if termo in pergunta:
            filtros["module"] = valor
            break
    print(filtros)

    return filtros


perguntas = [
    "Quais produtos a empresa VendeFácil oferece e quais são suas principais funcionalidades?",

    "Quais lojas estão localizadas em Minas Gerais e quais módulos elas possuem?",

    "Quais clientes estão no plano Enterprise e qual é o valor mensal?"
]

"""
for pergunta in perguntas:
    filtros = query_analyzer(pergunta)
    print(filtros)
"""


def filtrar_documentos(documentos, filtros):

    documentos_filtrados = []

    for documento in documentos:

        metadata = documento.metadata

        atende_filtros = True

        if "doc_type" in filtros:
            if metadata.get("doc_type") != filtros["doc_type"]:
                atende_filtros = False

        if "state" in filtros:
            if metadata.get('state') != filtros["state"]:
                atende_filtros = False

        if "module" in filtros:
            modulos_do_doc = metadata.get(
                'active_modules') or metadata.get('module') or []
            if isinstance(modulos_do_doc, list):

                termo_procurado = filtros['module'].lower()
                encontrou_modulo = any(termo_procurado in mod.lower()
                                       for mod in modulos_do_doc)
                if not encontrou_modulo:
                    atende_filtros = False
            else:

                if filtros['module'].lower() not in str(modulos_do_doc).lower():
                    atende_filtrros = False

        if atende_filtros:
            documentos_filtrados.append(documento)

    return documentos_filtrados


def similaridade_cosseno(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:

    v1 = np.array(vec1)
    v2 = np.array(vec2)

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2, axis=1)

    norm_v2[norm_v2 == 0] = 1.0
    if norm_v1 == 0:
        return np.zeros(len(v2))
    return np.dot(v2, v1) / (norm_v1 * norm_v2)


def busca_densa(pergunta, documentos_filtrados):

    model = HuggingFaceEmbeddings(
        model_name='paraphrase-multilingual-MiniLM-L12-v2')

    embedding_pergunta = model.embed_query(pergunta)

    textos = [documento.page_content for documento in documentos_filtrados]

    embeddings_documentos = model.embed_documents(textos)

    similaridades = similaridade_cosseno(
        embedding_pergunta, embeddings_documentos)

    resultados = list(zip(documentos_filtrados, similaridades))

    resultados.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return resultados


model = HuggingFaceEmbeddings(
    model_name='paraphrase-multilingual-MiniLM-L12-v2')


def abrir_banco():

    db = FAISS.load_local(
        'banco_faiss',
        model,
        allow_dangerous_deserialization=True)

    return db


db = abrir_banco()

pergunta = "Quais lojas de Minas Gerais possuem o módulo de estoque?"
filtros = query_analyzer(pergunta)

documentos = db.docstore._dict.values()

documentos_filtrados = filtrar_documentos(documentos, filtros)

resultados_dense = busca_densa(pergunta, documentos_filtrados)

for documento, score in resultados_dense[:5]:
    print(f"Score: {score:.4f}")
    print(f"Chunk: {documento.metadata.get('chunk_id')}")
    print(f"Texto: {documento.page_content}")
    print("-" * 60)
