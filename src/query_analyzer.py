from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
import numpy as np
from rapidfuzz import fuzz, process

model = HuggingFaceEmbeddings(
    model_name='paraphrase-multilingual-MiniLM-L12-v2')


def abrir_banco():

    db = FAISS.load_local(
        'banco_faiss',
        model,
        allow_dangerous_deserialization=True)

    return db


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
# rever campos que vão entrar no extract


def extrair_filtros_manuais(pergunta):

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


def extract_metadata(documentos):

    CAMPOS_NAO_FILTRAVEIS = {
        "chunk_id",
        "store_id",
        "customer_id",
        "product_id",
        "error_code"
    }

    filtros = {}

    for documento in documentos:

        for campo, valor in documento.metadata.items():

            if campo in CAMPOS_NAO_FILTRAVEIS:
                continue

            if campo not in filtros:
                filtros[campo] = set()

            if isinstance(valor, list):
                filtros[campo].update(str(item).lower() for item in valor)
            else:
                filtros[campo].add(str(valor).lower())

    return filtros


def extrair_filtros_automatico(pergunta, vocabulario, limiar=75):

    pergunta = pergunta.lower()
    filtros = {}

    for campo, valores in vocabulario.items():

        if campo == "state":
            continue

        if not valores:
            continue

        melhor_resultado = None
        melhor_score = 0

        for valor in valores:

            valor = str(valor).lower().strip()

            if len(valor) < 3:
                continue

            if valor in pergunta.split():
                score = 100

            else:
                score = fuzz.token_set_ratio(
                    valor,
                    pergunta
                )

            if score > melhor_score:
                melhor_score = score
                melhor_resultado = valor

        if melhor_score >= limiar:
            filtros[campo] = melhor_resultado

    return filtros


def combinar_filtros(filtros_manuais, filtros_automaticos):
    filtros_combinados = {}
    # metadatas = extract_metadata()

    filtros_combinados = filtros_automaticos | filtros_manuais

    return filtros_combinados


def filtrar_documento(documentos, filtros):

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
                    atende_filtros = False

        if atende_filtros:
            documentos_filtrados.append(documento)

    return documentos_filtrados


def processar_filtros(pergunta, documentos):

    vocabulario = extract_metadata(documentos)

    filtros_manuais = extrair_filtros_manuais(pergunta)

    filtros_automaticos = extrair_filtros_automatico(pergunta, vocabulario)

    filtro_combinado = combinar_filtros(filtros_manuais, filtros_automaticos)

    documentos_filtrados = filtrar_documento(documentos, filtro_combinado)

    return {
        "filtros_manuais": filtros_manuais,
        "filtros_automaticos": filtros_automaticos,
        "filtro_combinado": filtro_combinado,
        "documentos_filtrados": documentos_filtrados
    }


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


db = abrir_banco()

# problema com nomes pequenos Minas gerais com mg, rever a lógica
pergunta = "Quais tickets de clientes de Minas Gerais estão relacionados ao módulo de estoque?"

documentos = list(db.docstore._dict.values())

bm25_retriever = BM25Retriever.from_documents(documentos)
bm25_retriever.k = 3

faiss_retriever = db.as_retriever(search_kwargs={"k": 3})

resultados_faiss = faiss_retriever.invoke(pergunta)
resultados_bm25 = bm25_retriever.invoke(pergunta)

"""
documentos_finais = list({doc.page_content: doc for doc in (
    resultados_faiss + resultados_bm25)}.values())

print("\n" + "="*50)
print(f"📄 DOCUMENTOS RECUPERADOS (Total: {len(documentos_finais)})")
print("="*50 + "\n")

for i, doc in enumerate(documentos_finais, start=1):
    print(f"🔹 [Documento {i}]")
    print(f"📝 Conteúdo: {doc.page_content}")
    print(f"🗂️ Metadados: {doc.metadata}")
    print("-" * 50)


resultado_filtros = processar_filtros(pergunta, documentos)

print("\nFILTROS MANUAIS:")
print(resultado_filtros["filtros_manuais"])


print("\nFILTROS AUTOMÁTICOS:")
print(resultado_filtros["filtros_automaticos"])


print("\nFILTROS FINAIS:")
print(resultado_filtros["filtro_combinado"])


print("\nDOCUMENTOS ENCONTRADOS:")
print(len(resultado_filtros["documentos_filtrados"]))
"""
"""
for campo, valores in filtros.items():
    print(f"\n{campo}:")
    print(valores)

resultado = process.extract(
    frases,
    metadados,
    scorer=fuzz.partial_token_set_ratio,
    score_cutoff=75
)

for palavra_correta, score, index in resultado:
    print(f"Metadado encontrado: '{palavra_correta}' (Score: {score:.1f}%)")

"""
"""
documentos_filtrados = filtrar_documentos(documentos, filtros)

resultados_dense = busca_densa(pergunta, documentos_filtrados)

for documento, score in resultados_dense[:5]:
    print(f"Score: {score:.4f}")
    print(f"Chunk: {documento.metadata.get('chunk_id')}")
    print(f"Texto: {documento.page_content}")
    print("-" * 60)
"""
