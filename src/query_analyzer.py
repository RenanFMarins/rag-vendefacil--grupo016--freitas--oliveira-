from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever


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


db = abrir_banco()

# problema com nomes pequenos Minas gerais com mg, rever a lógica
pergunta = "Quais tickets de clientes de Minas Gerais estão relacionados ao módulo de estoque?"

documentos = list(db.docstore._dict.values())


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

filtros = {
    "doc_type": "store",
    "state": "MG",
    "module": "estoque"
}

documentos_filtrados = filtrar_documento(
    documentos, filtros)


resultados, scores = busca_bm25(pergunta, documentos_filtrados)

print("=" * 80)
print(f"🔍 BUSCA: '{pergunta}'")
print(f"⚙️ FILTROS: {filtros}")
print("-" * 80)
print(f"{'Rank':<6} | {'Score':<10} | {'Conteúdo do Documento'}")
print("-" * 80)

for i in range(len(resultados[0])):
    texto = resultados[0][i]
    score = scores[0][i]
    texto_resumido = texto if len(texto) <= 60 else texto[:57] + "..."

    print(f"{i+1:<6} | {score:<10.4f} | {texto_resumido}")

print("=" * 80)

"""   
resultados_dense = busca_densa(pergunta, documentos_filtrados)

for documento, score in resultados_dense[:5]:
    print(f"Score: {score:.4f}")
    print(f"Chunk: {documento.metadata.get('chunk_id')}")
    print(f"Texto: {documento.page_content}")
    print("-" * 60)
"""
