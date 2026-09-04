from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# No arquivo src/pipeline.py, mude a linha 3 para:
from src.retrieval.filters import combinar_filtros, extract_metadata, extrair_filtros_automatico, extrair_filtros_manuais, filtrar_documento
from src.retrieval.hybrid import busca_hibrida

model = HuggingFaceEmbeddings(
    model_name='paraphrase-multilingual-MiniLM-L12-v2')


def abrir_banco():

    db = FAISS.load_local(
        'banco_faiss',
        model,
        allow_dangerous_deserialization=True)

    return db


db = abrir_banco()

# problema com nomes pequenos Minas gerais com mg, rever a lógica
pergunta = "Quais produtos a empresa VendeFácil oferece e quais são suas principais funcionalidades?"

documentos = list(db.docstore._dict.values())

vocabulario = extract_metadata(documentos)

filtros_manuais = extrair_filtros_manuais(pergunta)

filtros_automaticos = extrair_filtros_automatico(
    pergunta,
    vocabulario
)

filtros_finais = combinar_filtros(
    filtros_manuais,
    filtros_automaticos
)

documentos_filtrados = filtrar_documento(
    documentos,
    filtros_finais
)

resultados = busca_hibrida(
    pergunta,
    documentos_filtrados,
    top_k=5
)

for posicao, (documento, score) in enumerate(resultados, start=1):

    print(f"\n========== RESULTADO {posicao} ==========")
    print(f"Score RRF: {score:.6f}")
    print("Metadata:", documento.metadata)
    print("Conteúdo:")
    print(documento.page_content)
