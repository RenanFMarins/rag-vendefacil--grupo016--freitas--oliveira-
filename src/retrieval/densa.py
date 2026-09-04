from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np


def similaridade_cosseno(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:

    v1 = np.array(vec1)
    v2 = np.array(vec2)

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2, axis=1)

    norm_v2[norm_v2 == 0] = 1.0
    if norm_v1 == 0:
        return np.zeros(len(v2))
    return np.dot(v2, v1) / (norm_v1 * norm_v2)


def busca_densa(pergunta, documentos_filtrados, top_k=5):

    if not documentos_filtrados:
        return []

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

    return resultados[:top_k]
