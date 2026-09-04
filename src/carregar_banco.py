from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path

model = HuggingFaceEmbeddings(
    model_name='paraphrase-multilingual-MiniLM-L12-v2')


# Garanta que o FAISS e o model estão importados corretamente neste arquivo

def abrir_banco():
    # Pega o caminho de 'src/carregar_banco.py', resolve o absoluto e pega a pasta pai (src)
    diretorio_src = Path(__file__).resolve().parent

    # Monta o caminho apontando direto para src/banco_faiss
    caminho_banco = diretorio_src / "banco_faiss"

    db = FAISS.load_local(
        # Converte o objeto Path para a string que o FAISS precisa
        str(caminho_banco),
        model,
        allow_dangerous_deserialization=True
    )

    return db
