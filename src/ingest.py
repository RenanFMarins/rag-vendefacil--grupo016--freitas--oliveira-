from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import glob
import os

from json_mod.json_utils import leitor_json
from csv_mod.csv_utils import leitor_csv

# pegar os caminhos dos arquivos
arquivos_recebidos = glob.glob(os.path.join(
    "../data/", "**/", "*"), recursive=True)
dados_arquivos = []

#
for doc in arquivos_recebidos:
    if os.path.isdir(doc):
        continue

    if "customers" in doc.lower():
        doc_type = "customer"
        sensitivity = "interno"
    elif "employees" in doc.lower():
        doc_type = "employee"
        sensitivity = "interno"
    elif "products" in doc.lower():
        doc_type = "product"
        sensitivity = "publico"
    elif "stores" in doc.lower():
        doc_type = "store"
        sensitivity = "publico"
    elif "tickets" in doc.lower():
        doc_type = "ticket"
        sensitivity = "interno"
    elif "logs" in doc.lower():
        doc_type = "log"
        sensitivity = "interno"
    elif "documentation" in doc.lower():
        doc_type = "manual"
        sensitivity = "publico"
    elif "policies" in doc.lower():
        doc_type = "policy"
        sensitivity = "interno"
    elif "meetings" in doc.lower():
        doc_type = "email"
        sensitivity = "interno"
    elif "emails" in doc.lower():
        doc_type = "email"
        sensitivity = "interno"
    elif "sales" in doc.lower():
        doc_type = "sale"
        sensitivity = "interno"
    else:
        doc_type = "outro"
        sensitivity = "outro"

    dados_arquivos.append(
        {"caminho": doc, "doc_type": doc_type, "sensitivity": sensitivity})


def criar_documento(texto, caminho, doc_type, sensitivity, chunk_id, **kwargs):
    metadata = {
        "source_file": caminho,
        "doc_type": doc_type,
        "chunk_id": chunk_id,
        "sensitivity": sensitivity,

    }
    metadata.update(kwargs)

    return Document(
        page_content=texto,
        metadata=metadata
    )


model = HuggingFaceEmbeddings(
    model_name='paraphrase-multilingual-MiniLM-L12-v2')


def salvar_banco_faiss():

    dados_json = leitor_json(dados_arquivos, criar_documento)
    dados_csv = leitor_csv(dados_arquivos, criar_documento)
    arquivos_banco = dados_json + dados_csv

    db = FAISS.from_documents(arquivos_banco, model)
    db.save_local('banco_faiss')


salvar_banco_faiss()


def abrir_banco():

    db = FAISS.load_local(
        'banco_faiss',
        model,
        allow_dangerous_deserialization=True)

    return db


query = "quais os nomes dos colaboradores do departamento de Suporte Técnic?"
db = abrir_banco()
docs = db.similarity_search(query)

print(docs[0].page_content)
