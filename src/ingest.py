import os
import sys
# fmt: off
# isort: skip

RAIZ_PROJETO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if RAIZ_PROJETO not in sys.path:
    sys.path.append(RAIZ_PROJETO)

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import glob
import os

from src.json_mod.json_utils import leitor_json
from src.csv_mod.csv_utils import leitor_csv
from src.jsonl_mod.jsonl_utils import leitor_jsonl
from src.md_mod.md_utils import leitor_markdown 
from src.txt_mod.txt_utils import leitor_txt_emails 
from src.carregar_banco import abrir_banco



PASTA_DATA = os.path.join(RAIZ_PROJETO, "data")
arquivos_recebidos = glob.glob(os.path.join(PASTA_DATA, "**", "*"), recursive=True)
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
"""

def salvar_banco_faiss():

    dados_json = leitor_json(dados_arquivos, criar_documento)
    dados_csv = leitor_csv(dados_arquivos, criar_documento)
    dados_jsonl = leitor_jsonl(dados_arquivos, criar_documento) 
    dados_markdown = leitor_markdown(dados_arquivos, criar_documento)
    dados_txt = leitor_txt_emails(dados_arquivos, criar_documento) 


    arquivos_banco = dados_json + dados_csv + dados_jsonl + dados_markdown + dados_txt



    db = FAISS.from_documents(arquivos_banco, model)
    db.save_local('banco_faiss')


salvar_banco_faiss()
"""

query = "Quais são os produtos oferecidos pela empresa VendeFácil?"
db = abrir_banco()
docs = db.similarity_search(query)

print(docs[0].page_content)



