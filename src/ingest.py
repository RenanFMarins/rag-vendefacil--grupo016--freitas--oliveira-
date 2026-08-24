from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import glob

def load_documents():
   laoder_pdf = DirectoryLoader(
        "../data",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
   )

   documentos_pdf = laoder_pdf.load()

   text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
            add_start_index=True
        )

   chunks = text_splitter.split_documents(documentos_pdf)
   return chunks

load_documents()  






        

    