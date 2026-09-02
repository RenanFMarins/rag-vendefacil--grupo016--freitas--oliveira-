
import json

from langchain_core.documents import Document

from ingestion.preview.preview_faiss import load_index_for_preview
from src.config import INDEX_PATHS
from src.retrieval.metadata_catalog import build_metadata_catalog


def load_indexed_documents() -> list[Document]:
    documents: list[Document] = []
    for index_path in INDEX_PATHS.values():
        vectorstore = load_index_for_preview(index_path)
        documents.extend(vectorstore.docstore._dict.values())
    return documents


def main() -> None:
    documents = load_indexed_documents()
    catalog = build_metadata_catalog(documents)

    print(f"Documents analisados: {len(documents)}")
    print(json.dumps(catalog, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

