"""Ingestão adaptativa de políticas em PDF."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError


PUBLIC_POLICY_STEMS = {"reembolso", "seguranca_lgpd"}


def _list_pdf_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.casefold() != ".pdf":
            raise ValueError(f"O arquivo informado não é PDF: {path}")
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("*.pdf"))
        if not files:
            raise ValueError(f"Nenhum arquivo PDF encontrado em: {path}")
        return files
    raise FileNotFoundError(f"Arquivo ou diretório não encontrado: {path}")


def _sensitivity(file_path: Path) -> str:
    if file_path.stem in PUBLIC_POLICY_STEMS:
        return "publico"
    return "interno"


def _read_pages(file_path: Path) -> list[str]:
    try:
        reader = PdfReader(file_path)
        return [(page.extract_text() or "").strip() for page in reader.pages]
    except (PdfReadError, OSError) as error:
        raise ValueError(f"Não foi possível ler o PDF: {file_path}") from error


def load_pdf_documents(
    file_or_directory: str | Path,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Document]:
    """Extrai páginas e divide somente textos longos por parágrafo."""
    if chunk_size < 1:
        raise ValueError("chunk_size deve ser maior que zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap deve ser maior ou igual a zero e menor que chunk_size."
        )

    files = _list_pdf_files(Path(file_or_directory))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents: list[Document] = []

    for file_path in files:
        pages = _read_pages(file_path)
        file_documents: list[Document] = []
        for page_number, page_text in enumerate(pages, start=1):
            if not page_text:
                continue
            parts = splitter.split_text(page_text)
            for part_number, part in enumerate(parts):
                file_documents.append(
                    Document(
                        page_content=f"Página {page_number}\n\n{part}".strip(),
                        metadata={
                            "source_file": file_path.name,
                            "doc_type": "policy",
                            "chunk_id": (
                                f"policy:{file_path.stem}:"
                                f"page-{page_number:03d}:part-{part_number:03d}"
                            ),
                            "sensitivity": _sensitivity(file_path),
                            "page": page_number,
                        },
                    )
                )
        if not file_documents:
            raise ValueError(f"PDF sem texto extraível: {file_path}")
        documents.extend(file_documents)

    return documents
