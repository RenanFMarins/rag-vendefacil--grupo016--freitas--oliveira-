import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


DOC_TYPE_BY_GROUP = {
    "documentation": "manual",
    "meetings": "ata",
    "policies": "policy",
}

PUBLIC_POLICY_FILES = {
    "atendimento_sla.md",
    "codigo_de_conduta.md",
    "reembolso.md",
    "seguranca_lgpd.md",
}

RESTRICTED_MEETING_FILES = {
    "2026-03-hr_performance_review_q1.md",
}

KNOWN_MODULES = {
    "analytics",
    "ecommerce",
    "estoque",
    "pay",
    "pdv",
}

MONTHS = {
    "janeiro": 1,
    "fevereiro": 2,
    "março": 3,
    "abril": 4,
    "maio": 5,
    "junho": 6,
    "julho": 7,
    "agosto": 8,
    "setembro": 9,
    "outubro": 10,
    "novembro": 11,
    "dezembro": 12,
}


def _list_markdown_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".md":
            raise ValueError(f"O arquivo informado não é Markdown: {path}")
        return [path]

    if path.is_dir():
        files = sorted(path.rglob("*.md"))
        if not files:
            raise ValueError(f"Nenhum arquivo Markdown encontrado em: {path}")
        return files

    raise FileNotFoundError(f"Arquivo ou diretório não encontrado: {path}")


def _source_group(file_path: Path) -> str:
    for part in file_path.parts:
        if part in DOC_TYPE_BY_GROUP:
            return part

    raise ValueError(
        "Não foi possível classificar o Markdown pelo caminho: "
        f"{file_path}"
    )


def _extract_module(file_path: Path, group: str) -> str | None:
    if group != "documentation":
        return None

    parts = file_path.parts
    group_position = parts.index(group)
    if group_position + 1 >= len(parts):
        return None

    module = parts[group_position + 1].casefold()
    return module if module in KNOWN_MODULES else None


def _classify_sensitivity(file_path: Path, group: str) -> str:
    if group == "documentation":
        return "publico"

    if group == "meetings":
        if file_path.name in RESTRICTED_MEETING_FILES:
            return "restrito"
        return "interno"

    if file_path.name in PUBLIC_POLICY_FILES:
        return "publico"
    return "interno"


def _extract_date(content: str) -> str | None:
    match = re.search(
        r"(?:\*\*Data:\*\*\s*)?"
        r"(\d{1,2})\s+de\s+([A-Za-zÀ-ÿ]+)\s+de\s+(\d{4})",
        content,
        re.IGNORECASE,
    )
    if not match:
        return None

    day, month_name, year = match.groups()
    month = MONTHS.get(month_name.casefold())
    if month is None:
        return None

    return f"{year}-{month:02d}-{int(day):02d}"


def _section_path(metadata: dict[str, str], fallback_title: str) -> str:
    headings = [
        metadata[key]
        for key in ("title", "section", "subsection")
        if metadata.get(key)
    ]
    return " > ".join(headings) or fallback_title


def load_markdown_documents(
    file_or_directory: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    path = Path(file_or_directory)
    files = _list_markdown_files(path)
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "title"),
            ("##", "section"),
            ("###", "subsection"),
        ],
        strip_headers=True,
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents: list[Document] = []

    for file_path in files:
        content = file_path.read_text(encoding="utf-8")
        if not content.strip():
            raise ValueError(f"Arquivo Markdown vazio: {file_path}")

        group = _source_group(file_path)
        sections = header_splitter.split_text(content)
        module = _extract_module(file_path, group)
        date = _extract_date(content)

        base_metadata = {
            "source_file": file_path.name,
            "doc_type": DOC_TYPE_BY_GROUP[group],
            "sensitivity": _classify_sensitivity(file_path, group),
        }
        if module:
            base_metadata["module"] = module
        if date:
            base_metadata["date"] = date

        for section_number, section_document in enumerate(sections):
            section = _section_path(
                section_document.metadata,
                fallback_title=file_path.stem.replace("_", " "),
            )
            section_parts = size_splitter.split_text(
                section_document.page_content
            ) or [""]

            for part_number, section_part in enumerate(section_parts):
                metadata = {
                    **base_metadata,
                    "section": section,
                    "chunk_id": (
                        f"{DOC_TYPE_BY_GROUP[group]}:{file_path.stem}:"
                        f"section-{section_number:03d}:part-{part_number:03d}"
                    ),
                }
                documents.append(
                    Document(
                        page_content=(
                            f"Seção: {section}\n\n{section_part}"
                        ).strip(),
                        metadata=metadata,
                    )
                )

    return documents
