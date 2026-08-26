import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


MESSAGE_SEPARATOR = re.compile(r"(?m)(?=^De:\s*)")
CUSTOMER_ID_PATTERN = re.compile(r"\bCUST\d+\b", re.IGNORECASE)
TICKET_ID_PATTERN = re.compile(r"\bTCK-\d+\b", re.IGNORECASE)
STATE_AFTER_CUSTOMER_PATTERN = re.compile(
    r"\bCUST\d+\s*/\s*([A-Z]{2})\b",
    re.IGNORECASE,
)

SENSITIVE_PATTERNS = (
    re.compile(r"\bsk_live_[A-Za-z0-9_]+", re.IGNORECASE),
    re.compile(r"\bAKIA[A-Z0-9]+"),
    re.compile(r"postgresql://\S+", re.IGNORECASE),
    re.compile(
        r"senha(?:\s+[\wáàâãéêíóôõúç]+){0,4}\s*:\s*\S+",
        re.IGNORECASE,
    ),
    re.compile(r"/\s*senha\s*['\"][^'\"]+", re.IGNORECASE),
)

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

MODULE_ALIASES = {
    "pdv": ("pdv", "cupom", "sat", "nfce", "nfc-e", "caixa"),
    "estoque": ("estoque", "inventário", "xml", "coletor"),
    "ecommerce": (
        "ecommerce",
        "e-commerce",
        "frete",
        "shopee",
        "mercado livre",
        "catálogo",
    ),
    "analytics": ("analytics", "dre", "curva abc"),
    "pay": ("vendefácil pay", "tef", "pix", "pinpad", "maquininha"),
}


def _list_txt_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".txt":
            raise ValueError(f"O arquivo informado não é TXT: {path}")
        return [path]

    if path.is_dir():
        files = sorted(path.rglob("*.txt"))
        if not files:
            raise ValueError(f"Nenhum arquivo TXT encontrado em: {path}")
        return files

    raise FileNotFoundError(f"Arquivo ou diretório não encontrado: {path}")


def _split_messages(content: str) -> list[str]:
    return [
        message.strip()
        for message in MESSAGE_SEPARATOR.split(content)
        if message.strip()
    ]


def _parse_message(message: str, file_path: Path) -> tuple[dict[str, str], str]:
    parts = re.split(r"\r?\n\s*\r?\n", message, maxsplit=1)
    header_text = parts[0]
    body = parts[1].strip() if len(parts) == 2 else ""

    headers: dict[str, str] = {}
    for line in header_text.splitlines():
        match = re.match(r"^(De|Para|Data|Assunto):\s*(.*)$", line.strip())
        if match:
            headers[match.group(1).lower()] = match.group(2).strip()

    required_headers = {"de", "para", "data", "assunto"}
    missing_headers = required_headers - headers.keys()
    if missing_headers:
        missing = ", ".join(sorted(missing_headers))
        raise ValueError(f"Cabeçalho ausente em {file_path}: {missing}")

    return headers, body


def _serialize_header(headers: dict[str, str]) -> str:
    return (
        f"E-mail de: {headers['de']}\n"
        f"Para: {headers['para']}\n"
        f"Data: {headers['data']}\n"
        f"Assunto: {headers['assunto']}"
    )


def _normalize_date(raw_date: str) -> str:
    match = re.search(
        r"(\d{1,2})\s+de\s+([A-Za-zÀ-ÿ]+)\s+de\s+(\d{4})",
        raw_date,
        re.IGNORECASE,
    )
    if not match:
        return raw_date

    day, month_name, year = match.groups()
    month = MONTHS.get(month_name.casefold())
    if month is None:
        return raw_date

    return f"{year}-{month:02d}-{int(day):02d}"


def _extract_optional_metadata(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}

    customer_match = CUSTOMER_ID_PATTERN.search(text)
    if customer_match:
        metadata["customer_id"] = customer_match.group(0).upper()

    ticket_match = TICKET_ID_PATTERN.search(text)
    if ticket_match:
        metadata["ticket_id"] = ticket_match.group(0).upper()

    state_match = STATE_AFTER_CUSTOMER_PATTERN.search(text)
    if state_match:
        metadata["state"] = state_match.group(1).upper()

    normalized_text = text.casefold()
    module_scores = {
        module: sum(normalized_text.count(alias) for alias in aliases)
        for module, aliases in MODULE_ALIASES.items()
    }
    module = max(module_scores, key=module_scores.get)
    if module_scores[module] > 0:
        metadata["module"] = module

    return metadata


def _detect_sensitivity(text: str) -> str:
    if any(pattern.search(text) for pattern in SENSITIVE_PATTERNS):
        return "restrito"
    return "interno"


def load_txt_documents(
    file_or_directory: str | Path,
    chunk_size: int = 1200,
    chunk_overlap: int = 120,
) -> list[Document]:
    path = Path(file_or_directory)
    files = _list_txt_files(path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents: list[Document] = []

    for file_path in files:
        content = file_path.read_text(encoding="utf-8")
        messages = _split_messages(content)

        for message_number, message in enumerate(messages):
            headers, body = _parse_message(message, file_path)
            serialized_header = _serialize_header(headers)
            complete_text = f"{serialized_header}\n\n{body}"
            body_parts = splitter.split_text(body) or [""]

            base_metadata = {
                "source_file": file_path.name,
                "doc_type": "email",
                "sensitivity": _detect_sensitivity(complete_text),
                "sender": headers["de"],
                "recipients": headers["para"],
                "subject": headers["assunto"],
                "date": _normalize_date(headers["data"]),
                **_extract_optional_metadata(complete_text),
            }

            for part_number, body_part in enumerate(body_parts):
                metadata = {
                    **base_metadata,
                    "chunk_id": (
                        f"{file_path.stem}:msg-{message_number:03d}:"
                        f"part-{part_number:03d}"
                    ),
                }
                documents.append(
                    Document(
                        page_content=f"{serialized_header}\n\n{body_part}".strip(),
                        metadata=metadata,
                    )
                )

    return documents
