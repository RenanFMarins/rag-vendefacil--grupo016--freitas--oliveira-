import csv
import re
from collections.abc import Callable, Mapping
from pathlib import Path

from langchain_core.documents import Document


Row = Mapping[str, str]
Serializer = Callable[[Row], str]

REQUIRED_COLUMNS = {
    "customers.csv": {
        "customer_id",
        "company_name",
        "cnpj",
        "state",
        "city",
        "segment",
        "plan",
        "main_product",
        "mrr",
        "status",
        "contact_email",
    },
    "employees.csv": {
        "id",
        "name",
        "email",
        "department",
        "role",
        "hire_date",
        "salary",
        "status",
    },
    "system_logs.csv": {
        "timestamp",
        "level",
        "service",
        "module",
        "customer_id",
        "event",
        "error_code",
        "message",
    },
    "sales.csv": {
        "sale_id",
        "customer_id",
        "company_name",
        "store_id",
        "store_name",
        "state",
        "city",
        "product_id",
        "product_name",
        "date",
        "payment_method",
        "amount_brl",
        "pos_terminal",
        "status",
    },
}

CREDENTIAL_PATTERNS = (
    re.compile(r"\bsenha\b", re.IGNORECASE),
    re.compile(r"\bpassword\b", re.IGNORECASE),
    re.compile(r"\btoken\b", re.IGNORECASE),
    re.compile(r"\bapi[ _-]?key\b", re.IGNORECASE),
    re.compile(r"\bchave\s+(?:pix|secreta|privada)\b", re.IGNORECASE),
    re.compile(r"\bcredencia(?:l|is)\b", re.IGNORECASE),
)

MODULE_ALIASES = {
    "analytics": "analytics",
    "estoque": "estoque",
    "loja": "ecommerce",
    "ecommerce": "ecommerce",
    "pay": "pay",
    "pdv": "pdv",
}


def _list_csv_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.casefold() != ".csv":
            raise ValueError(f"O arquivo informado não é CSV: {path}")
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("*.csv"))
        if not files:
            raise ValueError(f"Nenhum arquivo CSV encontrado em: {path}")
        return files
    raise FileNotFoundError(f"Arquivo ou diretório não encontrado: {path}")


def _serialize_customer(row: Row) -> str:
    return (
        f"Cliente {row['customer_id']}: {row['company_name']}, "
        f"CNPJ {row['cnpj']}, localizado em {row['city']}/{row['state']}, "
        f"segmento {row['segment']}. Plano {row['plan']}, produto principal "
        f"{row['main_product']}, MRR de R$ {row['mrr']}, situação "
        f"{row['status']} e contato {row['contact_email']}."
    )


def _serialize_employee(row: Row) -> str:
    return (
        f"Funcionário {row['id']}: {row['name']}, departamento "
        f"{row['department']}, cargo {row['role']}, contratado em "
        f"{row['hire_date']}, situação {row['status']}. E-mail profissional "
        f"{row['email']} e salário individual de R$ {row['salary']}."
    )


def _serialize_system_log(row: Row) -> str:
    return (
        f"Log registrado em {row['timestamp']}: nível {row['level']}, "
        f"serviço {row['service']}, módulo {row['module']}, cliente "
        f"{row['customer_id']}, evento {row['event']}, código "
        f"{row['error_code']}. Mensagem: {row['message']}"
    )


def _serialize_sale(row: Row) -> str:
    return (
        f"Venda {row['sale_id']} realizada em {row['date']} na loja "
        f"{row['store_name']} ({row['store_id']}), em "
        f"{row['city']}/{row['state']}, para o cliente "
        f"{row['company_name']} ({row['customer_id']}). Produto "
        f"{row['product_name']} ({row['product_id']}), valor de "
        f"R$ {row['amount_brl']}, pagamento {row['payment_method']}, "
        f"terminal {row['pos_terminal']} e situação {row['status']}."
    )


SERIALIZERS: dict[str, Serializer] = {
    "customers.csv": _serialize_customer,
    "employees.csv": _serialize_employee,
    "system_logs.csv": _serialize_system_log,
    "sales.csv": _serialize_sale,
}


def _log_sensitivity(row: Row) -> str:
    searchable_text = " ".join(
        (row["event"], row["error_code"], row["message"])
    )
    if any(pattern.search(searchable_text) for pattern in CREDENTIAL_PATTERNS):
        return "restrito"
    return "interno"


def _normalize_module(value: str) -> str:
    normalized = value.casefold().replace("e-commerce", "ecommerce")
    for alias, module in MODULE_ALIASES.items():
        if alias in normalized:
            return module
    return normalized.strip()


def _base_metadata(
    file_path: Path,
    row: Row,
    row_number: int,
) -> dict[str, object]:
    stem = file_path.stem
    if file_path.name == "customers.csv":
        return {
            "source_file": file_path.name,
            "doc_type": "customer",
            "chunk_id": f"{stem}:{row['customer_id']}",
            "sensitivity": "interno",
            "customer_id": row["customer_id"],
            "state": row["state"],
            "city": row["city"],
            "segment": row["segment"],
            "plan": row["plan"],
            "module": _normalize_module(row["main_product"]),
            "status": row["status"],
        }
    if file_path.name == "employees.csv":
        return {
            "source_file": file_path.name,
            "doc_type": "employee",
            "chunk_id": f"{stem}:{row['id']}",
            "sensitivity": "restrito",
            "employee_id": row["id"],
            "department": row["department"],
            "role": row["role"],
            "status": row["status"],
            "date": row["hire_date"],
        }
    if file_path.name == "system_logs.csv":
        return {
            "source_file": file_path.name,
            "doc_type": "log",
            "chunk_id": f"{stem}:row-{row_number:06d}",
            "sensitivity": _log_sensitivity(row),
            "customer_id": row["customer_id"],
            "module": _normalize_module(row["module"]),
            "level": row["level"],
            "service": row["service"],
            "event": row["event"],
            "error_code": row["error_code"],
            "date": row["timestamp"][:10],
        }
    return {
        "source_file": file_path.name,
        "doc_type": "sale",
        "chunk_id": f"{stem}:{row['sale_id']}",
        "sensitivity": "interno",
        "sale_id": row["sale_id"],
        "customer_id": row["customer_id"],
        "store_id": row["store_id"],
        "product_id": row["product_id"],
        "state": row["state"],
        "city": row["city"],
        "module": _normalize_module(row["product_name"]),
        "date": row["date"],
        "status": row["status"],
    }


def _validate_header(file_path: Path, fieldnames: list[str] | None) -> None:
    actual_columns = set(fieldnames or [])
    missing = REQUIRED_COLUMNS[file_path.name] - actual_columns
    if missing:
        fields = ", ".join(sorted(missing))
        raise ValueError(f"Colunas ausentes em {file_path}: {fields}")


def _validate_row(file_path: Path, row: Row, row_number: int) -> None:
    empty_fields = [
        field
        for field in REQUIRED_COLUMNS[file_path.name]
        if not str(row.get(field, "")).strip()
    ]
    if empty_fields:
        fields = ", ".join(sorted(empty_fields))
        raise ValueError(
            f"Valores obrigatórios vazios em {file_path}, "
            f"linha {row_number}: {fields}"
        )


def load_csv_documents(file_or_directory: str | Path) -> list[Document]:
    """Converte cada registro CSV em um Document sem dividir a linha."""
    files = _list_csv_files(Path(file_or_directory))
    documents: list[Document] = []

    for file_path in files:
        if file_path.name not in SERIALIZERS:
            raise ValueError(f"Arquivo CSV não reconhecido: {file_path}")

        with file_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            _validate_header(file_path, reader.fieldnames)
            for row_number, row in enumerate(reader, start=1):
                _validate_row(file_path, row, row_number)
                documents.append(
                    Document(
                        page_content=SERIALIZERS[file_path.name](row),
                        metadata=_base_metadata(
                            file_path,
                            row,
                            row_number,
                        ),
                    )
                )

    return documents
