"""Ingestão adaptativa dos registros estruturados em JSON."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


PRODUCT_FIELDS = {
    "product_id",
    "name",
    "category",
    "description",
    "tech_lead",
    "product_manager",
    "pricing_by_plan",
    "features",
    "supported_os",
    "sla_uptime",
}
STORE_FIELDS = {
    "store_id",
    "customer_id",
    "company_name",
    "store_name",
    "state",
    "city",
    "pos_terminals_count",
    "active_modules",
}
PLAN_FIELDS = {
    "monthly_fee_brl",
    "included_terminals",
    "extra_terminal_fee_brl",
    "description",
}
MODULE_ALIASES = {
    "analytics": "analytics",
    "estoque": "estoque",
    "loja": "ecommerce",
    "ecommerce": "ecommerce",
    "pay": "pay",
    "pdv": "pdv",
}


def _list_json_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.casefold() != ".json":
            raise ValueError(f"O arquivo informado não é JSON: {path}")
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("*.json"))
        if not files:
            raise ValueError(f"Nenhum arquivo JSON encontrado em: {path}")
        return files
    raise FileNotFoundError(f"Arquivo ou diretório não encontrado: {path}")


def _require_mapping(
    value: object,
    *,
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Estrutura JSON inválida em {context}: objeto esperado.")
    return value


def _validate_fields(
    record: Mapping[str, Any],
    required: set[str],
    context: str,
) -> None:
    missing = required - record.keys()
    if missing:
        fields = ", ".join(sorted(missing))
        raise ValueError(f"Campos ausentes em {context}: {fields}")


def _normalize_module(value: str) -> str:
    normalized = value.casefold().replace("e-commerce", "ecommerce")
    for alias, module in MODULE_ALIASES.items():
        if alias in normalized:
            return module
    return normalized.strip()


def _serialize_product(
    product: Mapping[str, Any],
    company: str,
    last_updated: str,
) -> str:
    features = "; ".join(map(str, product["features"]))
    supported_os = ", ".join(map(str, product["supported_os"]))
    standalone_price = product.get("standalone_monthly_price_brl")
    price_text = (
        f"Preço mensal avulso de R$ {standalone_price}."
        if standalone_price is not None
        else "Preço mensal avulso não informado."
    )
    return (
        f"Produto {product['product_id']}: {product['name']}, oferecido pela "
        f"{company}, categoria {product['category']}. "
        f"{product['description']} Tech Lead: {product['tech_lead']}; "
        f"Product Manager: {product['product_manager']}. {price_text} "
        f"Funcionalidades: "
        f"{features}. Sistemas suportados: {supported_os}. SLA de "
        f"disponibilidade: {product['sla_uptime']}. Atualizado em "
        f"{last_updated}."
    )


def _serialize_plan(
    plan_name: str,
    plan: Mapping[str, Any],
    company: str,
    last_updated: str,
) -> str:
    return (
        f"Plano {plan_name} da {company}: mensalidade de "
        f"R$ {plan['monthly_fee_brl']}, com "
        f"{plan['included_terminals']} terminais incluídos e valor de "
        f"R$ {plan['extra_terminal_fee_brl']} por terminal adicional. "
        f"{plan['description']} Atualizado em {last_updated}."
    )


def _serialize_store(store: Mapping[str, Any]) -> str:
    modules = ", ".join(map(str, store["active_modules"]))
    return (
        f"Loja {store['store_id']}: {store['store_name']}, pertencente ao "
        f"cliente {store['company_name']} ({store['customer_id']}), localizada "
        f"em {store['city']}/{store['state']}. Possui "
        f"{store['pos_terminals_count']} terminais PDV e os módulos ativos: "
        f"{modules}."
    )


def _load_products(file_path: Path, payload: Mapping[str, Any]) -> list[Document]:
    required_root = {"company", "last_updated", "pricing_plans", "products"}
    _validate_fields(payload, required_root, str(file_path))
    plans = _require_mapping(
        payload["pricing_plans"],
        context=f"{file_path}:pricing_plans",
    )
    products = payload["products"]
    if not isinstance(products, list):
        raise ValueError(f"Estrutura JSON inválida em {file_path}: products.")

    company = str(payload["company"])
    last_updated = str(payload["last_updated"])
    documents: list[Document] = []
    for plan_name in sorted(plans):
        plan = _require_mapping(
            plans[plan_name],
            context=f"{file_path}:plano:{plan_name}",
        )
        _validate_fields(plan, PLAN_FIELDS, f"{file_path}:plano:{plan_name}")
        plan_key = plan_name.casefold().replace(" ", "-")
        documents.append(
            Document(
                page_content=_serialize_plan(
                    plan_name,
                    plan,
                    company,
                    last_updated,
                ),
                metadata={
                    "source_file": file_path.name,
                    "doc_type": "product",
                    "chunk_id": f"{file_path.stem}:plan-{plan_key}",
                    "sensitivity": "publico",
                    "category": "pricing_plan",
                    "plan": plan_name,
                    "date": last_updated,
                },
            )
        )

    for position, raw_product in enumerate(products, start=1):
        product = _require_mapping(
            raw_product,
            context=f"{file_path}:produto:{position}",
        )
        _validate_fields(
            product,
            PRODUCT_FIELDS,
            f"{file_path}:produto:{position}",
        )
        documents.append(
            Document(
                page_content=_serialize_product(
                    product,
                    company,
                    last_updated,
                ),
                metadata={
                    "source_file": file_path.name,
                    "doc_type": "product",
                    "chunk_id": f"{file_path.stem}:{product['product_id']}",
                    "sensitivity": "publico",
                    "product_id": str(product["product_id"]),
                    "category": str(product["category"]),
                    "module": _normalize_module(str(product["name"])),
                    "product_manager": str(product["product_manager"]),
                    "tech_lead": str(product["tech_lead"]),
                    "date": last_updated,
                },
            )
        )
    return documents


def _load_stores(file_path: Path, payload: Mapping[str, Any]) -> list[Document]:
    _validate_fields(payload, {"network_stores"}, str(file_path))
    stores = payload["network_stores"]
    if not isinstance(stores, list):
        raise ValueError(
            f"Estrutura JSON inválida em {file_path}: network_stores."
        )

    documents: list[Document] = []
    for position, raw_store in enumerate(stores, start=1):
        store = _require_mapping(
            raw_store,
            context=f"{file_path}:loja:{position}",
        )
        _validate_fields(store, STORE_FIELDS, f"{file_path}:loja:{position}")
        active_modules = store["active_modules"]
        if not isinstance(active_modules, list) or not active_modules:
            raise ValueError(
                f"Módulos inválidos em {file_path}:loja:{position}."
            )
        modules = tuple(
            _normalize_module(str(module)) for module in active_modules
        )
        documents.append(
            Document(
                page_content=_serialize_store(store),
                metadata={
                    "source_file": file_path.name,
                    "doc_type": "store",
                    "chunk_id": f"{file_path.stem}:{store['store_id']}",
                    "sensitivity": "publico",
                    "store_id": str(store["store_id"]),
                    "customer_id": str(store["customer_id"]),
                    "state": str(store["state"]),
                    "city": str(store["city"]),
                    "module": modules,
                },
            )
        )
    return documents


def load_json_documents(file_or_directory: str | Path) -> list[Document]:
    """Converte produtos, planos e lojas em registros sem fragmentá-los."""
    files = _list_json_files(Path(file_or_directory))
    documents: list[Document] = []
    for file_path in files:
        if file_path.name not in {"products.json", "stores.json"}:
            raise ValueError(f"Arquivo JSON não reconhecido: {file_path}")
        try:
            raw_payload = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"JSON inválido em {file_path}") from error
        payload = _require_mapping(raw_payload, context=str(file_path))
        loader = _load_products if file_path.name == "products.json" else _load_stores
        documents.extend(loader(file_path, payload))
    return documents

