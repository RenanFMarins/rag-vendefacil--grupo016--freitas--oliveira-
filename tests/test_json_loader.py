import json
from pathlib import Path

import pytest

from ingestion.loaders.json_loader import load_json_documents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

PRODUCTS = DATA_DIR / "structured" / "products.json"
STORES = DATA_DIR / "structured" / "stores.json"


def test_loads_products_plans_and_stores_as_records() -> None:
    assert len(load_json_documents(PRODUCTS)) == 8
    assert len(load_json_documents(STORES)) == 50


def test_serializes_product_with_responsible_people() -> None:
    documents = load_json_documents(PRODUCTS)
    product = next(doc for doc in documents if doc.metadata.get("product_id"))

    assert "Tech Lead:" in product.page_content
    assert "Product Manager:" in product.page_content
    assert product.metadata["doc_type"] == "product"
    assert product.metadata["sensitivity"] == "publico"
    assert product.metadata["module"] in {
        "analytics",
        "ecommerce",
        "estoque",
        "pay",
        "pdv",
    }


def test_keeps_pricing_plan_as_one_record() -> None:
    plans = [
        document
        for document in load_json_documents(PRODUCTS)
        if document.metadata.get("category") == "pricing_plan"
    ]

    assert len(plans) == 3
    assert {plan.metadata["plan"] for plan in plans} == {
        "Basic",
        "Pro",
        "Enterprise",
    }


def test_store_preserves_normalized_active_modules() -> None:
    store = load_json_documents(STORES)[0]

    assert store.metadata["doc_type"] == "store"
    assert store.metadata["sensitivity"] == "publico"
    assert isinstance(store.metadata["module"], tuple)
    assert set(store.metadata["module"]) <= {
        "analytics",
        "ecommerce",
        "estoque",
        "pay",
        "pdv",
    }


def test_json_chunk_ids_are_unique_and_deterministic() -> None:
    first = load_json_documents(DATA_DIR / "structured")
    second = load_json_documents(DATA_DIR / "structured")
    first_ids = [document.metadata["chunk_id"] for document in first]

    assert len(first_ids) == len(set(first_ids)) == 58
    assert first_ids == [
        document.metadata["chunk_id"] for document in second
    ]


def test_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "products.json"
    path.write_text("{json inválido}", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON inválido"):
        load_json_documents(path)


def test_reports_missing_structure(tmp_path: Path) -> None:
    path = tmp_path / "stores.json"
    path.write_text(json.dumps({"outro": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="Campos ausentes"):
        load_json_documents(path)


def test_rejects_unknown_json(tmp_path: Path) -> None:
    path = tmp_path / "unknown.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON não reconhecido"):
        load_json_documents(path)
