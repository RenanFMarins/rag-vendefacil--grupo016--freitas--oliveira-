from copy import deepcopy

from langchain_core.documents import Document

from src.retrieval.metadata_catalog import build_metadata_catalog


def test_builds_deduplicated_deterministic_catalog_and_ignores_none() -> None:
    documents = [
        Document(
            page_content="Primeiro chunk",
            metadata={
                "doc_type": "ticket",
                "state": "SP",
                "module": "pdv",
                "plan": "Enterprise",
                "priority": "Alta",
                "customer_id": "CUST001",
            },
        ),
        Document(
            page_content="Segundo chunk",
            metadata={
                "doc_type": "manual",
                "state": "MG",
                "module": "estoque",
                "plan": "Basic",
                "priority": None,
                "customer_id": "CUST002",
            },
        ),
        Document(
            page_content="Terceiro chunk",
            metadata={
                "doc_type": "ticket",
                "state": "SP",
                "module": "pdv",
                "plan": "Enterprise",
                "priority": "Alta",
                "customer_id": "CUST003",
            },
        ),
    ]

    catalog = build_metadata_catalog(documents)

    assert catalog == {
        "doc_type": ["manual", "ticket"],
        "state": ["MG", "SP"],
        "module": ["estoque", "pdv"],
        "plan": ["Basic", "Enterprise"],
        "priority": ["Alta"],
    }
    assert "customer_id" not in catalog


def test_does_not_modify_documents() -> None:
    documents = [
        Document(
            page_content="Chunk",
            metadata={"doc_type": "ticket", "state": "MG"},
        )
    ]
    original_metadata = deepcopy(documents[0].metadata)

    build_metadata_catalog(documents)

    assert documents[0].metadata == original_metadata


def test_accepts_explicit_high_cardinality_fields() -> None:
    documents = [
        Document(
            page_content="Chunk A",
            metadata={"customer_id": "CUST002"},
        ),
        Document(
            page_content="Chunk B",
            metadata={"customer_id": "CUST001"},
        ),
    ]

    catalog = build_metadata_catalog(documents, fields=("customer_id",))

    assert catalog == {"customer_id": ["CUST001", "CUST002"]}


def test_expands_multivalued_metadata_into_individual_catalog_values() -> None:
    documents = [
        Document(
            page_content="Loja com dois módulos",
            metadata={"module": ("estoque", "pdv")},
        ),
        Document(
            page_content="Loja com módulo repetido",
            metadata={"module": ["pdv", "pay"]},
        ),
    ]

    catalog = build_metadata_catalog(documents)

    assert catalog == {"module": ["estoque", "pay", "pdv"]}

