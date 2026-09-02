import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from ingestion.build._faiss import load_faiss_index
from ingestion.build.builder_file.csv_builder import build_csv_documents
from ingestion.build.builder_file.csv_builder import build_csv_index
from ingestion.build.builder_file.json_builder import build_json_documents
from ingestion.build.builder_file.json_builder import build_json_index
from ingestion.build.builder_file.jsonl_builder import build_jsonl_documents
from ingestion.build.builder_file.jsonl_builder import build_jsonl_index
from ingestion.build.builder_file.markdown_builder import build_markdown_documents
from ingestion.build.builder_file.markdown_builder import build_markdown_index
from ingestion.build.builder_file.pdf_builder import build_pdf_documents
from ingestion.build.builder_file.pdf_builder import build_pdf_index
from ingestion.build.builder_file.txt_builder import build_txt_documents
from ingestion.build.builder_file.txt_builder import build_txt_index


REQUIRED_METADATA = {
    "source_file",
    "doc_type",
    "chunk_id",
    "sensitivity",
}


def test_builds_expected_documents_for_each_format() -> None:
    assert len(build_csv_documents()) == 5460
    assert len(build_json_documents()) == 58
    assert len(build_jsonl_documents()) == 75
    assert len(build_markdown_documents()) == 81
    assert len(build_pdf_documents()) == 6
    assert len(build_txt_documents()) == 43


def test_builders_share_the_required_metadata_contract() -> None:
    documents = [
        *build_csv_documents(),
        *build_json_documents(),
        *build_jsonl_documents(),
        *build_markdown_documents(),
        *build_pdf_documents(),
        *build_txt_documents(),
    ]

    assert len(documents) == 5723
    assert all(REQUIRED_METADATA <= doc.metadata.keys() for doc in documents)
    assert len({doc.metadata["chunk_id"] for doc in documents}) == 5723


@pytest.mark.parametrize(
    ("index_builder", "expected_total"),
    [
        (build_csv_index, 5460),
        (build_json_index, 58),
        (build_jsonl_index, 75),
        (build_markdown_index, 81),
        (build_pdf_index, 6),
        (build_txt_index, 43),
    ],
)
def test_persists_and_reloads_one_faiss_index_per_format(
    tmp_path,
    index_builder,
    expected_total,
) -> None:
    embeddings = DeterministicFakeEmbedding(size=32)
    index_path = tmp_path / index_builder.__name__

    created = index_builder(index_path=index_path, embeddings=embeddings)

    assert created.index.ntotal == expected_total
    assert (index_path / "index.faiss").is_file()
    assert (index_path / "index.pkl").is_file()

    loaded = load_faiss_index(index_path, embeddings=embeddings)

    assert loaded.index.ntotal == expected_total
    assert len(loaded.docstore._dict) == expected_total
