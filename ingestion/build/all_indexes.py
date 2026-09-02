from ingestion.build.builder_file.csv_builder import DEFAULT_INDEX as CSV_INDEX
from ingestion.build.builder_file.csv_builder import build_csv_index
from ingestion.build.builder_file.json_builder import DEFAULT_INDEX as JSON_INDEX
from ingestion.build.builder_file.json_builder import build_json_index
from ingestion.build.builder_file.jsonl_builder import DEFAULT_INDEX as JSONL_INDEX
from ingestion.build.builder_file.jsonl_builder import build_jsonl_index
from ingestion.build.builder_file.markdown_builder import DEFAULT_INDEX as MARKDOWN_INDEX
from ingestion.build.builder_file.markdown_builder import build_markdown_index
from ingestion.build.builder_file.pdf_builder import DEFAULT_INDEX as PDF_INDEX
from ingestion.build.builder_file.pdf_builder import build_pdf_index
from ingestion.build.builder_file.txt_builder import DEFAULT_INDEX as TXT_INDEX
from ingestion.build.builder_file.txt_builder import build_txt_index


def main() -> None:
    builders = [
        ("csv", build_csv_index, CSV_INDEX),
        ("json", build_json_index, JSON_INDEX),
        ("jsonl", build_jsonl_index, JSONL_INDEX),
        ("markdown", build_markdown_index, MARKDOWN_INDEX),
        ("pdf", build_pdf_index, PDF_INDEX),
        ("txt", build_txt_index, TXT_INDEX),
    ]

    for format_name, builder, index_path in builders:
        print(f"Gerando índice {format_name}...")
        vectorstore = builder()
        print(f"Vetores indexados: {vectorstore.index.ntotal}")
        print(f"Salvo em: {index_path}\n")


if __name__ == "__main__":
    main()
