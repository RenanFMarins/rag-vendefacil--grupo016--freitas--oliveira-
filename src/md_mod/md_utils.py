from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import os

headers_para_split = [
    ("#", "titulo_h1"),
    ("##", "titulo_h2"),
    ("###", "titulo_h3"),
]

splitter_headers = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_para_split)

# Fallback: se mesmo assim uma seção ficar grande demais, divide por tamanho
splitter_fallback = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def leitor_markdown(dados_arquivos, criar_documento):
    chunks_finais = []
    for dados in dados_arquivos:
        caminho = dados["caminho"]
        nome_arquivo = os.path.basename(caminho)

        if not caminho.endswith(".md"):
            continue

        with open(caminho, "r", encoding="utf-8") as f:
            texto = f.read()

        secoes = splitter_headers.split_text(texto)

        for i, secao in enumerate(secoes):

            sub_partes = splitter_fallback.split_text(secao.page_content)

            for j, sub_parte in enumerate(sub_partes):
                documento = criar_documento(
                    texto=sub_parte,
                    caminho=nome_arquivo,
                    doc_type=dados["doc_type"],
                    sensitivity=dados["sensitivity"],
                    chunk_id=f"{nome_arquivo}_sec{i}_p{j}",
                    titulo_h1=secao.metadata.get("titulo_h1"),
                    titulo_h2=secao.metadata.get("titulo_h2"),
                    titulo_h3=secao.metadata.get("titulo_h3"),
                    part_index=j,
                    total_parts=len(sub_partes),
                )
                chunks_finais.append(documento)

    return chunks_finais
