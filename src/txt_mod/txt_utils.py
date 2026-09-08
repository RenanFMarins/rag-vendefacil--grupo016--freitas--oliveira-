from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import os

splitter_ticket = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
)

PADRAO_INICIO_MENSAGEM = re.compile(r'^(De|From):\s*.+$', re.MULTILINE)

splitter_email = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def dividir_por_mensagem(texto):
    """
    Divide o thread de e-mail em mensagens individuais, usando a posição
    de cada cabeçalho "De:"/"From:" como marcador de início de mensagem.
    """
    posicoes = [m.start() for m in PADRAO_INICIO_MENSAGEM.finditer(texto)]

    if not posicoes:
        return [texto.strip()]  # não achou marcador -> trata como 1 bloco só

    posicoes.append(len(texto))
    mensagens = [
        texto[posicoes[i]:posicoes[i + 1]].strip()
        for i in range(len(posicoes) - 1)
        if texto[posicoes[i]:posicoes[i + 1]].strip()
    ]
    return mensagens


def leitor_txt_emails(dados_arquivos, criar_documento):
    chunks_finais = []
    for dados in dados_arquivos:
        caminho = dados["caminho"]
        nome_arquivo = os.path.basename(caminho)

        if not caminho.endswith(".txt"):
            continue

        with open(caminho, "r", encoding="utf-8") as f:
            texto = f.read()

        mensagens = dividir_por_mensagem(texto)

        for i, mensagem in enumerate(mensagens):
            partes = splitter_email.split_text(mensagem)

            for j, parte in enumerate(partes):
                documento = criar_documento(
                    texto=parte,
                    caminho=nome_arquivo,
                    doc_type=dados["doc_type"],
                    sensitivity=dados["sensitivity"],
                    chunk_id=f"{nome_arquivo}_msg{i}_p{j}",
                    message_index=i,
                    part_index=j,
                    total_parts=len(partes),
                )
                chunks_finais.append(documento)
    return chunks_finais
