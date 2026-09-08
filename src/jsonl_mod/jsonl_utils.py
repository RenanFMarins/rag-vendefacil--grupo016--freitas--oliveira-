import os
import sys
# fmt: off
# isort: skip


import json
import re

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter_ticket = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
)

#

def criar_documento(texto, caminho, doc_type, sensitivity, chunk_id, **kwargs):
    metadata = {
        "source_file": caminho,
        "doc_type": doc_type,
        "chunk_id": chunk_id,
        "sensitivity": sensitivity,

    }
    metadata.update(kwargs)

    return Document(
        page_content=texto,
        metadata=metadata
    )


def processar_ticket(linha):
    """
    Serializa o cabeçalho do ticket em linguagem natural, com os campos
    reais do tickets.jsonl.
    """
    texto = (
        f"Chamado de suporte {linha['ticket_id']}, aberto pelo cliente "
        f"{linha['customer_name']} (ID: {linha['customer_id']}) do estado {linha['state']}, "
        f"referente ao módulo {linha['module']}. Título: {linha['title']}. "
        f"Categoria: {linha['category']}. Prioridade: {linha['priority']}. "
        f"Status: {linha['status']}. Criado em {linha['created_at']}."
    )
    if linha.get("sentiment"):
        texto += f" Sentimento do cliente: {linha['sentiment']}."
    return texto


def leitor_jsonl(dados_arquivos, criar_documento):
    chunks_finais = []
    for dados in dados_arquivos:
        caminho = dados["caminho"]
        nome_arquivo = os.path.basename(caminho)

        if not caminho.endswith(".jsonl"):
            continue

        with open(caminho, "r", encoding="utf-8") as f:
            for linha_bruta in f:
                if not linha_bruta.strip():
                    continue

                linha = json.loads(linha_bruta)
                cabecalho = processar_ticket(linha)

                # Corpo = descrição + resolução (quando existir)
                corpo = linha.get("description", "").strip()
                if linha.get("resolution"):
                    corpo += f"\nResolução: {linha['resolution'].strip()}"

                texto_completo = f"{cabecalho}\nDescrição: {corpo}" if corpo else cabecalho
                partes_corpo = splitter_ticket.split_text(texto_completo)

                for i, parte in enumerate(partes_corpo):
                    documento = criar_documento(
                        texto=parte,
                        caminho=nome_arquivo,
                        doc_type=dados["doc_type"],
                        sensitivity=dados["sensitivity"],
                        chunk_id=f"tickets_{linha['ticket_id']}_p{i}",
                        ticket_id=linha["ticket_id"],
                        customer_id=linha.get("customer_id"),
                        customer_name=linha.get("customer_name"),
                        state=linha.get("state"),
                        module=linha.get("module"),
                        category=linha.get("category"),
                        priority=linha.get("priority"),
                        status=linha.get("status"),
                        sentiment=linha.get("sentiment"),
                        part_index=i,
                        total_parts=len(partes_corpo),
                    )
                    chunks_finais.append(documento)

    return chunks_finais

