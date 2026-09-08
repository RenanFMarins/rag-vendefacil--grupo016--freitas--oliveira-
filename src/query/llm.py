from groq import Groq
from pydantic import BaseModel, ValidationError
import os
import getpass
from src.query.schema import SourceEvidence, RAGResponse

if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for GROQ: ")

cliente = Groq()


def montar_contexto(chunks_top_k):

    campos_ja_mostrados = {"source_file", "chunk_id", "sensitivity"}

    partes = []

    for i, doc in enumerate(chunks_top_k):
        fonte = doc.metadata.get("source_file", "desconhecido")
        chunk_id = doc.metadata.get("chunk_id", f"chunk_{i}")
        partes.append(
            f"[Fonte {i+1} | arquivo={fonte} | chunk_id={chunk_id}]\n{doc.page_content}")

        metadados_extra = "\n".join(
            f"{chave}: {valor}"
            for chave, valor in doc.metadata.items()
            if chave not in campos_ja_mostrados and valor not in (None, "")
        )

        bloco = f"[Fonte {i+1} | arquivo={fonte} | chunk_id={chunk_id}]\n{doc.page_content}"
        if metadados_extra:
            bloco += f"\nMetadados adicionais:\n{metadados_extra}"

        partes.append(bloco)
        print("-" * 60, "\n\n".join(partes))
    return "\n\n".join(partes)


def gerar_resposta_llm(pergunta, chunks_top_k, nivel_confianca):
    contexto = montar_contexto(chunks_top_k)

    prompt_sistema = """
        Você é o assistente RAG da VendeFácil Tecnologia Ltda.

        Sua tarefa é responder à pergunta do usuário EXCLUSIVAMENTE com base nas informações presentes no CONTEXTO fornecido.

        REGRAS:
        1. Não use conhecimento externo ou conhecimento próprio do modelo.
        2. Não invente, complete ou suponha informações que não estejam no contexto.
        3. Se o contexto não possuir informação suficiente para responder, informe que não há evidências suficientes.
        4. Responda de forma objetiva e diretamente relacionada à pergunta.
        5. Para cada informação relevante apresentada, indique a fonte correspondente.
        6. As fontes devem utilizar somente documentos presentes no contexto.
        7. Não crie nomes de arquivos, trechos, IDs ou informações que não estejam no contexto.
        8. O trecho utilizado como evidência deve corresponder ao conteúdo realmente presente no contexto.
        9. Se houver informações conflitantes entre fontes, informe a existência do conflito em vez de escolher uma informação arbitrariamente.
        10. Respeite as regras de privacidade e LGPD já aplicadas pela pipeline.

        IMPORTANTE:
        - O CONTEXTO é a única fonte de verdade para esta resposta.
        - A pergunta do usuário serve apenas para determinar o que deve ser buscado no contexto.
        - Não responda utilizando informações que não possam ser sustentadas pelas evidências recuperadas.
        """

    prompt_usuario = f"CONTEXTO:\n{contexto}\n\nPERGUNTA: {pergunta}"

    class RespostaLLM(BaseModel):
        answer: str
        reasoning: str
        sources_used: list[SourceEvidence]

    resposta = cliente.chat.completions.create(
        model='openai/gpt-oss-120b',
        max_tokens=2000,
        temperature=0,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": prompt_usuario}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "resposta_rag",
                "strict": False,
                "schema": RespostaLLM.model_json_schema(),
            },
        }
    )

    texto_json = resposta.choices[0].message.content

    try:
        resposta_llm = RespostaLLM.model_validate_json(texto_json)

    except ValidationError as e:
        return RAGResponse(
            answer="Não foi possível gerar uma resposta estruturada no momento.",
            confidence_level="Recusado",
            sources_used=[],
            reasoning=f"Erro de validação do schema: {e}",
            is_refusal=True,
            refusal_reason="ERRO_VALIDACAO_SCHEMA",
        )

    doc_type = {
        doc.metadata.get("source_file"): doc.metadata.get("doc_type")
        for doc in chunks_top_k
    }

    for fonte in resposta_llm.sources_used:
        fonte.doc_type = doc_type.get(fonte.filepath, fonte.doc_type)

    return RAGResponse(
        answer=resposta_llm.answer,
        confidence_level=nivel_confianca,
        sources_used=resposta_llm.sources_used,
        reasoning=resposta_llm.reasoning,
        is_refusal=False,
        refusal_reason=None
    )
