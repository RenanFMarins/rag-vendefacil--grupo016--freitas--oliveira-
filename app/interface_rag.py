import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _environment_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        return default


api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
timeout = _environment_int("OPENAI_TIMEOUT", 120)
max_retries = _environment_int("OPENAI_MAX_RETRIES", 2)
embedding_model = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)
web_model = os.getenv("OPENAI_WEB_MODEL", "gpt-5.4-mini")
index_path = PROJECT_ROOT / "storage" / "faiss_markdown_demo"


def cosine_relevance_score(squared_l2_distance: float) -> float:
    """Converte a distância L2 quadrática do FAISS em similaridade cosseno."""
    return max(0.0, min(1.0, 1.0 - squared_l2_distance / 2.0))


@st.cache_resource(show_spinner=False)
def load_vectorstore(index_directory: str, model: str, key: str):
    """Carrega o índice local sem vetorizar novamente o Markdown."""
    embeddings = OpenAIEmbeddings(model=model, api_key=key)
    return FAISS.load_local(
        index_directory,
        embeddings,
        allow_dangerous_deserialization=True,
        relevance_score_fn=cosine_relevance_score,
    )


def _web_sources(response) -> list[dict[str, str]]:
    """Extrai e remove duplicatas das citações retornadas pela busca web."""
    response_data = response.model_dump()
    sources_by_url: dict[str, dict[str, str]] = {}

    for item in response_data.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                for annotation in content.get("annotations", []):
                    if annotation.get("type") != "url_citation":
                        continue
                    citation = annotation.get("url_citation", annotation)
                    url = citation.get("url", "")
                    if url.startswith(("https://", "http://")):
                        sources_by_url[url] = {
                            "source_type": "web",
                            "title": citation.get("title") or url,
                            "url": url,
                        }

        if item.get("type") == "web_search_call":
            action = item.get("action") or {}
            for source in action.get("sources", []):
                url = source.get("url", "")
                if url.startswith(("https://", "http://")):
                    sources_by_url.setdefault(
                        url,
                        {
                            "source_type": "web",
                            "title": source.get("title") or url,
                            "url": url,
                        },
                    )

    return list(sources_by_url.values())


def search_web(question: str):
    """Pesquisa a web com a Responses API e devolve resposta e fontes."""
    client = OpenAI(
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )
    response = client.responses.create(
        model=web_model,
        tools=[{"type": "web_search", "search_context_size": "low"}],
        tool_choice="required",
        include=["web_search_call.action.sources"],
        input=[
            {
                "role": "system",
                "content": (
                    "Responda em português com base na busca na internet. "
                    "Diferencie fatos de incertezas e preserve as citações "
                    "das fontes usadas."
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    return response.output_text, _web_sources(response)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "test_records" not in st.session_state:
    st.session_state.test_records = []

st.title("💬 RAG com Markdown e Memória")
st.caption(
    f"Chat: {model_name} | Embeddings: {embedding_model} | Base: manual_pdv.md"
)

top_k = st.slider(
    "Número de documentos a recuperar (top_k):", 1, 10, 3
)
score_threshold = st.slider(
    "Limite mínimo de similaridade (score_threshold):",
    0.0,
    1.0,
    0.5,
    0.05,
)
fonte = st.text_input(
    "Filtrar por arquivo de origem (opcional):",
    placeholder="Ex.: manual_pdv.md",
)
web_fallback = st.toggle(
    "Buscar na internet quando a base não tiver a resposta",
    value=False,
    help=(
        "Quando ativado, perguntas sem resultado local são enviadas "
        "para a busca web da OpenAI e podem gerar custo adicional."
    ),
)

status_column, clear_column = st.columns([3, 1])
interaction_count = sum(
    isinstance(message, HumanMessage)
    for message in st.session_state.chat_history
)
status_column.metric("Interações armazenadas", interaction_count)

if clear_column.button("Limpar histórico"):
    st.session_state.chat_history = []
    st.session_state.test_records = []
    st.rerun()

for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(message.content)
        sources = message.additional_kwargs.get("sources", [])
        if sources:
            with st.expander("Fontes recuperadas"):
                for source in sources:
                    if source.get("source_type") == "web":
                        st.markdown(
                            f"- [{source['title']}]({source['url']}) "
                            "— fonte externa"
                        )
                    else:
                        st.markdown(
                            f"- `{source['source_file']}` — "
                            f"seção: {source['section']}"
                        )

if not index_path.exists():
    st.warning(
        "A base vetorial de demonstração ainda não foi criada. "
        "Você pode criá-la com "
        "`.venv/bin/python -m ingestion.build_markdown_demo_index` ou "
        "ativar a busca na internet."
    )

user_input = st.text_input("Digite sua pergunta:")
send = st.button("Enviar")

if send:
    if not api_key:
        st.error(
            "A variável OPENAI_API_KEY não foi configurada. "
            "Copie .env.example para .env e informe uma nova chave."
        )
        st.stop()

    if not user_input.strip():
        st.warning("Digite uma pergunta antes de enviar.")
        st.stop()

    if not index_path.exists() and not web_fallback:
        st.error(
            "Crie a base vetorial ou ative a busca na internet antes de enviar."
        )
        st.stop()

    try:
        with st.spinner("Recuperando trechos e gerando a resposta..."):
            documents = []
            if index_path.exists():
                vectorstore = load_vectorstore(
                    str(index_path), embedding_model, api_key
                )
                search_kwargs = {
                    "k": top_k,
                    "score_threshold": score_threshold,
                }
                if fonte.strip():
                    search_kwargs["filter"] = {
                        "source_file": fonte.strip()
                    }

                retriever = vectorstore.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs=search_kwargs,
                )
                documents = retriever.invoke(user_input)

            if documents:
                llm = ChatOpenAI(
                    model=model_name,
                    api_key=api_key,
                    timeout=timeout,
                    max_retries=max_retries,
                )
                context = "\n\n".join(
                    f"Fonte {document.metadata.get('source_file', 'desconhecida')}\n"
                    f"{document.page_content}"
                    for document in documents
                )
                system_message = SystemMessage(
                    content=(
                        "Responda em português usando somente o contexto "
                        "recuperado abaixo. Se o contexto não contiver a "
                        "resposta, diga que não encontrou essa informação.\n\n"
                        f"CONTEXTO:\n{context}"
                    )
                )
                messages = [
                    system_message,
                    *st.session_state.chat_history,
                    HumanMessage(content=user_input),
                ]
                response = llm.invoke(messages)
                answer = response.content
                answer_origin = "Base interna"
                sources = [
                    {
                        "source_type": "internal",
                        "source_file": document.metadata.get(
                            "source_file", "desconhecida"
                        ),
                        "section": document.metadata.get(
                            "subsection",
                            document.metadata.get("section", "sem seção"),
                        ),
                    }
                    for document in documents
                ]
            elif web_fallback:
                answer, sources = search_web(user_input)
                answer_origin = "Busca externa"
            else:
                answer = (
                    "Não encontrei trechos que atendam ao limite de "
                    "similaridade e ao filtro informados."
                )
                sources = []
                answer_origin = "Sem resultado"
    except Exception:
        st.error(
            "Não foi possível consultar a base ou a OpenAI. Verifique "
            "a chave, os modelos configurados e se o índice foi criado."
        )
    else:
        st.session_state.chat_history.append(
            HumanMessage(content=user_input)
        )
        st.session_state.chat_history.append(
            AIMessage(content=answer, additional_kwargs={"sources": sources})
        )
        st.session_state.test_records.append(
            {
                "Pergunta": user_input,
                "top_k": top_k,
                "score_threshold": score_threshold,
                "Origem": answer_origin,
                "Fontes retornadas": ", ".join(
                    source.get("source_file", source.get("title", "desconhecida"))
                    for source in sources
                ) or "Nenhuma",
                "Clareza e relevância": "Preencha sua observação",
            }
        )
        st.rerun()

st.subheader("Registro dos testes")
st.caption(
    "Sugestões: Como fazer uma sangria?; O que fazer quando a SEFAZ "
    "está indisponível?; Qual desconto exige autorização?"
)
edited_records = st.data_editor(
    st.session_state.test_records,
    num_rows="dynamic",
    width="stretch",
)
st.session_state.test_records = (
    edited_records.to_dict("records")
    if hasattr(edited_records, "to_dict")
    else edited_records
)
