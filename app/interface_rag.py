"""Interface Streamlit para o pipeline RAG estruturado do projeto."""

import logging
import sys
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.bootstrap import build_rag_pipeline  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402
from starter.schema import RAGResponse  # noqa: E402


LOGGER = logging.getLogger(__name__)
REFUSAL_LABELS = {
    "lgpd": (
        "A solicitação envolve informação protegida pela política "
        "de privacidade."
    ),
    "fora_de_escopo": (
        "A pergunta está fora do escopo corporativo da VendeFácil."
    ),
    "sem_evidencia": "Não há evidência suficiente no corpus autorizado.",
}
FRIENDLY_ERROR = (
    "Não foi possível processar a pergunta neste momento. "
    "Verifique a configuração da aplicação e tente novamente."
)


@st.cache_resource(show_spinner=False)
def load_pipeline(top_k: int, candidate_k: int) -> RAGPipeline:
    """Constrói e reutiliza o pipeline durante os reruns do Streamlit."""
    return build_rag_pipeline(top_k=top_k, candidate_k=candidate_k)


def response_view_model(response: RAGResponse) -> dict[str, Any]:
    """Prepara rótulos de apresentação sem alterar o contrato."""
    validated = RAGResponse.model_validate(response)
    return {
        "answer": validated.answer,
        "confidence_level": validated.confidence_level,
        "sources_used": [
            source.model_dump() for source in validated.sources_used
        ],
        "reasoning": validated.reasoning,
        "is_refusal": validated.is_refusal,
        "refusal_reason": validated.refusal_reason,
        "refusal_label": REFUSAL_LABELS.get(validated.refusal_reason),
    }


def ask_pipeline(
    pipeline: RAGPipeline,
    question: str,
    *,
    debug: bool = False,
) -> tuple[RAGResponse | None, str | None]:
    """Executa o pipeline e converte falhas técnicas em erro seguro."""
    try:
        return pipeline.answer(question, debug=debug), None
    except Exception as error:  # fronteira segura da aplicação
        LOGGER.error(
            "Falha controlada na consulta do pipeline: %s",
            type(error).__name__,
        )
        return None, FRIENDLY_ERROR


def render_response(response: RAGResponse) -> None:
    """Renderiza resposta, recusa e evidências já validadas."""
    view = response_view_model(response)
    if view["is_refusal"]:
        st.warning(view["answer"])
        if view["refusal_label"]:
            st.caption(view["refusal_label"])
    else:
        st.write(view["answer"])

    st.caption(f"Confiança: {view['confidence_level']}")
    with st.expander("Fundamentação", expanded=False):
        st.write(view["reasoning"])

    if view["sources_used"]:
        st.markdown("**Evidências utilizadas**")
        for position, source in enumerate(view["sources_used"], start=1):
            with st.expander(
                f"Fonte {position}: {source['filepath']}",
                expanded=False,
            ):
                st.caption(f"Chunk: {source['chunk_id']}")
                st.write(source["quotation"])


def _initialize_session() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def _render_history() -> None:
    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            render_response(RAGResponse.model_validate(item["response"]))


def main() -> None:
    st.set_page_config(
        page_title="Assistente VendeFácil",
        page_icon="💬",
        layout="wide",
    )
    _initialize_session()
    st.title("💬 Assistente corporativo VendeFácil")
    st.caption(
        "Respostas estruturadas a partir dos índices CSV, JSON, JSONL, "
        "Markdown, PDF e TXT. O assistente não consulta a internet."
    )

    with st.sidebar:
        st.header("Configuração")
        top_k = st.slider("Top-K final", 1, 10, 5)
        candidate_k = st.slider(
            "Candidatos por retriever",
            min_value=top_k,
            max_value=50,
            value=max(20, top_k),
        )
        debug = st.toggle("Logs seguros de debug", value=False)
        st.caption(
            "O pipeline aplica escopo, LGPD, retrieval híbrido, "
            "geração fundamentada e validação Pydantic."
        )

    header, clear_column = st.columns([3, 1])
    header.metric("Interações na sessão", len(st.session_state.chat_history))
    if clear_column.button("Limpar histórico"):
        st.session_state.chat_history = []
        st.rerun()

    _render_history()
    question = st.chat_input("Digite sua pergunta sobre a VendeFácil")
    if not question:
        return

    with st.chat_message("user"):
        st.write(question)

    try:
        pipeline = load_pipeline(top_k, candidate_k)
    except (FileNotFoundError, RuntimeError, ValueError):
        LOGGER.error("Falha controlada ao inicializar o pipeline.")
        st.error(
            "A base corporativa não está pronta. Confira os seis índices "
            "e a configuração da OPENAI_API_KEY."
        )
        return
    except Exception as error:
        LOGGER.error(
            "Falha controlada no bootstrap: %s", type(error).__name__
        )
        st.error(FRIENDLY_ERROR)
        return

    with st.chat_message("assistant"):
        with st.spinner("Consultando o corpus autorizado..."):
            response, error_message = ask_pipeline(
                pipeline,
                question,
                debug=debug,
            )
        if error_message:
            st.error(error_message)
            return
        assert response is not None
        render_response(response)

    st.session_state.chat_history.append(
        {
            "question": question,
            "response": response.model_dump(mode="json"),
        }
    )


if __name__ == "__main__":
    main()
