import sys
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.bootstrap import RetrievalRuntime  # noqa: E402
from app.bootstrap import build_retrieval_runtime  # noqa: E402
from src.retrieval.pipeline import HybridRetrievalResponse  # noqa: E402


ALLOWED_SENSITIVITIES = frozenset({"publico", "interno"})


@st.cache_resource(show_spinner=False)
def load_runtime(top_k: int, candidate_k: int) -> RetrievalRuntime:
    return build_retrieval_runtime(
        top_k=top_k,
        candidate_k=candidate_k,
    )


def serialize_response(response: HybridRetrievalResponse) -> dict[str, Any]:
    return {
        "question": response.question,
        "semantic_query": response.analysis.query,
        "filters": response.analysis.filters.model_dump(exclude_none=True),
        "rejected_filters": [
            item.model_dump(exclude_none=True)
            for item in response.analysis.rejected_filters
        ],
        "dense_candidates": len(response.dense_response.results),
        "sparse_candidates": len(response.sparse_results),
        "diagnostics": [
            {
                "index": item.index_name,
                "strategy": item.strategy,
                "eligible": item.eligible_documents,
                "total": item.total_documents,
                "fetch_k": item.fetch_k,
            }
            for item in response.dense_response.diagnostics
        ],
        "results": [
            {
                "rank": item.rank,
                "chunk_id": item.chunk_id,
                "source_file": item.document.metadata.get(
                    "source_file", "ausente"
                ),
                "doc_type": item.document.metadata.get(
                    "doc_type", "ausente"
                ),
                "sensitivity": item.document.metadata.get(
                    "sensitivity", "ausente"
                ),
                "section": item.document.metadata.get("section"),
                "score": item.score,
                "dense_rank": item.dense_rank,
                "bm25_rank": item.sparse_rank,
                "retrievers": item.matched_retrievers,
                "content": item.document.page_content,
            }
            for item in response.results
        ],
    }


def _render_record(record: dict[str, Any], show_debug: bool) -> None:
    with st.chat_message("user"):
        st.write(record["question"])

    with st.chat_message("assistant"):
        result_count = len(record["results"])
        st.write(
            f"Recuperei {result_count} chunk(s) com FAISS + BM25 + RRF. "
            "Esta tela ainda não gera uma resposta textual: ela mostra "
            "somente a saída da Etapa 2."
        )
        st.caption(f"Query semântica: {record['semantic_query']}")
        st.write("Filtros validados:", record["filters"] or "nenhum")

        if record["rejected_filters"]:
            st.warning(
                f"Filtros rejeitados: {record['rejected_filters']}"
            )

        for result in record["results"]:
            title = (
                f"{result['rank']}. {result['source_file']} "
                f"— chunk {result['chunk_id']}"
            )
            with st.expander(title, expanded=result["rank"] == 1):
                st.write(result["content"])
                st.caption(
                    f"doc_type={result['doc_type']} | "
                    f"sensitivity={result['sensitivity']} | "
                    f"RRF={result['score']:.6f} | "
                    f"dense_rank={result['dense_rank']} | "
                    f"bm25_rank={result['bm25_rank']}"
                )
                if result["section"]:
                    st.caption(f"Seção: {result['section']}")

        if show_debug:
            with st.expander("Diagnóstico do retrieval"):
                st.json(
                    {
                        "dense_candidates": record["dense_candidates"],
                        "bm25_candidates": record["sparse_candidates"],
                        "dense_diagnostics": record["diagnostics"],
                    }
                )


def main() -> None:
    st.set_page_config(
        page_title="Retrieval VendeFácil",
        page_icon="🔎",
        layout="wide",
    )
    st.title("🔎 Retrieval híbrido VendeFácil")
    st.caption(
        "Interface da Etapa 2: Query Analyzer → normalização → "
        "validação → FAISS + BM25 → RRF → Top-K."
    )
    st.info(
        "Os chunks restritos ficam excluídos. A geração textual e os "
        "guardrails completos serão conectados com a Etapa 3."
    )

    if "retrieval_history" not in st.session_state:
        st.session_state.retrieval_history = []

    with st.sidebar:
        st.header("Configuração")
        top_k = st.slider("Top-K final", 1, 10, 5)
        candidate_k = st.slider(
            "Candidatos por retriever",
            min_value=top_k,
            max_value=50,
            value=max(20, top_k),
        )
        show_debug = st.toggle("Exibir diagnóstico", value=False)
        st.write(
            "Sensibilidades permitidas: `publico`, `interno`. "
            "A categoria `restrito` não é exibida."
        )
        if st.button("Limpar histórico", use_container_width=True):
            st.session_state.retrieval_history = []
            st.rerun()

    for record in st.session_state.retrieval_history:
        _render_record(record, show_debug)

    question = st.chat_input("Digite uma pergunta sobre a VendeFácil")
    if not question:
        return

    try:
        with st.spinner("Analisando e recuperando chunks..."):
            runtime = load_runtime(top_k, candidate_k)
            response = runtime.retriever.retrieve(
                question,
                debug=show_debug,
                allowed_sensitivities=ALLOWED_SENSITIVITIES,
            )
            record = serialize_response(response)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        st.error(str(error))
        return
    except Exception:
        st.error(
            "Não foi possível executar a busca. Verifique a chave da "
            "OpenAI, a conexão e os seis índices persistidos."
        )
        return

    st.session_state.retrieval_history.append(record)
    st.rerun()


if __name__ == "__main__":
    main()
