import json
from pathlib import Path

import streamlit as st
import sys
from app.pipeline_test import timed_answer_question

sys.path.insert(0, str(Path(__file__).resolve().parent / "eval"))
# from pipeline_adapter import timed_answer_question  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "eval" / "results.json"

st.set_page_config(
    page_title="VendeFácil RAG - Demo & Benchmark", layout="wide")
st.title("VendeFácil RAG — Demo & Avaliação (Etapa 4)")

tab_demo, tab_bench = st.tabs(["🔎 Demo", "📊 Benchmark"])


with tab_demo:
    st.subheader("Faça uma pergunta ao sistema")
    question = st.text_input(
        "Pergunta", placeholder="Ex: Qual é a política de home ofice da Engenharia?")
    run_btn = st.button("Perguntar", type="primary")

    if run_btn and question.strip():
        with st.spinner("Consultando o pipeline..."):
            result = timed_answer_question(question)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### Resposta")
            if result.get("is_refusal"):
                st.warning(result["answer"])
            else:
                st.write(result["answer"])

        with col2:
            st.markdown("### Metadados")
            st.metric("Recusou?", "Sim" if result.get("is_refusal") else "Não")
            st.metric("Confiança", result.get("confidence_level", "—"))
            if result.get("_latency_s") is not None:
                st.metric("Latência", f"{result['_latency_s']}s")

        st.markdown("### Citações")
        citations = result.get("citations", [])
        if citations:
            for c in citations:
                st.code(
                    f"{c.get('source', '?')}  (chunk: {c.get('chunk_id', '—')})")
        else:
            st.caption("Nenhuma citação retornada.")

        with st.expander("Chunks recuperados (debug)"):
            chunks = result.get("retrieved_chunks", [])
            if chunks:
                for c in chunks:
                    st.markdown(
                        f"**{c.get('source', '?')}** (`{c.get('chunk_id', '—')}`)")
                    st.text(c.get("text", "")[:500])
                    st.divider()
            else:
                st.caption("Nenhum chunk retornado pelo adapter.")
    elif run_btn:
        st.info("Digite uma pergunta primeiro.")


# ======================================================================================
with tab_bench:
    st.subheader("Resultados do benchmark")

    if not RESULTS_PATH.exists():
        st.error(
            "eval/results.json não encontrado. Rode primeiro:\n\n"
            "```\nexport GROQ_API_KEY='sua_chave'\npython eval/run_benchmark.py\n```"
        )
    else:
        data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        results = data["results"]
        n = len(results)
        total = sum(r["score_total"] for r in results)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Perguntas", n)
        c2.metric("Pontuação total",
                  f"{total:.2f} / {n:.1f}", f"{100*total/n:.1f}%")

        def avg(vals):
            vals = [v for v in vals if v is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        ctx_avg = avg([r["context_relevance"] for r in results])
        ans_avg = avg([r["answer_relevance"]["score"]
                      for r in results if r.get("answer_relevance")])
        gnd_avg = avg([r["groundedness"]["score"]
                      for r in results if r.get("groundedness")])

        c3.metric("Context Relevance", ctx_avg if ctx_avg is not None else "—")
        c4.metric("Answer Rel. / Groundedness",
                  f"{ans_avg or '—'} / {gnd_avg or '—'}")

        st.markdown("---")
        st.markdown("### Pontuação por categoria")

        import pandas as pd

        by_cat = {}
        for r in results:
            by_cat.setdefault(r["category"], []).append(r)
        rows = []
        for cat, items in by_cat.items():
            rows.append({
                "Categoria": cat,
                "Qtde": len(items),
                "Score médio": avg([i["score_total"] for i in items]),
                "Context Relevance médio": avg([i["context_relevance"] for i in items]),
            })
        st.dataframe(pd.DataFrame(rows),
                     use_container_width=True, hide_index=True)

        st.markdown("### Todas as questões")
        table_rows = []
        for r in results:
            table_rows.append({
                "ID": r["id"],
                "Categoria": r["category"],
                "Score": r["score_total"],
                "Recusou?": "Sim" if r["is_refusal"] else "Não",
                "Confiança": r["confidence_level"],
                "Context Rel.": r["context_relevance"],
                "Answer Rel.": r["answer_relevance"]["score"] if r.get("answer_relevance") else None,
                "Groundedness": r["groundedness"]["score"] if r.get("groundedness") else None,
            })
        df = pd.DataFrame(table_rows).sort_values("Score")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("### Detalhe de uma questão")
        selected_id = st.selectbox("Escolha a questão", [
                                   r["id"] for r in results])
        r = next(r for r in results if r["id"] == selected_id)
        st.json(r)
