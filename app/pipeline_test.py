"""
Adapter entre o benchmark (Etapa 4) e o pipeline RAG (pipeline_rag.py).

Isto é o único arquivo que muda quando o pipeline real evolui.

`answer_question(pergunta) -> dict` é chamada pelo app.py e pelo
run_benchmark.py. Por baixo, ela chama a sua `pipeline_rag()` de verdade
(ver `_real_answer_question` abaixo) e cai automaticamente pro mock se algo
falhar (ex: filtros ainda instáveis, import quebrado).

Formato de retorno esperado por quem chama `answer_question`:
{
    "answer": str,
    "is_refusal": bool,
    "confidence_level": "alta" | "media" | "baixa",
    "citations": [{"source": ..., "chunk_id": ...}, ...],
    "retrieved_chunks": [{"source": ..., "chunk_id": ..., "text": ...}, ...],
}
"""

import time

# ---------------------------------------------------------------------------
# Integração real (pipeline_rag.py) — com fallback pro mock se algo falhar.
#
# ASSUNÇÕES (confirmar / ajustar quando o schema real chegar):
# - existe algum jeito de carregar `vocabulario` e `documentos` uma única vez
#   (ETAPA1_CARREGAR abaixo) — troque pelos imports reais da Etapa 1.
# - `RAGResponse.sources_used` é uma lista de dicts com pelo menos "source"
#   (e idealmente "chunk_id"). Se vier como lista de strings, o código abaixo
#   também lida com isso.
# - `confidence_level` vem como "Alto"/"Médio"/"Baixo"/"Recusado" (PT,
#   capitalizado) — normalizado para "alta"/"media"/"baixa" abaixo.
# - `pipeline_rag()` não expõe os chunks recuperados antes da geração (só o
#   `sources_used` final), então por enquanto `retrieved_chunks` é aproximado
#   pelas mesmas fontes de `sources_used`. Isso faz Context Relevance e
#   citation_score ficarem correlacionados — não ideal, mas funcional até a
#   Etapa 3 expor os chunks intermediários (ideal: pipeline_rag retornar
#   também `chunks_recuperados` além de `sources_used`).
# ---------------------------------------------------------------------------

_REAL_PIPELINE_AVAILABLE = True
try:
    from src.query.pipeline_rag import pipeline_rag  # seu arquivo, na raiz do projeto

    # TODO: confirme/ajuste os caminhos de import reais (Etapa 1/2)
    from src.carregar_banco import abrir_banco          # <-- ajuste o módulo aqui
    from src.retrieval.filters import extract_metadata  # <-- ajuste o módulo aqui

    def _carregar_vocabulario_e_documentos():
        db = abrir_banco()
        documentos = list(db.docstore._dict.values())
        vocabulario = extract_metadata(documentos)
        return vocabulario, documentos

    _vocabulario = None
    _documentos = None
except Exception:  # noqa: BLE001 - se a Etapa 3 não importar, cai pro mock
    _REAL_PIPELINE_AVAILABLE = False


def _normalize_confidence(raw: str) -> str:
    m = {
        "alto": "alta", "alta": "alta",
        "médio": "media", "medio": "media", "media": "media",
        "baixo": "baixa", "baixa": "baixa",
        "recusado": "baixa",
    }
    return m.get((raw or "").strip().lower(), "baixa")


def _extract_source_chunk(item) -> dict:
    """Normaliza um item de sources_used para {"source","chunk_id"}.
    Lida com dict, string, ou objeto tipo SourceEvidence(filepath, quotation, doc_type)."""
    if isinstance(item, dict):
        return {
            "source": item.get("source") or item.get("path") or item.get("filepath") or item.get("doc") or "",
            "chunk_id": item.get("chunk_id") or item.get("id") or "",
        }
    if isinstance(item, str):
        return {"source": item, "chunk_id": ""}
    # objeto (ex: SourceEvidence com .filepath / .quotation / .doc_type)
    return {
        "source": (
            getattr(item, "source", "")
            or getattr(item, "path", "")
            or getattr(item, "filepath", "")
            or ""
        ),
        "chunk_id": getattr(item, "chunk_id", "") or getattr(item, "id", "") or "",
    }


def _extract_chunk_text_source_id(item) -> dict:
    """Normaliza um item de chunks_top_k (tupla (doc, score) ou objeto Document)
    para {"source", "chunk_id", "text"}."""
    doc = item[0] if isinstance(item, (tuple, list)) else item
    metadata = getattr(doc, "metadata", {}) or {}
    return {
        "source": metadata.get("source") or metadata.get("filepath") or metadata.get("path") or "",
        "chunk_id": metadata.get("chunk_id") or metadata.get("id") or "",
        "text": getattr(doc, "page_content", "") or getattr(doc, "text", "") or "",
    }


def _real_answer_question(question: str) -> dict:
    global _vocabulario, _documentos
    if _vocabulario is None or _documentos is None:
        _vocabulario, _documentos = _carregar_vocabulario_e_documentos()

    resp = pipeline_rag(question, _vocabulario, _documentos)
    sources_used = getattr(resp, "sources_used", []) or []
    citations = [_extract_source_chunk(s) for s in sources_used]

    chunks_utilizados = getattr(resp, "chunks_utilizados", None)
    if chunks_utilizados:
        retrieved_chunks = [_extract_chunk_text_source_id(
            c) for c in chunks_utilizados]
    else:
        # fallback se pipeline_rag.py ainda não expõe chunks_utilizados
        retrieved_chunks = citations

    return {
        "answer": getattr(resp, "answer", ""),
        "is_refusal": bool(getattr(resp, "is_refusal", False)),
        "confidence_level": _normalize_confidence(getattr(resp, "confidence_level", "")),
        "citations": citations,
        "retrieved_chunks": retrieved_chunks,
    }


# Palavras-gatilho usadas só pelo MOCK abaixo para simular recusas de forma
# minimamente plausível enquanto o pipeline real não está plugado.
_MOCK_REFUSAL_KEYWORDS = [
    "salário", "senha", "chave de api", "chave secreta", "credencial",
    "jwt", "postgresql", "smtp",
]


def _mock_answer_question(question: str) -> dict:
    """MOCK — usado apenas até você plugar o pipeline real (ver docstring acima)."""
    q_lower = question.lower()
    is_refusal = any(k in q_lower for k in _MOCK_REFUSAL_KEYWORDS)
    return {
        "answer": (
            "[MOCK] Recusa simulada por guardrail de segurança/LGPD."
            if is_refusal
            else "[MOCK] Resposta simulada — conecte o pipeline real em eval/pipeline_adapter.py."
        ),
        "is_refusal": is_refusal,
        "confidence_level": "baixa",
        "citations": [],
        "retrieved_chunks": [],
    }


def answer_question(question: str) -> dict:
    if _REAL_PIPELINE_AVAILABLE:
        try:
            result = _real_answer_question(question)
        except Exception as e:  # noqa: BLE001
            # Etapa 3 ainda instável: não deixa a demo quebrar, mas avisa no terminal.
            print(
                f"[pipeline_adapter] pipeline real falhou ({e}); usando MOCK.")
            result = _mock_answer_question(question)
    else:
        result = _mock_answer_question(question)
    result.setdefault("_latency_s", None)
    return result


def timed_answer_question(question: str) -> dict:
    """Wrapper que mede latência — usado pelo run_benchmark.py."""
    t0 = time.time()
    result = answer_question(question)
    result["_latency_s"] = round(time.time() - t0, 3)
    return result
