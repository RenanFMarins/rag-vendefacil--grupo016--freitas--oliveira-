# Busca híbrida e filtragem por metadados

## Fluxo implementado

```text
Pergunta
  -> Query Analyzer (structured output com Pydantic)
  -> query semântica + filtros extraídos
  -> normalização dos filtros
  -> validação contra o catálogo dinâmico do corpus
  -> FAISS Dense Search + BM25 Sparse Search
  -> Reciprocal Rank Fusion por chunk_id
  -> Top-K final
```

Os filtros aceitos são aplicados igualmente aos dois retrievers. O FAISS usa
uma estratégia adaptativa de `fetch_k`, com pré-filtragem exata em casos muito
seletivos. O BM25 filtra os `Documents` elegíveis antes de ordenar os matches
lexicais.

O RRF não compara a distância FAISS com o score BM25. Para cada chunk, ele
soma `1 / (rank_constant + rank)` de cada ranking em que o `chunk_id` aparece.
O valor padrão de `rank_constant` é 60. Empates são resolvidos de maneira
determinística.

## Execução

```bash
.venv/bin/python -m src.inspection.hybrid_retrieval --top-k 5 --candidate-k 20
```

Para testar uma pergunta específica:

```bash
.venv/bin/python -m src.inspection.hybrid_retrieval \
  --query "Quais tickets de MG estão abertos?" \
  --top-k 5 \
  --candidate-k 20
```

O comando imprime a pergunta, a query semântica, filtros válidos e rejeitados,
quantidade de candidatos de cada retriever e o Top-K após o RRF.

## Testes

```bash
.venv/bin/python -m pytest \
  tests/retrieval/test_rrf.py \
  tests/retrieval/test_hybrid_retrieval.py \
  tests/retrieval/test_dense_search.py \
  tests/retrieval/test_sparse_search.py \
  tests/retrieval/test_query_analyzer.py \
  tests/retrieval/test_filter_normalization.py \
  tests/retrieval/test_filter_validation.py \
  tests/retrieval/test_metadata_catalog.py -q
```
