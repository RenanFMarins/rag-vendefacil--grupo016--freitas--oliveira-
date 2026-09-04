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

## Comparativo com e sem filtro

O critério de pronto foi validado em 2026-09-03 com os seis índices e 5.723
chunks. O comando usado foi:

```bash
.venv/bin/python -m src.inspection.dense_search --k 5 --fetch-k 20
```

| Pergunta | Filtros validados | Sem filtro | Com filtro |
| --- | --- | --- | --- |
| Tickets de MG relacionados a estoque | `doc_type=ticket`, `state=MG`, `module=estoque` | Trouxe 2 tickets compatíveis e 3 logs sem estado | Retornou somente os 4 tickets MG/estoque existentes |
| Tickets de SP relacionados ao módulo Pay | `doc_type=ticket`, `state=SP`, `module=pay` | Priorizou loja e vendas de SP/Pay | Retornou somente `TCK-1005` |
| Informações sobre o módulo de estoque | `module=estoque` | Os 5 primeiros já eram compatíveis | Preservou os 5 e restringiu todos os candidatos ao módulo |

Nos dois filtros altamente seletivos, o Dense Search escolheu
`exact_prefilter`. No terceiro, combinou pré-filtragem exata para conjuntos
pequenos com `adaptive_postfilter` e fallback exato nos demais índices. Isso
evita o retorno vazio causado pelo `fetch_k=20` fixo do FAISS.

## Comparativo Dense e BM25

```bash
.venv/bin/python -m src.inspection.sparse_search --k 5
```

| Consulta | Dense/FAISS | BM25 | Melhor comportamento |
| --- | --- | --- | --- |
| `TCK-1005` | O ticket exato apareceu na 4ª posição | O ticket exato apareceu na 1ª posição | BM25, por preservar o código literal |
| Como recolher periodicamente o dinheiro acumulado pelo atendente? | O manual do PDV apareceu na 1ª posição | Priorizou e-mail e logs por coincidência lexical | Dense, por reconhecer a paráfrase de sangria |

Esses casos justificam o uso do RRF: nenhum dos dois retrievers é superior em
todas as consultas.

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
