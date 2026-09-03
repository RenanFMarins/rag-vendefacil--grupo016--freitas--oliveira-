# Avaliação RAG Triad

Este projeto calcula separadamente as três dimensões da RAG Triad. Os
scores variam de `0.0` a `1.0` e não substituem os checks determinísticos do
benchmark.

## Fluxo da avaliação

```text
pergunta
  -> pipeline RAG
  -> RAGResponse validada
  -> trace seguro do Top-K
  -> Context Relevance determinística
  -> LLM-as-a-Judge opcional
       -> Answer Relevance
       -> Groundedness
  -> JSON de resultado sem respostas ou quotations
```

## Context Relevance

Mede se o Top-K recuperado contém o contexto esperado. O pipeline expõe para
o avaliador somente `chunk_id`, `source_file` e `rank`; `page_content` não faz
parte do trace persistido.

A base da medição é registrada em `context_relevance_basis`:

- `chunk_id`: recall dos IDs esperados, quando o gabarito os fornece;
- `source_file`: fallback por arquivo, quando não existem IDs esperados;
- `not_applicable`: perguntas cuja resposta esperada é uma recusa.

O benchmark oficial atual informa arquivos, mas não informa `chunk_id` de
gabarito. Por isso, suas execuções usam `source_file`. PDF e Markdown com o
mesmo nome-base são tratados como representações alternativas do mesmo
documento.

## Answer Relevance

Um LLM juiz compara pergunta, resposta, comportamento esperado, resposta de
referência e pontos principais do gabarito. A saída é validada pelo schema
Pydantic `TriadJudgeDraft` e deve conter uma nota entre `0` e `1`.

## Groundedness

O mesmo juiz verifica se as afirmações factuais da resposta estão apoiadas
nas quotations construídas deterministicamente pelo pipeline. O gabarito
não é apresentado como evidência.

Groundedness não é calculada para respostas recusadas, porque elas não
apresentam uma resposta factual fundamentada no corpus.

## Segurança e custo

O juiz é opcional e exige uma chamada adicional de LLM por pergunta
avaliável. Perguntas cuja resposta esperada é uma recusa LGPD ou de escopo
não são enviadas ao juiz. Elas continuam sendo verificadas pelos checks
determinísticos.

As justificativas textuais do juiz são validadas durante a execução, mas não
são persistidas. O resultado armazena apenas scores, status do juiz e
identificadores seguros do Top-K.

O structured output possui duas tentativas. Se ambas falharem, o caso mantém
sua avaliação determinística, `judge_status` recebe `failed` e os dois scores
qualitativos ficam sem valor. Não existe loop infinito.

## Execução

Validar o arquivo oficial sem chamar modelos:

```bash
.venv/bin/python -m eval.run_benchmark --validate-only
```

Executar somente as métricas determinísticas e Context Relevance:

```bash
.venv/bin/python -m eval.run_benchmark
```

Executar a RAG Triad completa:

```bash
.venv/bin/python -m eval.run_benchmark --with-judge
```

Fazer um smoke test de apenas uma pergunta:

```bash
.venv/bin/python -m eval.run_benchmark --with-judge --limit 1
```

O modelo pode ser configurado no `.env`:

```text
OPENAI_JUDGE_MODEL=gpt-4o-mini
```

## Interpretação

O resumo possui uma seção `rag_triad` com a média e a quantidade de casos
efetivamente avaliados. Não foi definido um corte arbitrário para transformar
os scores do juiz em `passed` ou `failed`; o status funcional continua sendo
determinado pelos checks objetivos do benchmark.

O arquivo oficial recebido possui 24 perguntas, apesar de o guia mencionar
20. Ele também não marca nenhuma pergunta como `masked_answer`, embora o guia
mencione um caso de mascaramento. O projeto preserva o arquivo oficial e
reporta essa divergência em vez de alterar o gabarito silenciosamente.

