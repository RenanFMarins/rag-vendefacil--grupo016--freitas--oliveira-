# Debug do projeto no VS Code

## Preparação

1. Abra no VS Code a raiz `vende-facil`, não apenas `app/` ou `src/`.
2. Instale as extensões recomendadas pelo projeto:
   - Python (`ms-python.python`);
   - Python Debugger (`ms-python.debugpy`).
3. Execute `Python: Select Interpreter` na paleta de comandos e confirme
   `.venv/bin/python`.
4. Abra **Run and Debug** com `Ctrl+Shift+D`.

As configurações ficam em `.vscode/launch.json`. Como `.vscode/` está no
`.gitignore`, elas são locais e não aparecem nos commits.

## Configurações disponíveis

| Configuração | Uso |
| --- | --- |
| `Etapa 1: loader + preview` | Lê e transforma um dos seis formatos sem indexar |
| `Etapa 1: builder sem reindexar` | Valida conteúdo, metadata e unicidade dos chunks |
| `Etapa 1: indexar JSONL em área de debug` | Gera um FAISS de estudo sem sobrescrever o oficial |
| `Etapa 1: recarga e sanidade FAISS` | Recarrega os seis índices e executa três buscas |
| `RAG: Streamlit` | Interface completa na porta 8502 |
| `RAG: pipeline completo` | Uma pergunta passando por todas as etapas |
| `RAG: somente Query Analyzer` | Query semântica, filtros e normalização |
| `RAG: retrieval híbrido` | Dense, BM25, RRF e Top-K |
| `RAG: validar benchmark (sem API)` | Carregamento dos 24 casos sem chamadas ao modelo |
| `RAG: caso do benchmark com judge` | Um caso com Triad; gera custo de API |
| `Python: arquivo atual` | Executa o arquivo aberto |
| `Pytest: arquivo atual` | Depura o teste aberto com `pytest -s` |

Ao iniciar as configurações de pergunta, o VS Code abre uma caixa no topo
para receber a consulta. No benchmark, informe um ID como `Q18`.

## Primeira sessão: Etapa 1 com JSONL

Selecione `Etapa 1: loader + preview`, pressione `F5` e escolha `jsonl`. Use
os breakpoints abaixo para acompanhar um ticket desde a linha do arquivo até
o `Document` final:

1. `ingestion/preview/preview_jsonl.py:34`: entrada do loader;
2. `ingestion/loaders/jsonl_loader.py:37`: conversão para `Path`;
3. `ingestion/loaders/jsonl_loader.py:47`: abertura do arquivo;
4. `ingestion/loaders/jsonl_loader.py:53`: conversão da linha JSON;
5. `ingestion/loaders/jsonl_loader.py:60`: serialização do cabeçalho;
6. `ingestion/loaders/jsonl_loader.py:64`: chunking apenas do corpo;
7. `ingestion/loaders/jsonl_loader.py:66`: montagem da metadata comum;
8. `ingestion/loaders/jsonl_loader.py:81`: criação de cada parte;
9. `ingestion/loaders/jsonl_loader.py:89`: criação do `Document`;
10. `ingestion/preview/preview_jsonl.py:41`: visualização do resultado.

No painel **Watch**, adicione gradualmente:

```text
path
line_number
ticket
header
body
body_parts
base_metadata
metadata
documents[-1].page_content
documents[-1].metadata
```

Depois selecione `Etapa 1: builder sem reindexar` e escolha `jsonl`. Coloque
breakpoints em:

1. `ingestion/build/builder_file/jsonl_builder.py:50`;
2. `ingestion/build/builder_file/jsonl_builder.py:23`;
3. `ingestion/build/_common.py:22`;
4. `ingestion/build/_common.py:26`;
5. `ingestion/build/_common.py:32`;
6. `ingestion/build/_common.py:39`;
7. `ingestion/build/_common.py:45`.

Esse segundo percurso demonstra que o builder reutiliza o loader e valida
texto, campos obrigatórios, sensibilidade e unicidade de `chunk_id`. Ele não
gera embeddings porque a configuração não passa `--save-index`.

Para estudar a persistência sem sobrescrever índices, abra
`ingestion/build/_faiss.py` e conheça `save_faiss_index`, mas execute
`Etapa 1: recarga e sanidade FAISS`. Os pontos principais são:

1. `ingestion/build/_faiss.py:52`: resolve o diretório do índice;
2. `ingestion/build/_faiss.py:53`: verifica `index.faiss`;
3. `ingestion/build/_faiss.py:55`: verifica `index.pkl`;
4. `ingestion/build/_faiss.py:59`: executa `FAISS.load_local`;
5. `ingestion/preview/sanity_faiss.py:66`: inicia o cálculo das estatísticas;
6. `ingestion/preview/sanity_faiss.py:87`: vetoriza a pergunta da busca.

Essa configuração chama a API para vetorizar as três perguntas de teste,
mas não relê o corpus e não salva um novo índice.

## Breakpoints para acompanhar o fluxo completo

Coloque um breakpoint clicando à esquerda do número da linha nestes pontos:

1. `app/interface_rag.py`, em `ask_pipeline`;
2. `src/pipeline.py`, em `answer_with_trace`;
3. `src/guardrails/lgpd.py`, em `classify_lgpd_question`;
4. `src/guardrails/scope.py`, em `classify_question_scope`;
5. `src/retrieval/query_analyzer.py`, em `analyze_question`;
6. `src/retrieval/dense.py`, em `dense_search`;
7. `src/retrieval/sparse.py`, no método `search`;
8. `src/retrieval/fusion.py`, em `reciprocal_rank_fusion`;
9. `src/generation/generator.py`, em `generate_rag_response`.

Use:

- `F5`: iniciar ou continuar;
- `F10`: executar a linha sem entrar na função;
- `F11`: entrar na função;
- `Shift+F11`: sair da função;
- `Shift+F5`: encerrar.

## Variáveis úteis

No painel **Variables** ou **Watch**, acompanhe:

```text
normalized_question
lgpd_decision.action
scope_decision.classification
retrieval_response.analysis.query
retrieval_response.analysis.filters
dense_response.results
sparse_results
fused_results
retrieved_ids
documents
response
```

Para inspecionar um objeto Pydantic no **Debug Console**:

```python
retrieval_response.analysis.model_dump(exclude_none=True)
response.model_dump()
```

Para examinar apenas metadados de um resultado, sem imprimir todo o texto:

```python
retrieval_response.results[0].document.metadata
```

## Breakpoint condicional

Clique com o botão direito no breakpoint e escolha **Edit Breakpoint**. Um
exemplo para parar somente no ticket desejado é:

```python
result.chunk_id == "tickets:TCK-1005:000"
```

A expressão deve usar variáveis existentes exatamente naquele escopo.

## Depurando o Streamlit

Selecione `RAG: Streamlit`, pressione `F5` e abra
`http://localhost:8502`. O processo fica preso ao depurador; breakpoints na
interface e no pipeline serão atingidos depois que a pergunta for enviada.

A porta 8502 foi reservada para o debug, permitindo manter uma execução normal
na porta 8501. Se a 8502 estiver ocupada, encerre a execução anterior com
`Ctrl+C` no terminal ou `Shift+F5` no depurador antes de iniciar outra.

## Cuidados

- As configurações de pipeline, retrieval e judge chamam a API e podem gerar
  custo. `validate-only` e os testes com fakes não fazem chamadas reais.
- O painel **Variables** pode mostrar variáveis de ambiente. Não compartilhe
  capturas que exibam `OPENAI_API_KEY`.
- Evite imprimir `page_content` de documentos restritos ou dados pessoais.
- `justMyCode=true` mantém o passo a passo no código do projeto. Troque
  temporariamente para `false` somente se precisar investigar internamente o
  LangChain ou o Streamlit.

## Problemas comuns

- **Tipo `debugpy` não reconhecido:** instale a extensão Python Debugger e
  execute `Developer: Reload Window`.
- **`ModuleNotFoundError`:** confirme que a pasta aberta é a raiz do projeto e
  que o `cwd` exibido no terminal é `vende-facil`.
- **Breakpoint cinza ou não atingido:** confirme o interpretador da `.venv` e
  inicie a configuração correspondente ao módulo que contém o breakpoint.
- **Resposta demora:** uma linha que chama `invoke` ou `embed_query` aguarda a
  API. Use `F10` e espere o retorno; pressionar `F5` repetidamente não acelera
  a chamada.
