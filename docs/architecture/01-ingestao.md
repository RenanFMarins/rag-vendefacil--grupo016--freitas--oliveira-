# Etapa 1 - Ingestão e indexação

## Responsabilidade

A Etapa 1 transforma arquivos heterogêneos em `Document` do LangChain. Cada
`Document` possui `page_content` apropriado para embeddings e metadata
padronizada para filtros, rastreabilidade e segurança.

```mermaid
flowchart LR
    A[CSV JSON JSONL MD PDF TXT] --> B[Loader específico]
    B --> C[Chunking adaptativo]
    C --> D[Document + metadata]
    D --> E[OpenAI Embeddings]
    E --> F[FAISS por formato]
    F --> G[index.faiss + index.pkl]
```

## Estratégias por formato

| Formato | Unidade semântica | Implementação | Índice atual |
| --- | --- | --- | ---: |
| CSV | Um registro | Serializa cliente, funcionário, log ou venda em linguagem natural; não divide linhas | 5.460 |
| JSON | Um produto, plano ou loja | Valida o objeto e serializa cada entidade | 58 |
| JSONL | Um ticket | Mantém cabeçalho; divide somente corpo longo em 1.200/120 | 75 |
| Markdown | Uma seção | Divide por cabeçalhos; fallback de tamanho em 1.000/150 | 81 |
| PDF | Parágrafo ou cláusula | Extrai páginas e aplica recursive split em 800/120 | 6 |
| TXT | Uma mensagem de e-mail | Separa pelo marcador `De:`; replica cabeçalho e divide corpo em 1.200/120 | 43 |

Total atualmente persistido: **5.723 chunks**.

## Metadata obrigatória

Todo chunk deve possuir:

```python
{
    "source_file": str,
    "doc_type": str,
    "chunk_id": str,
    "sensitivity": "publico" | "interno" | "restrito",
}
```

Campos adicionais incluem, conforme a fonte: `customer_id`, `ticket_id`,
`employee_id`, `store_id`, `product_id`, `sale_id`, `state`, `module`, `plan`,
`priority`, `status`, `category`, `sentiment`, `date`, `section` e
`error_code`.

`chunk_id` é estável e globalmente único. `sensitivity` é atribuída na
ingestão, e não delegada ao LLM no momento da resposta.

## Arquivos

- Loaders: `ingestion/loaders/*_loader.py`.
- Builders: `ingestion/build/builder_file/*_builder.py`.
- Persistência: `ingestion/build/_faiss.py`.
- Construção conjunta: `ingestion/build/all_indexes.py`.
- Previews e sanidade: `ingestion/preview/`.
- Configuração de caminhos: `src/config.py`.

## Persistência FAISS

Cada formato possui um diretório em `storage/faiss_<formato>/`:

- `index.faiss`: vetores e estrutura de busca;
- `index.pkl`: docstore, IDs e metadata associados.

O carregamento usa `load_local` e não relê nem reindexa as fontes. A opção
`allow_dangerous_deserialization=True` exige que os arquivos locais sejam
confiáveis; nunca se deve carregar um `index.pkl` recebido de origem desconhecida.

## Execução

```bash
# Construção completa; chama embeddings e substitui os índices locais.
.venv/bin/python -m ingestion.build.all_indexes

# Sanidade sem reindexação.
.venv/bin/python -m ingestion.preview.sanity_faiss

# Visualizar chunks persistidos de um formato.
.venv/bin/python -m ingestion.preview.preview_faiss jsonl --limit 5
```

## Testes

- Loaders: `tests/test_*_loader.py`.
- Builders: `tests/test_builders.py`.
- Sanidade: `tests/test_sanity_faiss.py`.

Os testes verificam schema, IDs, sensibilidade, unidades semânticas e
persistência sem depender de um diretório atual específico do terminal.
