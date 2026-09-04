# Mini Desafio RAG - VendeFácil Knowledge Base

Bem-vindo ao **Mini Desafio RAG (Retrieval-Augmented Generation)** baseado no ecossistema da empresa fictícia **VendeFácil Tecnologia Ltda.**!

Este repositório contém bases de dados sintéticas multi-formato e uma
implementação completa do assistente RAG desenvolvida em dupla.

---

## Contexto do Negócio: VendeFácil Tecnologia Ltda.

A VendeFácil é uma empresa brasileira de tecnologia que fornece sistemas de gestão para pequenos e médios varejistas (supermercados, farmácias, lojas de vestuário, petshops, etc.). O portfólio da empresa é composto por 5 produtos principais:

1. **VendeFácil PDV:** Sistema de frente de caixa com suporte a NFC-e, SAT e funcionamento offline.
2. **VendeFácil Estoque:** Gestão de inventário multiloja, inventário cego, transferência entre filiais e entrada via XML de NF-e.
3. **VendeFácil Loja:** Plataforma de e-commerce omnicanal e catálogo para WhatsApp/marketplaces.
4. **VendeFácil Analytics:** Dashboards executivos, DRE gerencial e curva ABC de vendas.
5. **VendeFácil Pay:** Solução de pagamento com TEF IP e PIX dinâmico integrado às maquininhas Pinpad.

---



## O Desafio

O objetivo das duplas é **construir um Assistente de Inteligência Artificial para a Knowledge Base da VendeFácil**, capaz de:

- Processar e indexar fontes de dados heterogêneas (CSV, JSON, JSONL, Markdown, PDF e TXT).
- Executar busca híbrida (Embeddings + BM25) com **filtragem avançada por metadados** (ex: estado `MG`, módulo `estoque`, cliente `CUST001`).
- Gerar respostas estritamente fundamentadas em fatos, com **saída estruturada em Pydantic** citando fontes e nível de confiança.
- Aplicar **Guardrails e regras de segurança (LGPD)** para impedir o vazamento de informações sensíveis (salários de colaboradores, senhas/chaves de API) e reconhecer perguntas fora do escopo.
- Avaliar o desempenho do sistema através do benchmark fornecido utilizando as métricas da **RAG Triad** (Relevância do Contexto, Relevância da Resposta e Groundedness).

---



## Estrutura do Repositório

```
vende-facil/
├── app/                    # Interface Streamlit e composition root
├── benchmark/              # Perguntas e ground truth oficiais
├── data/                   # Corpus CSV, JSON, JSONL, MD, PDF e TXT
├── eval/                   # Runner do benchmark e RAG Triad
├── ingestion/
│   ├── loaders/            # Leitura e chunking adaptativo
│   ├── build/              # Builders e persistência FAISS
│   └── preview/            # Inspeção e sanidade
├── src/
│   ├── retrieval/          # Query Analyzer, Dense, BM25 e RRF
│   ├── generation/         # Geração e evidências
│   ├── guardrails/         # LGPD, escopo e mascaramento
│   ├── inspection/         # Scripts executáveis de diagnóstico
│   └── pipeline.py         # Orquestração do RAG
├── starter/schema.py       # Contrato Pydantic oficial
├── storage/                # Índices locais; não versionados
├── tests/                  # Testes por domínio
└── docs/                   # Arquitetura, acompanhamento e relatório
```

## Documentação da arquitetura

- [Visão geral](architecture/00-visao-geral.md)
- [Etapa 1 - Ingestão e indexação](architecture/01-ingestao.md)
- [Etapa 2 - Recuperação híbrida](architecture/02-retrieval.md)
- [Etapa 3 - Geração e guardrails](architecture/03-generation-guardrails.md)
- [Etapa 4 - Interface](architecture/04-interface.md)
- [Etapa 4 - Avaliação](architecture/05-avaliacao.md)
- [Integração final](architecture/06-integracao-final.md)
- [Decisões arquiteturais](architecture/DECISOES.md)
- [Detalhes da RAG Triad](RAG_TRIAD.md)
- [Debug no VS Code](DEBUG_VSCODE.md)
- [Roteiro do Demo Day](DEMO_DAY.md)
- [Relatório](RELATORIO.md)

## Execução do projeto

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r starter/requirements.txt
cp .env.example .env

# Depois de configurar OPENAI_API_KEY:
.venv/bin/python -m ingestion.build.all_indexes
.venv/bin/python -m pytest -q
.venv/bin/python -m streamlit run app/interface_rag.py
```

O assistente usa somente o corpus interno e não possui fallback para busca na
internet.

---



## Cronograma do Desafio


| Aula        | Carga Horária | Tópico Principal                                | Entregável da Aula                                                                      |
| ----------- | ------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Etapa 1** | 4h            | **Ingestão Heterogênea, Metadados e Vector DB** | Pipeline de carregamento, chunking e indexação no FAISS/Qdrant com metadados extraídos. |
| **Etapa 2** | 4h            | **Busca Híbrida e Filtragem por Metadados**     | Roteador de queries e buscador híbrido (Embeddings + BM25/Filtros).                     |
| **Etapa 3** | 4h            | **Síntese Estruturada e Guardrails / LGPD**     | Pipeline LLM com saída Pydantic, citação de evidências e bloqueio de dados sensíveis.   |
| **Etapa 4** | 4h            | **Avaliação (RAG Triad), UI e Defesa Técnica**  | Execução do benchmark, relatório de métricas e apresentação da dupla para a turma.      |


---



## Início Rápido



### 1. Pré-requisitos

- Python 3.10+
- Chave de API de LLM (OpenAI, Groq, OpenRouter ou ambiente Ollama local)



### 2. Instalação das Dependências

```bash
# Clone ou acesse a pasta do repositório
cd mini-desafio

# Crie e ative um ambiente virtual
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale as dependências
pip install -r starter/requirements.txt
```



### 3. Consultar o Guia Didático

- [Guia Didático - Mini Desafio RAG VendeFácil](https://app.notion.com/p/IA-Generativa-RAG-3b3185f9f7ed806b8820ee5292611ee4)

