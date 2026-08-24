# Mini Desafio RAG - VendeFácil Knowledge Base

Bem-vindo ao **Mini Desafio RAG (Retrieval-Augmented Generation)** baseado no ecossistema da empresa fictícia **VendeFácil Tecnologia Ltda.**!

Este repositório contém a bases de dados sintéticas multi-formato e orientações pedagógicas para a realização de um mini desafio prático projetado ser realizado em **duplas**.

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
mini-desafio/
├── data/                                 # Base de Conhecimento VendeFácil (Multi-formato)
│   ├── structured/                       # CSV e JSON (employees, customers, products, stores)
│   ├── semi_structured/                  # JSONL e CSV (tickets.jsonl, system_logs.csv)
│   └── unstructured/                     # Documentação (.md), Políticas (.pdf/.md), Reuniões (.md), E-mails (.txt)
esperadas
├── starter/                              # Código de partida para os alunos
│   ├── requirements.txt                  # Dependências Python recomendadas
│   ├── schema.py                         # Estrutura Pydantic exigida para as respostas
│   └── ingest_template.py                # Esqueleto didático do pipeline de ingestão
└── docs/                                 # Documentação Didática


```

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

