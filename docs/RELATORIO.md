# Relatório técnico - Assistente RAG VendeFácil

## Identificação

- Integrante 1: José Renan Freitas Marins
- Integrante 2: a preencher
- Branch de integração: `dev`
- Baseline Git: `62cac4b` mais as alterações locais descritas neste relatório
- Data da avaliação: 2026-09-04

## Resumo executivo

O projeto implementa ingestão de seis formatos, busca híbrida com filtros,
geração estruturada com evidências, guardrails de LGPD, interface Streamlit
e avaliação por RAG Triad. As quatro etapas estão integradas. A execução
completa dos 24 casos presentes no arquivo oficial atingiu 21,0/24 pontos
(87,5%), sem erro de execução e com 100% das saídas válidas no schema.

## Etapa 1 - Ingestão

- Formatos: CSV, JSON, JSONL, Markdown, PDF e TXT.
- Chunks persistidos atualmente: 5.723.
- Estratégia: chunking adaptativo por unidade semântica.
- Metadata obrigatória: `source_file`, `doc_type`, `chunk_id` e `sensitivity`.
- Persistência: seis índices FAISS, cada um com `index.faiss` e `index.pkl`.

Detalhes: [arquitetura da ingestão](architecture/01-ingestao.md).

## Etapa 2 - Retrieval

- Query Analyzer com saída Pydantic.
- Catálogo dinâmico, normalização e validação de filtros.
- Dense Search em FAISS com estratégia adaptativa de filtragem.
- Sparse Search com BM25.
- Fusão por Reciprocal Rank Fusion e Top-K configurável.

Detalhes: [arquitetura do retrieval](architecture/02-retrieval.md).

## Etapa 3 - Geração e segurança

- `RAGResponse` validada por Pydantic.
- Evidências construídas deterministicamente a partir dos Documents.
- LGPD com ações RECUSAR, MASCARAR e RESPONDER.
- Classificação de escopo corporativo.
- Geração baseada somente no contexto e retry limitado.
- Exclusão de chunks `restrito` antes da geração.

Detalhes: [geração e guardrails](architecture/03-generation-guardrails.md).

## Etapa 4 - Interface e avaliação

A interface Streamlit utiliza o mesmo `RAGPipeline` da aplicação e apresenta
resposta, confiança, recusa e evidências. O arquivo oficial possui 24 casos,
apesar de sua descrição e do guia mencionarem 20.

### Resultado do benchmark completo

| Métrica | Resultado |
| --- | ---: |
| Pontuação oficial | 21,0/24 (87,5%) |
| Casos executados | 24 |
| Casos aprovados | 19 (79,2%) |
| Casos com falha | 5 |
| Casos com erro | 0 |
| Validade do schema | 24/24 (100%) |
| Acurácia de recusa | 23/24 (95,8%) |
| Presença de evidência | 18/19 (94,7%) |
| Correspondência de fonte | 18/19 (94,7%) |
| Quotation literal no chunk | 18/19 (94,7%) |
| Context Relevance | 0,939 em 19 casos |
| Answer Relevance | 0,782 em 19 casos |
| Groundedness | 0,922 em 18 casos |

### Resultado por categoria

| Categoria | Pontos | Taxa |
| --- | ---: | ---: |
| Fácil (RAG Básico) | 5,0/5 | 100% |
| Filtragem por Metadados | 3,0/4 | 75% |
| Múltiplas Fontes | 2,5/3 | 83,3% |
| Razão e Solução de Problemas | 4,0/4 | 100% |
| Guardrails e LGPD | 6,0/6 | 100% |
| Políticas Internas | 0,5/2 | 25% |

Comando oficial:

```bash
.venv/bin/python -m eval.run_benchmark --with-judge
```

### Três piores falhas

1. **Q21 — reembolso de cursos (0,0/1,0).** O retrieval encontrou
   `beneficios_e_viagens.md`, mas o arquivo fala apenas de benefícios e
   despesas comerciais. Os valores de 80%, R$ 2.500 e as condições de curso
   do gabarito não existem no corpus. A geração recusou por falta de evidência,
   que é o comportamento seguro. Origem: divergência corpus/gabarito, não
   falha de recuperação.
2. **Q10 — maior MRR em SP (0,5/1,0).** O gabarito aponta `CUST008` com
   R$ 3.100, mas o CSV atual contém clientes de SP com MRR superior, como
   `CUST1214` com R$ 3.487,22. Além disso, Top-K semântico não garante máximo
   global. Origem: divergência corpus/gabarito e limitação do retrieval para
   agregação numérica.
3. **Q22 — conectividade no home office (0,5/1,0).** O arquivo recuperado é
   correto, mas não contém 100/20 Mbps, disponibilidade em Slack/Teams,
   registro de ponto ou câmera aberta esperados no gabarito. O modelo se
   limitou a auxílio de internet e VPN presentes no texto. Origem: divergência
   corpus/gabarito; completar a resposta seria alucinação.

Outras falhas: Q06 encontra sete tickets críticos no JSONL atual, enquanto o
gabarito menciona apenas `TCK-1005`; Q19 respondeu corretamente quais são as
duas filiais, mas o juiz também esperava a quantidade de terminais descrita no
ground truth, detalhe não solicitado explicitamente na pergunta.

### O que faríamos com mais 4 horas

1. versionar corpus e benchmark como um par compatível;
2. validar automaticamente se fatos e fontes do ground truth existem;
3. adicionar uma rota determinística para agregações tabulares (`max`, `sum`,
   `count`) antes da síntese;
4. calibrar a rubrica do juiz com casos limítrofes como Q19;
5. repetir o benchmark em mais de uma rodada para medir variância do LLM.

## Testes

Baseline após a correção determinística de nomes de produtos:

```text
270 passed
0 failed
1 aviso de depreciação de langchain-community
```

## Limitações

- Não há autenticação ou autorização por perfil.
- O histórico da interface não constitui memória conversacional.
- FAISS local e BM25 em memória não são uma arquitetura distribuída.
- A qualidade depende do corpus, dos embeddings e dos modelos configurados.
- O benchmark completo com juiz gera custo adicional de API.
- O arquivo recebido declara 20 perguntas, mas possui 24; nenhuma está
  marcada como `masked_answer`, apesar de o guia prever um caso.
- O corpus ampliado diverge de alguns fatos do ground truth, conforme o
  diagnóstico de falhas.

## Conclusão

O pipeline atende aos requisitos arquiteturais e obteve 87,5% da pontuação
objetiva. As principais perdas restantes não justificam inserir fatos no prompt:
elas devem ser tratadas por alinhamento de versões do corpus/gabarito e por um
operador determinístico de agregação para perguntas tabulares.
