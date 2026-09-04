# Relatório final — Assistente RAG VendeFácil

## Resultado executivo

As quatro etapas foram integradas e validadas. A execução de 2026-09-04
avaliou os 24 casos presentes no arquivo recebido, apesar de sua descrição
mencionar 20.

| Indicador | Resultado |
| --- | ---: |
| Pontuação pela rubrica | 21,0/24 (87,5%) |
| Casos aprovados | 19/24 (79,2%) |
| Casos falhos | 5 |
| Erros de execução | 0 |
| `RAGResponse` válida | 24/24 (100%) |
| Acurácia de resposta/recusa | 23/24 (95,8%) |
| Evidência presente e literal | 18/19 (94,7%) |
| Context Relevance | 0,939 |
| Answer Relevance | 0,782 |
| Groundedness | 0,922 |

O resultado detalhado e os checks de cada questão estão em
[`results/results.json`](results/results.json) e a tabela gerada pelo runner
está em [`results/benchmark_summary.md`](results/benchmark_summary.md).

## Resultado por categoria

| Categoria | Pontos | Taxa |
| --- | ---: | ---: |
| Fácil (RAG Básico) | 5,0/5 | 100% |
| Filtragem por Metadados | 3,0/4 | 75% |
| Múltiplas Fontes | 2,5/3 | 83,3% |
| Razão e Solução de Problemas | 4,0/4 | 100% |
| Guardrails e LGPD | 6,0/6 | 100% |
| Políticas Internas | 0,5/2 | 25% |

## Três piores falhas e origem

1. **Q21 — reembolso de cursos (0,0/1,0).** O retrieval encontrou o arquivo
   esperado, porém `beneficios_e_viagens.md` não contém os valores de 80%,
   R$ 2.500 nem as condições de cursos descritas no ground truth. O pipeline
   recusou por falta de evidência, evitando alucinação. A origem é a
   divergência entre corpus e gabarito.
2. **Q10 — maior MRR em SP (0,5/1,0).** O gabarito indica `CUST008` com
   R$ 3.100, mas o CSV atual possui clientes paulistas com MRR maior, como
   `CUST1214` com R$ 3.487,22. Uma busca Top-K semântica também não garante o
   máximo global. A origem combina divergência de dados e ausência de um
   operador determinístico de agregação.
3. **Q22 — conectividade no home office (0,5/1,0).** O arquivo correto foi
   recuperado, mas não possui as velocidades 100/20 Mbps nem as exigências de
   disponibilidade, ponto e câmera esperadas. A resposta ficou limitada ao
   auxílio de internet e à VPN presentes na fonte. A origem é a divergência
   entre corpus e gabarito.

Q06 também diverge porque o JSONL atual contém sete tickets críticos, enquanto
o ground truth menciona somente `TCK-1005`. Em Q19, a resposta identificou as
duas filiais corretamente, mas o juiz exigiu também quantidades de terminais
presentes no ground truth e não pedidas explicitamente na pergunta.

## Diagnóstico por camada

- **Ingestão:** seis formatos, 5.723 chunks e metadata obrigatória validada.
- **Retrieval:** filtros seletivos funcionam; BM25 vence em códigos exatos e
  Dense vence em paráfrases. O mapeamento `VendeFácil Loja -> ecommerce` foi
  estabilizado deterministicamente após Q18 expor ambiguidade com "estoque".
- **Geração:** todas as 24 saídas respeitaram o schema e nenhuma citação foi
  inventada pelo LLM.
- **Guardrails:** a categoria Guardrails e LGPD obteve 6,0/6; o arquivo oficial
  não possui caso marcado como `masked_answer`, então mascaramento é coberto
  pela suíte automatizada.

## O que faríamos com mais 4 horas

1. versionar corpus e benchmark como um par compatível;
2. validar automaticamente fatos e fontes do ground truth;
3. adicionar uma rota de consulta tabular para `max`, `sum` e `count`;
4. calibrar o judge com casos limítrofes como Q19;
5. repetir o benchmark para medir a variância entre execuções do LLM.

## Verificação

```text
270 passed
0 failed
1 aviso de depreciação de langchain-community
```

O relatório técnico expandido está em
[`docs/RELATORIO.md`](docs/RELATORIO.md).
