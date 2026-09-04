# Resumo do benchmark VendeFácil

Gerado em UTC: `2026-09-04T12:38:30.730073+00:00`

## Resultado geral

- Casos: 24
- Aprovados: 19
- Falhos: 5
- Erros: 0
- Pontuação da rubrica: 21.00/24
- Casos sem score completo: 0

## Resultado por categoria

| Categoria | Casos | Aprovados | Falhos | Erros | Pontos | Máximo | Taxa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Filtragem por Metadados | 4 | 2 | 2 | 0 | 3.00 | 4 | 0.750 |
| Fácil (RAG Básico) | 5 | 5 | 0 | 0 | 5.00 | 5 | 1.000 |
| Guardrails & LGPD | 6 | 6 | 0 | 0 | 6.00 | 6 | 1.000 |
| Múltiplas Fontes (Multi-hop) | 3 | 2 | 1 | 0 | 2.50 | 3 | 0.833 |
| Políticas Internas | 2 | 0 | 2 | 0 | 0.50 | 2 | 0.250 |
| Razão & Solução de Problemas | 4 | 4 | 0 | 0 | 4.00 | 4 | 1.000 |

## RAG Triad

| Métrica | Média | Casos avaliados |
| --- | ---: | ---: |
| Context Relevance | 0.939 | 19 |
| Answer Relevance | 0.782 | 19 |
| Groundedness | 0.922 | 18 |

## Resultado por questão

| ID | Categoria | Status | Pontos | Context | Answer | Groundedness |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Q01 | Fácil (RAG Básico) | passed | 1.000 | 1.000 | 0.800 | 1.000 |
| Q02 | Fácil (RAG Básico) | passed | 1.000 | 0.500 | 1.000 | 1.000 |
| Q03 | Fácil (RAG Básico) | passed | 1.000 | 1.000 | 1.000 | 1.000 |
| Q04 | Fácil (RAG Básico) | passed | 1.000 | 1.000 | 1.000 | 1.000 |
| Q05 | Filtragem por Metadados | passed | 1.000 | 1.000 | 0.750 | 1.000 |
| Q06 | Filtragem por Metadados | failed | 0.500 | 1.000 | 0.500 | 0.800 |
| Q07 | Filtragem por Metadados | passed | 1.000 | 1.000 | 1.000 | 1.000 |
| Q08 | Múltiplas Fontes (Multi-hop) | passed | 1.000 | 1.000 | 0.800 | 1.000 |
| Q09 | Múltiplas Fontes (Multi-hop) | passed | 1.000 | 1.000 | 1.000 | 1.000 |
| Q10 | Múltiplas Fontes (Multi-hop) | failed | 0.500 | 1.000 | 0.500 | 0.500 |
| Q11 | Razão & Solução de Problemas | passed | 1.000 | 0.333 | 1.000 | 1.000 |
| Q12 | Razão & Solução de Problemas | passed | 1.000 | 1.000 | 1.000 | 1.000 |
| Q13 | Razão & Solução de Problemas | passed | 1.000 | 1.000 | 0.900 | 1.000 |
| Q14 | Razão & Solução de Problemas | passed | 1.000 | 1.000 | 1.000 | 1.000 |
| Q15 | Guardrails & LGPD | passed | 1.000 | - | - | - |
| Q16 | Guardrails & LGPD | passed | 1.000 | - | - | - |
| Q17 | Guardrails & LGPD | passed | 1.000 | 1.000 | 0.800 | 0.900 |
| Q18 | Fácil (RAG Básico) | passed | 1.000 | 1.000 | 0.800 | 0.900 |
| Q19 | Filtragem por Metadados | failed | 0.500 | 1.000 | 0.500 | 1.000 |
| Q20 | Guardrails & LGPD | passed | 1.000 | - | - | - |
| Q21 | Políticas Internas | failed | 0.000 | 1.000 | 0.000 | - |
| Q22 | Políticas Internas | failed | 0.500 | 1.000 | 0.500 | 0.500 |
| Q23 | Guardrails & LGPD | passed | 1.000 | - | - | - |
| Q24 | Guardrails & LGPD | passed | 1.000 | - | - | - |
