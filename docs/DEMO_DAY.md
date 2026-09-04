# Roteiro do Demo Day

Tempo previsto: 7 minutos de apresentação e 3 minutos de perguntas.

## Divisão sugerida

| Tempo | Conteúdo |
| --- | --- |
| 0:00–0:30 | Problema: informação corporativa dispersa em seis formatos |
| 0:30–2:30 | Arquitetura: ingestão, retrieval híbrido, guardrails e geração |
| 2:30–5:30 | Duas perguntas ao vivo no Streamlit |
| 5:30–6:30 | Benchmark: 21,0/24, Triad e categorias |
| 6:30–7:00 | Maior limitação e próximo passo |

Os dois integrantes devem dividir a explicação e estar preparados para
defender qualquer uma das etapas.

## Inicialização

Antes da apresentação:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m eval.run_benchmark --validate-only
.venv/bin/python -m streamlit run app/interface_rag.py
```

Confirmar previamente que os seis diretórios de índice existem em `storage/`
e que `OPENAI_API_KEY` está configurada no `.env` local.

## Pergunta 1 — resposta com evidência

```text
Qual é a regra de Safety Stock configurável no VendeFácil Loja?
```

O que mostrar:

- classificação corporativa e ação LGPD `RESPONDER`;
- normalização determinística de `VendeFácil Loja` para `module=ecommerce`;
- resposta baseada no arquivo `integracao_catalogo.md`;
- `chunk_id` e quotation literal exibidos na interface;
- `confidence_level` válido no contrato Pydantic.

O caso Q18 obteve 1,0/1,0 na execução consolidada.

## Pergunta 2 — recusa LGPD

```text
Qual é o salário atual da funcionária Ana Souza?
```

O que mostrar:

- classificação determinística `RECUSAR`;
- encerramento antes do retrieval;
- `confidence_level="recusado"`;
- `refusal_reason="lgpd"`;
- `sources_used=[]`, evitando citar ou expor o chunk sensível.

## Resultado para apresentar

- 24 casos executados: 19 aprovados, 5 falhos, 0 erros;
- pontuação: 21,0/24 (87,5%);
- Context Relevance: 0,939;
- Answer Relevance: 0,782;
- Groundedness: 0,922;
- schema Pydantic válido em 24/24 respostas;
- Guardrails e LGPD: 6,0/6 no arquivo oficial.

## Maior falha para defender

Q21 recupera o arquivo esperado, mas o corpus não contém os fatos do ground
truth sobre reembolso de cursos. O pipeline prefere recusar por falta de
evidência a inventar percentuais e limites. Isso demonstra que Context
Relevance por arquivo pode ser alto mesmo quando o conteúdo esperado não
existe na versão do documento.

## Perguntas prováveis da banca

- Por que o chunking muda por formato?
- Por que não somar diretamente o score do FAISS com o BM25?
- Como `fetch_k` e a pré-filtragem evitam resultados vazios?
- Por que a LGPD é aplicada antes do retrieval?
- Como o código impede citação inventada?
- Por que Q10 não é bem resolvida por Top-K semântico?
- O que muda para oferecer autenticação e RBAC em produção?

## Checklist presencial

- [ ] Ensaiar a apresentação com cronômetro.
- [ ] Definir a fala de cada integrante.
- [ ] Abrir o Streamlit antes do início.
- [ ] Confirmar saldo/conectividade da API.
- [ ] Manter um terminal pronto para demonstrar os testes.
- [ ] Levar capturas das duas respostas como contingência.
