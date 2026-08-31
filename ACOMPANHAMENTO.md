# Acompanhamento - Mini Desafio RAG VendeFácil

**Integrante 1:** José Renan Freitas Marins - [@usuario-github](https://github.com/RenanFMarins)
**Integrante 2:** Nome Completo - [@usuario-github](https://github.com/usuario-github)

**Repositório:** `rag-vendefacil--grupo016--freitas--oliveira-`

---

## Como preencher

- Um bloco por encontro, em **ordem cronológica** - o encontro mais recente vai no **fim** do arquivo.
- O relato individual é escrito **pelo próprio integrante**, em primeira pessoa. Não escreva pelo colega.
- Escrever entre **17:30 e 17:40**. `commit` + `push` até as **18:00**, mesmo que o dia não tenha fechado.
- Mensagem de commit: `acompanhamento: AAAA-MM-DD`

**Um relato útil responde:** o que eu implementei, qual decisão técnica eu tomei e por quê, onde travei, e como (ou se) resolvi.

<details>
<summary>Exemplo de relato individual bom × ruim</summary>

❌ *"Trabalhei na parte de ingestão junto com meu colega. Avançamos bastante e conseguimos carregar os arquivos."*

✅ *"Implementei os loaders de CSV e JSONL em `src/ingest.py`. Decidi serializar cada linha do `customers.csv` como frase em linguagem natural em vez de manter o formato separado por vírgula, porque nos primeiros testes de similaridade os chunks CSV crus não recuperavam nada - o embedding não separa campo de valor. Travei ~40 min no `tickets.jsonl`: o `state` estava indo para o texto do chunk mas não para os metadados, então o filtro voltava vazio. Resolvi movendo a extração para antes da criação do `Document`. Usei o Claude para gerar o esqueleto do parser de JSONL; ajustei o schema de metadados na mão."*

</details>

---

## Encontro 1 - 2026-08-24

**Etapa:** 1 - Ingestão heterogênea, metadados e indexação vetorial

### Relato individual - [José Renan Freitas Marins] - 2026-08-24

<!-- Nessa etapa foi discutido e analisado a documentação de como o desafio deveria ser seguido e quais bibliotecas utilizariamos. Foi divido as tarefas e quais arquivos deveriamos analisar e tratar. Fiquei com o tipos de documento no formato de TXT, MK e jsonl. Criei algumas pastas para organizar melhor o projeto. -->

### Relato individual - [Nome do Integrante 2]

<!-- Escreva você mesmo, em primeira pessoa. O que implementou, que decisão tomou e por quê, onde travou. -->

### Resumo do dia (escrito em conjunto)

**Entregamos hoje:**
2026-08-24 - estrutura base do projeto e as dependencias necessarias e divisão das tarefas por arquivos

**Ficou pendente:**
ingestao dos arquivos de formato txt, mk e jsonl

**Bloqueios em aberto:**


**Próximo passo (início do encontro 2):**
analisar, tratar e fazer a ingestao dos arquivos e gerar um resumo desses arquivos.

**Uso de assistentes de IA:**
-

---

## Encontro 2 - 2026-08-26

**Etapa:** 2 - Busca híbrida e filtragem por metadados

### Relato individual - [José Renan Freitas Marins] - 2026-08-26

<!-- Hoje implementei o loader de arquivos TXT para os e-mails em data/unstructured/emails. Decidi considerar cada mensagem iniciada por De: como uma unidade semântica e aplicar o chunking somente no corpo, replicando o cabeçalho em todas as partes para preservar remetente, destinatário, data e assunto. Também extraí metadados como customer_id, ticket_id, estado e módulo, sem gerar ticket_id quando ele não aparece no conteúdo. Implementei ainda a classificação de sensibilidade durante a ingestão, identificando cinco e-mails com credenciais como restrito. Criei um script de preview para visualizar os Documents produzidos e oito testes para validar os 43 e-mails, threads com múltiplas mensagens, corpos longos, metadados, IDs estáveis e conteúdo sensível.Usei o ChatGPT para auxiliar na estrutura inicial, explicar as expressões regulares e sugerir casos de teste; revisei o código e validei os resultados pelo preview e pelo Pytest. -->


### Relato individual - [Nome do Integrante 2]

### Resumo do dia (escrito em conjunto)

**Entregamos hoje:**
- Loader adaptativo para os 43 e-mails TXT.
- Extração de metadados e relacionamento por customer_id e ticket_id.
- Classificação de 38 documentos internos e 5 restritos.
- Preview dos Documents gerados.
- Oito testes do loader TXT, com 14 testes totais passando.

**Ficou pendente:**
- Implementar o loader Markdown.
- Integrar todos os loaders ao pipeline de embeddings e FAISS.
- Definir posteriormente as regras de audiência e autorização por perfil.

**Bloqueios em aberto:**
- A integração dos loaders com embeddings e FAISS depende da conclusão do loader Markdown e da definição do modelo de embeddings.

**Próximo passo (início do encontro 3):**
- Concluir o tratamento e a ingestão dos arquivos Markdown.
- Gerar os embeddings, criar e persistir o índice FAISS.
- Executar o script de sanidade para conferir a quantidade de chunks

**Uso de assistentes de IA:**
- Utilizei o ChatGPT para auxiliar na estruturação dos loaders de TXT, sugerir metadados e casos de teste. Cconferi os documentos gerados pelos previews e validei a implementação executando os testes automatizados.

---

## Encontro 3 - 2026-03-28

**Etapa:** 3 - Síntese estruturada, evidência e guardrails de LGPD

### Relato individual - [Nome do Integrante 1]
Hoje finalizei minha parte da Etapa 1, responsável pelos formatos JSONL, TXT e Markdown. Organizei os builders responsáveis por carregar os documentos processados, gerar embeddings e salvar um índice FAISS para cada formato. Também acompanhei o funcionamento dos arquivos index.faiss e index.pkl e validei a recarga dos índices sem reindexar os documentos.
Criei o preview_faiss.py, que permite visualizar os documentos, metadados e vetores armazenados. Também implementei o sanity_faiss.py, que carrega os três índices, contabiliza os chunks, apresenta a distribuição por doc_type e executa três perguntas de teste, retornando os cinco chunks mais similares.
Os índices foram testados com dados reais e apresentaram 199 chunks: 75 de JSONL, 43 de TXT e 81 de Markdown. Também organizei as alterações em commits separados por configuração, construção dos índices e scripts de validação.
### Relato individual - [Nome do Integrante 2]

### Resumo do dia (escrito em conjunto)

**Entregamos hoje:**
-Builders para JSONL, TXT e Markdown.
Geração de um índice FAISS para cada formato.
Persistência dos índices com save_local.
Recarga dos índices com load_local, sem reindexação.
Script para visualizar os dados persistidos no FAISS.
Script de sanidade com total de chunks e distribuição por doc_type.
Três perguntas de teste com os cinco resultados mais similares.
Validação dos metadados obrigatórios.
Testes automatizados para a busca de sanidade.

**Ficou pendente:**
-
Executar a validação final considerando os seis formatos da Etapa 1.
**Bloqueios em aberto:**
-

**Próximo passo (início do encontro 4):**
-

**Uso de assistentes de IA:**
-O assistente de IA foi utilizado para auxiliar na organização dos builders, na implementação da persistência e recarga dos índices FAISS e na criação dos scripts de preview e sanidade.
Também foi utilizado para explicar o funcionamento dos embeddings, dos arquivos index.faiss e index.pkl, dos metadados e da busca por similaridade.

---

## Encontro 4 - 2026-08-31

**Etapa:** 4 - Avaliação (RAG Triad), interface e relatório

### Relato individual - [José Renan Freitas Marins]
Hoje integrei a interface Streamlit ao pipeline estruturado do projeto. Substituí o fluxo antigo, que acessava diretamente o FAISS e o ChatOpenAI, pelo uso de `RAGPipeline`. Também participei da criação do `app/bootstrap.py`, responsável por carregar os índices JSONL, TXT e Markdown e montar o Dense, BM25, RRF, modelos e guardrails.

Testei perguntas corporativas, LGPD, fora de escopo, filtros por metadados e consultas com múltiplas fontes. Ao comparar o projeto com o arquivo de benchmark fornecido pelo professor, alguns formatos ainda não estão integrados nesta branch.


### Relato individual - [Nome do Integrante 2]

### Resumo do dia (escrito em conjunto)

**Entregamos hoje:**
- Integração funcional da interface Streamlit com o `RAGPipeline`.
- Criação do bootstrap para montar e manter em cache embeddings, índices, BM25, modelos e pipeline.
- Remoção do pipeline paralelo e da busca externa da interface.
- Renderização de `RAGResponse`, confiança, recusas e evidências.
- Testes de interface e bootstrap sem chamadas reais à OpenAI.
- Testes manuais de respostas corporativas, LGPD, fora de escopo e ausência de evidência.
- Diagnóstico de divergências entre o benchmark e o corpus atual.

**Ficou pendente:**

- Integrar os formatos CSV, JSON e PDF exigidos pela Etapa 1.
- Corrigir consultas multi-documento, evitando um único filtro `doc_type`
- Preservar nomes de clientes e outras entidades na query semântica.
- Repor candidatos permitidos quando um chunk restrito for removido do Top-K.
- Adaptar o runner para executar as 20/24 perguntas oficiais.
- Preencher `RELATORIO.md` com métricas e diagnóstico das três piores falhas.
- Atualizar o README e organizar os arquivos obrigatórios na raiz.

**Bloqueios em aberto:**
- Algumas perguntas dependem de CSV, JSON e PDF, que ainda não estão indexados.
- O gabarito da pergunta sobre reembolso de cursos contém informações que não aparecem no arquivo indicado.
- Precisamos integrar ou receber o trabalho da dupla referente aos formatos ainda ausentes.

**Preparação para o Demo Day:**
- Selecionar duas perguntas estáveis para a demonstração.
- Sugestão de pergunta corporativa:
  “Como deve ser realizada uma operação de sangria no caixa do VendeFácil PDV?”
- Sugestão de guardrail:
  “Qual é o salário atual da funcionária Ana Souza?”

**Uso de assistentes de IA:**
- Utilizamos assistência de IA para revisar a integração entre Streamlit e o pipeline structured e analisar o comportamento do classificador de escopo.
- A IA ajudou a comparar o benchmark com os documentos efetivamente indexados e a identificar fontes ausentes ou gabaritos divergentes.
- As sugestões foram verificadas por testes automatizados, inspeção dos índices e execução manual de perguntas.
- Ajustamos o resultado sugerido para preservar o pipeline oficial, remover a busca web e impedir respostas sem evidência.

---

*TIC em Trilhas · PUC-Rio · Instituto ECOA · MCTI Futuro · Softex*
