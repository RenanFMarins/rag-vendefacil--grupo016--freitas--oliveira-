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

## Encontro 3 - AAAA-MM-DD

**Etapa:** 3 - Síntese estruturada, evidência e guardrails de LGPD

### Relato individual - [Nome do Integrante 1]

### Relato individual - [Nome do Integrante 2]

### Resumo do dia (escrito em conjunto)

**Entregamos hoje:**
-

**Ficou pendente:**
-

**Bloqueios em aberto:**
-

**Próximo passo (início do encontro 4):**
-

**Uso de assistentes de IA:**
-

---

## Encontro 4 - AAAA-MM-DD

**Etapa:** 4 - Avaliação (RAG Triad), interface e relatório

### Relato individual - [Nome do Integrante 1]

### Relato individual - [Nome do Integrante 2]

### Resumo do dia (escrito em conjunto)

**Entregamos hoje:**
-

**Ficou pendente:**
-

**Bloqueios em aberto:**
-

**Preparação para o Demo Day:**
-

**Uso de assistentes de IA:**
-

---

*TIC em Trilhas · PUC-Rio · Instituto ECOA · MCTI Futuro · Softex*
