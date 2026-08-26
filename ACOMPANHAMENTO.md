# Acompanhamento - Mini Desafio RAG VendeFácil

**Integrante 1:** Nome Completo - [@usuario-github](https://github.com/usuario-github)
**Integrante 2:** Nome Completo - [@usuario-github](https://github.com/usuario-github)

**Repositório:** `rag-vendefacil-<sobrenome1>-<sobrenome2>`

---

## Como preencher

- Um bloco por encontro, em **ordem cronológica** - o encontro mais recente vai no **fim** do arquivo.
- O relato individual é escrito **pelo próprio integrante**, em primeira pessoa. Não escreva pelo colega.
- Escrever entre **17:30 e 17:40**. `commit` + `push` até as **18:00**, mesmo que o dia não tenha fechado.
- Mensagem de commit: `acompanhamento: AAAA-MM-DD`

**Um relato útil responde:** o que eu implementei, qual decisão técnica eu tomei e por quê, onde travei, e como (ou se) resolvi.

<details>
<summary>Exemplo de relato individual bom × ruim</summary>

❌ _"Trabalhei na parte de ingestão junto com meu colega. Avançamos bastante e conseguimos carregar os arquivos."_

✅ _"Implementei os loaders de CSV e JSONL em `src/ingest.py`. Decidi serializar cada linha do `customers.csv` como frase em linguagem natural em vez de manter o formato separado por vírgula, porque nos primeiros testes de similaridade os chunks CSV crus não recuperavam nada - o embedding não separa campo de valor. Travei ~40 min no `tickets.jsonl`: o `state` estava indo para o texto do chunk mas não para os metadados, então o filtro voltava vazio. Resolvi movendo a extração para antes da criação do `Document`. Usei o Claude para gerar o esqueleto do parser de JSONL; ajustei o schema de metadados na mão."_

</details>

---

## Encontro 1 - 2026-08-24

**Etapa:** 1 - Ingestão heterogênea, metadados e indexação vetorial

### Relato individual - [Nome do Integrante 1]

<!-- Escreva você mesmo, em primeira pessoa. O que implementou, que decisão tomou e por quê, onde travou. -->

### Relato individual - Luciano Oliveira da Costa

Nesta primeira etapa,trabalhei em conjunto com a minha dupla no alinhamento inicial do desenvolvimento. Discutimos e definimos a estruturação de pastas do repositório para garantir a organização do código e dos dados.
Hoje realizei o carregamento dos arquivos PDF, configurei o text_splitter e realizei a transformação do texto em chunks para preparar a base de conhecimento.

### Resumo do dia (escrito em conjunto)

Hoje iniciamos a Etapa 1, realizando inicialmente uma conversa e alinhamento sobre o escopo da atividade, as etapas que precisam ser desenvolvidas e a organização do projeto. Definimos de forma inicial a estrutura de pastas, a organização do repositório no GitHub e como iremos estruturar o projeto para facilitar o desenvolvimento em dupla.

Após essa etapa de preparação, iniciamos a implementação da ingestão de documentos. Conseguimos realizar o carregamento e processamento inicial de arquivos PDF utilizando bibliotecas do LangChain e fizemos um teste de divisão do conteúdo em chunks com o RecursiveCharacterTextSplitter.

O chunking realizado neste momento teve caráter experimental, apenas para validar o funcionamento do processo. Ainda não foram definidos os parâmetros finais, como tamanho dos chunks e overlap.

## **Entregamos hoje:**

Alinhamento entre a dupla sobre o escopo e os requisitos da Etapa 1.
Análise das atividades que precisam ser desenvolvidas ao longo da etapa.
Definição inicial da organização do projeto e da estrutura de pastas.
Organização inicial do repositório e planejamento de utilização do GitHub.
Implementação inicial do carregamento de documentos PDF utilizando LangChain.
Primeiro teste de divisão dos documentos em chunks utilizando o RecursiveCharacterTextSplitter.
Validação inicial de que o processo de carregamento e chunking está funcionando.
Investigação de problemas relacionados ao carregamento e configuração de bibliotecas utilizadas no projeto.

## **Ficou pendente:**

Implementação do carregamento dos demais formatos: JSON, JSONL, Markdown, CSV e TXT.
Ajustes e melhorias no processamento dos arquivos PDF.
Definição dos parâmetros adequados de chunking, como tamanho dos chunks e overlap.
Estruturação e inclusão dos metadados dos documentos.
Configuração e integração do Vector DB.
Validação do processo completo de ingestão para os diferentes formatos.
Continuidade da organização e implementação das demais partes do projeto.

## **Bloqueios em aberto:**

No momento, não há bloqueios técnicos em aberto. O avanço da implementação foi limitado principalmente pelo tempo disponível no encontro e por alguns problemas iniciais relacionados à configuração e carregamento de bibliotecas. Esses problemas foram investigados durante o desenvolvimento e não impedem a continuidade da atividade.

## **Próximo passo (início do encontro 2):**

Dar continuidade à implementação da ingestão heterogênea, começando pelo carregamento e processamento dos arquivos JSON, JSONL, Markdown, CSV e TXT. Também será necessário realizar os ajustes no processamento dos PDFs e definir os parâmetros de chunking.

Após essa etapa, o foco será estruturar os metadados dos documentos e iniciar a configuração do Vector DB, avançando gradualmente para a validação do fluxo completo de ingestão.

## **Uso de assistentes de IA:**

Foram utilizados assistentes de IA principalmente como apoio durante a configuração e desenvolvimento do projeto, especialmente na investigação de erros relacionados às bibliotecas utilizadas. O auxílio foi utilizado para compreender possíveis causas dos problemas, avaliar alternativas de configuração e orientar a resolução dos erros encontrados.

---

## Encontro 2 - 2026-08-26

**Etapa:** 1 - Ingestão heterogênea, metadados e indexação vetorial

### Relato individual - [Nome do Integrante 1]

<!-- Escreva você mesmo, em primeira pessoa. O que implementou, que decisão tomou e por quê, onde travou. -->

### Relato individual - Luciano Oliveira da Costa(manhã)

Durante o período da manhã, dei continuidade à Etapa 1, concentrando o trabalho no processamento dos arquivos JSON.

Preparei o leitor responsável pelo carregamento dos arquivos JSON.
Implementei o processo de serialização dos registros em linguagem natural.
Estruturei as funções responsáveis pelo processamento dos dados.
Preparei a transformação dos registros para que cada unidade de informação seja convertida em uma única linha, conforme a estratégia definida para o chunking dos dados tabulares.
Realizei testes para validar o funcionamento do processo de leitura, processamento e serialização dos dados.

O objetivo dessa etapa foi preparar os registros JSON para que posteriormente possam ser utilizados como unidades individuais no processo de chunking e indexação vetorial.

### Relato individual - Luciano Oliveira da Costa(tarde)

### Resumo do dia (escrito em conjunto)

## **Entregamos hoje:**

## **Ficou pendente:**

## **Bloqueios em aberto:**

## **Próximo passo (início do encontro 2):**

## **Uso de assistentes de IA:**

---

## Encontro 2 - AAAA-MM-DD

**Etapa:** 2 - Busca híbrida e filtragem por metadados

### Relato individual - [Nome do Integrante 1]

### Relato individual - [Nome do Integrante 2]

### Resumo do dia (escrito em conjunto)

## **Entregamos hoje:**

## **Ficou pendente:**

## **Bloqueios em aberto:**

## **Próximo passo (início do encontro 3):**

## **Uso de assistentes de IA:**

---

## Encontro 3 - AAAA-MM-DD

**Etapa:** 3 - Síntese estruturada, evidência e guardrails de LGPD

### Relato individual - [Nome do Integrante 1]

### Relato individual - [Nome do Integrante 2]

### Resumo do dia (escrito em conjunto)

## **Entregamos hoje:**

## **Ficou pendente:**

## **Bloqueios em aberto:**

## **Próximo passo (início do encontro 4):**

## **Uso de assistentes de IA:**

---

## Encontro 4 - AAAA-MM-DD

**Etapa:** 4 - Avaliação (RAG Triad), interface e relatório

### Relato individual - [Nome do Integrante 1]

### Relato individual - [Nome do Integrante 2]

### Resumo do dia (escrito em conjunto)

## **Entregamos hoje:**

## **Ficou pendente:**

## **Bloqueios em aberto:**

## **Preparação para o Demo Day:**

## **Uso de assistentes de IA:**

---

_TIC em Trilhas · PUC-Rio · Instituto ECOA · MCTI Futuro · Softex_
