# Decisões arquiteturais

Este registro resume decisões que afetam mais de uma etapa. Mudanças futuras
devem atualizar a decisão correspondente ou adicionar uma nova entrada.

## D01 - Metadata classificada na ingestão

**Decisão:** atribuir `doc_type`, `sensitivity`, `source_file` e `chunk_id` ao
criar cada `Document`.

**Motivo:** filtragem por metadata é determinística, barata e auditável. Pedir
ao LLM que classifique sensibilidade depois da recuperação aumenta risco.

## D02 - Chunking por natureza da fonte

**Decisão:** usar registro para dados tabulares, ticket para JSONL, seção
para Markdown, parágrafo para PDF e mensagem para TXT.

**Motivo:** uma configuração única poderia quebrar registros ou misturar
unidades que não possuem relação semântica.

## D03 - Um índice FAISS por formato

**Decisão:** persistir seis índices e pesquisá-los como um conjunto lógico.

**Motivo:** simplifica inspeção e reconstrução parcial sem perder um Top-K
global. O custo é realizar busca em mais de um índice.

## D04 - Catálogo de metadata dinâmico

**Decisão:** descobrir valores nos Documents em vez de codificá-los no prompt.

**Motivo:** evita divergência entre filtros aceitos e valores existentes no
corpus. Identificadores de alta cardinalidade continuam pesquisáveis, mas
exigem tratamento cuidadoso.

## D05 - Query Analyzer com structured output

**Decisão:** usar LLM validado por `QueryAnalysis`, seguido de normalização e
validação determinísticas.

**Motivo:** linguagem natural exige interpretação, mas nenhum valor inventado
pode chegar aos retrievers.

## D06 - Busca híbrida e RRF

**Decisão:** combinar FAISS e BM25 por Reciprocal Rank Fusion.

**Motivo:** embeddings lidam melhor com significado; BM25 recupera melhor
códigos, nomes, IDs e termos exatos. RRF combina posições sem exigir calibrar
escalas incompatíveis de score.

## D07 - Filtragem FAISS explícita

**Decisão:** usar `fetch_k` adaptativo e pré-filtro exato em casos seletivos
ou de política.

**Motivo:** o filtro do FAISS no LangChain ocorre depois da busca e pode
retornar menos que `k` se o conjunto inicial de candidatos for insuficiente.

## D08 - Guardrail LGPD antes do retrieval

**Decisão:** classificar categorias confiáveis por regras determinísticas e
encerrar recusas antes de consultar o corpus.

**Motivo:** evita recuperar conteúdo extremamente sensível sem necessidade e
torna a política explicável em testes.

## D09 - Classificação de escopo híbrida

**Decisão:** aceitar sinais corporativos inequívocos por regra e usar LLM
estruturado nos demais casos.

**Motivo:** uma lista de palavras-chave isolada recusou perguntas legítimas e
aceitou termos genéricos. A combinação melhora cobertura mantendo explicação.

## D10 - Evidência construída pelo código

**Decisão:** o LLM escolhe apenas IDs recuperados; filepath e quotation vêm
do `Document`.

**Motivo:** impede fontes ou citações inventadas e permite verificar
literalmente cada quotation.

## D11 - Resposta final validada por Pydantic

**Decisão:** `RAGResponse` usa `Literal`, proíbe campos extras e valida
consistência entre recusa e evidência.

**Motivo:** o contrato precisa falhar de forma visível diante de combinações
como recusa com fonte ou resposta normal sem evidência.

## D12 - Interface sem pipeline paralelo

**Decisão:** Streamlit chama somente `RAGPipeline` construído pelo bootstrap e
não acessa FAISS, LLM ou internet diretamente.

**Motivo:** todos os caminhos de resposta precisam passar pelos mesmos
guardrails e validações.

## D13 - LLM-as-a-Judge opcional

**Decisão:** Context Relevance é determinística; Answer Relevance e
Groundedness usam juiz opcional com structured output.

**Motivo:** separa checks objetivos de avaliação qualitativa e torna custo e
falhas do juiz explícitos.

## D14 - Resultados sem conteúdo sensível

**Decisão:** persistir perguntas do benchmark, checks, scores, arquivos e IDs,
sem respostas, quotations ou justificativas textuais do juiz.

**Motivo:** artefatos de avaliação não devem se transformar em uma nova fonte
de vazamento de dados.

## D15 - Nomes de produtos normalizados deterministicamente

**Decisão:** depois do Query Analyzer, mapear nomes comerciais inequívocos
para os cinco módulos canônicos presentes no catálogo. Se mais de um produto
for citado, não forçar um filtro escalar.

**Motivo:** termos do assunto podem induzir o LLM ao módulo errado. No caso de
"Safety Stock no VendeFácil Loja", a referência explícita ao produto deve
prevalecer e selecionar `ecommerce`, sem depender da variação do modelo.
