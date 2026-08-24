# Guia Técnico: Sincronização de Estoque Multiloja e Inventário

## 1. Arquitetura de Sincronização (Matriz x Filiais)
O módulo **VendeFácil Estoque** utiliza uma arquitetura de sincronização híbrida (WebSocket + Webhook REST com fila RabbitMQ).
- **Entrada de NF-e na Matriz:** Quando uma nota fiscal de entrada é importada na Matriz, o estoque central atualiza os saldos e publica mensagens de atualização para os nós de borda das filiais.
- **Transferência entre Filiais:** Ao gerar uma Guia de Transferência no painel Web, o saldo é bloqueado na filial de origem ("Estoque em Trânsito") até que o recebimento seja confirmado fisicamente pela filial de destino.

## 2. Resolução de Conflitos e Travamentos (Locks)
Em redes com instabilidade de internet ou volumes altos de venda simultânea em múltiplas filiais:
- Caso ocorra a mensagem **"Conflict during inventory sync" (Erro STK-409)**, o motivo é um *table lock* temporário no banco de dados distribuído na tabela `inventory_stock`.
- **Procedimento de Resolução:**
  1. Verifique no menu `Configurações -> Rede -> Status da Sincronização` se há pacotes pendentes na fila local.
  2. Clique em **"Forçar Reconciliação de Estoque"**.
  3. Caso o saldo permaneça divergente por mais de 15 minutos, reinicie o serviço `vendefacil-estoque-agent` no servidor local da loja ou force um sync cego via comando no terminal.

## 3. Inventário Cego com Coletor de Dados
- Para evitar fraudes ou contagens viciadas, o VendeFácil Estoque suporta a função de **Inventário Cego**, onde o operador não visualiza a quantidade esperada no sistema durante a bipagem.
- Os arquivos do coletor devem ser exportados no formato `.CSV` codificado em `UTF-8` com os campos: `SKU;QUANTIDADE;LOTE;VALIDADE`.
