# Manual do Operador - VendeFácil PDV (v3.4)

## 1. Operações de Frente de Caixa

### 1.1 Sangria e Suprimento
A **sangria** é a retirada de valores do gaveteiro do caixa por medida de segurança quando o limite preestabelecido de numerário (dinheiro) é atingido.
- **Passo a passo para Sangria:**
  1. No menu principal do PDV, pressione `F8` ou clique em `Caixa -> Sangria`.
  2. Informe o valor a ser retirado e o motivo (ex: "Sangria periódica para cofre").
  3. Insira a senha do supervisor/gerente do caixa.
  4. O PDV emitirá o comprovante de sangria em duas vias. A primeira via deve ser assinada e depositada no gaveteiro.

### 1.2 Abertura e Fechamento de Caixa
- Para abrir o caixa, o operador deve realizar o **Suprimento Inicial** (fundo de troco).
- No encerramento do turno, selecione `Caixa -> Fechamento`. O sistema solicitará a contagem cega das formas de pagamento (Dinheiro, Cartão Crédito/Débito, PIX, VOUCHER).
- Diferenças entre o valor contado e o valor lógico do sistema geram lançamentos automatizados de "Quebra de Caixa".

### 1.3 Alçada de Descontos e Perfis de Acesso
- Descontos de até 5% podem ser aplicados livremente pelo operador de caixa.
- Descontos acima de 5% ou cancelamento de itens já bipados exigem autorização presencial ou chave de gerência.

## 2. Emissão Fiscal e Contingência Offline
- Quando a conexão com a SEFAZ do estado ficar indisponível, o VendeFácil PDV altera automaticamente para o modo **Contingência Offline (NFC-e / SAT)**.
- Todas as vendas realizadas offline são armazenadas em banco de dados seguro local (SQLite criptografado) no terminal.
- **Importante:** Assim que a conexão for reestabelecida, o operador deve acionar a rotina `Fiscal -> Sincronizar Contingências` para transmitir as notas acumuladas dentro do prazo legal de 24 horas.
