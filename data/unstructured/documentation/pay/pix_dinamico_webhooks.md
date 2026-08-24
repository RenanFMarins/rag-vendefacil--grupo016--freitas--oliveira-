# Guia de Configuração do PIX Dinâmico com Webhooks - VendeFácil Pay

## 1. Funcionamento no Frente de Caixa
1. Ao selecionar a opção PIX no PDV, o VendeFácil Pay gera instantaneamente um **QR Code Dinâmico** na tela do caixa ou no display do Pinpad.
2. O QR Code contém o valor exato da venda e o identificador único da transação (txid).
3. O cliente realiza a leitura pelo aplicativo bancário de sua preferência.
4. O banco adquirente dispara um **Webhook REST** de confirmação para o serviço `vendefacil-pay`. O caixa fecha e imprime o cupom fiscal automaticamente em menos de 2 segundos.
