# Manual de Configuração do VendeFácil Pay e Máquinas TEF

## 1. Visão Geral
O **VendeFácil Pay** conecta o PDV físico aos Pinpads e adquirentes de cartão através do protocolo TEF IP.

## 2. Requisitos de Rede e Portas do Firewall
Para o correto funcionamento da maquininha Pinpad acoplada ao computador do caixa:
- O agente TEF local escuta solicitações na porta **TCP 6090**.
- **Regra de Firewall Importante:** O firewall do sistema operacional (Windows Firewall ou iptables) no computador do caixa **deve permitir conexões de entrada na porta TCP 6090**.
- Caso a porta 6090 esteja bloqueada, o pagamento será aprovado na maquininha pelo adquirente, mas o PDV registrará erro de **Timeout de confirmação TEF (Erro PAY-504)** e a venda não será finalizada no caixa.
