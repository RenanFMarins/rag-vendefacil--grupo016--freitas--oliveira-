# Relatório de Simulação de Incidentes (Game Day): Resiliência VendeFácil Pay
**Data:** 20 de Março de 2026  
**Participantes:** Fernanda Rocha (DevOps), Igor Oliveira (Tech Lead Pay), Equipe de Plantão

## Cenários Simulados:
1. **Simulação de Queda de Link da Adquirente:** O sistema acionou com sucesso o adquirente de backup em menos de 3 segundos, mantendo as aprovações de cartão sem queda de vendas no PDV.
2. **Simulação de Queda do Webhook de PIX:** O serviço acumulou as confirmações na fila DLQ e processou todas as notas assim que o serviço foi restaurado.
