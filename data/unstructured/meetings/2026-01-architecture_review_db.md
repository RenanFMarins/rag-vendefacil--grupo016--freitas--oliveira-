# Ata de Reunião: Arquitetura de Banco de Dados e Cache Redis
**Data:** 22 de Janeiro de 2026  
**Participantes:** Carlos Mendes (Tech Lead), Fernanda Rocha (DevOps), Igor Oliveira (Tech Lead Pay)

## Pauta e Resultados:
1. **Invalidação de Cache no VendeFácil Analytics:** Reduzido o TTL do Redis de 1 hora para 5 minutos para os dashboards de vendas em tempo real.
2. **Otimização de Índices SQLite nos Caixa:** Criado índice composto na tabela `cupons_fiscais` nos terminais PDV para acelerar o fechamento de caixa de 45s para 3s.
