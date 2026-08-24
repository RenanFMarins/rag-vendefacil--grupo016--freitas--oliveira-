# Relatório de Testes de Estresse: Contingência Offline do VendeFácil PDV
**Data:** 06 de Fevereiro de 2026  
**Participantes:** Beatriz Lima (Frontend), Carlos Mendes (Tech Lead), Diego Alves (Suporte)

## Resultados dos Testes:
1. **Volume Máximo Local:** O banco SQLite local no terminal de caixa suportou o acúmulo de até 5.000 NFC-e em contingência offline sem perda de performance na bipagem.
2. **Sincronização em Lote:** A transmissão de notas acumuladas após o retorno da internet levou 42 segundos para 1.000 notas, sem rejeição por duplicidade de numeração.
