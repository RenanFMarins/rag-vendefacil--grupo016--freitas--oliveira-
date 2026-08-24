# Alinhamento Técnico: Aplicativo do Coletor Android para Inventário Cego
**Data:** 24 de Fevereiro de 2026  
**Participantes:** Carlos Mendes (Tech Lead), Lucas Ferreira (PM), Igor Oliveira (Mobile/Pay)

## Definições da Sprint:
1. **Cache Local de Bipagens:** O aplicativo Android para coletores Zebra/Honeywell passará a salvar as contagens de estoque em banco SQLite local antes de enviar a requisição REST para o VendeFácil Estoque.
2. **Prevenção de Timeout em Wi-Fi Instável:** O app fará reenvios automáticos em lotes de 50 SKUs, eliminando o erro de desconexão relatado por distribuidores e empórios.
