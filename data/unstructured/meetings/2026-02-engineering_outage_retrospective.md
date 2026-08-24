# Retrospectiva de Incidente (Outage Post-Mortem): Queda do Serviço de TEF/Pay
**Data:** 12 de Fevereiro de 2026  
**Participantes:** Fernanda Rocha (DevOps), Igor Oliveira (Tech Lead Pay), Carlos Mendes (Tech Lead), Diego Alves (Suporte N2)

## Resumo do Incidente:
No dia 08 de Fevereiro de 2026, entre 16:30 e 18:15, 35% dos clientes com VendeFácil Pay enfrentaram lentidão e erros de timeout em transações de cartão de crédito no PDV.

## Causa Raiz:
Uma atualização de regras de firewall nos computadores locais dos clientes (caixas Windows) associada a um vazamento de conexões abertas (socket leak) na versão v2.8.1 do agente TEF.

## Ações e Decisões Aprovadas:
1. **Release de Correção v2.8.2:** Publicar patch emergencial para fechar sockets ociosos no agente local do PDV. (Responsável: Igor Oliveira - Concluído).
2. **Documentação de Firewall:** Atualizar expressamente o manual do VendeFácil Pay destacando a **obrigatoriedade de liberação da porta TCP 6090** nos firewalls locais dos clientes. (Responsável: Diego Alves - Concluído).
3. **Módulo de Diagnóstico Automático:** Desenvolver no VendeFácil PDV um botão de "Testar Conexão TEF e Porta 6090" para acelerar o suporte técnico. (Responsável: Beatriz Lima - Entrega em 25/02).
