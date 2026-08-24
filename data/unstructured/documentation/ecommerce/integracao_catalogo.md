# Integração de Catálogo e Estoque - VendeFácil Loja

## 1. Sincronização Omnichannel
O **VendeFácil Loja** conecta o catálogo de produtos da loja física com canais de e-commerce e e-marketplaces (Mercado Livre, Shopee, WhatsApp Commerce).
- **Reserva de Estoque:** Ao receber um pedido online, o saldo do item no VendeFácil Estoque é reservado em tempo real por até 2 horas até a aprovação do pagamento.
- **Regra de Segurança de Estoque (Safety Stock):** É possível configurar uma margem de segurança de estoque (ex: manter 2 unidades reservadas apenas para a loja física) para evitar venda duplicada de itens com baixo giro.

## 2. Requisitos de Imagens e Feed XML
- Imagens do catálogo devem possuir resolução mínima de `800x800px`, no formato JPEG ou WebP, com tamanho máximo de 2MB.
- As integrações de catálogo necessitam do cadastro prévio do código EAN/GTIN e NCM correto para emissão da nota fiscal eletrônica.
