# POLÍTICA DE SEGURANÇA DA INFORMAÇÃO E LGPD

**VendeFácil Tecnologia Ltda.**  
*Documento Interno e Externo - DPO: Gabriel Ramos (dpo@vendefacil.com.br)*

## 1. Diretrizes da LGPD (Lei Geral de Proteção de Dados - Lei 13.709/2018)
1.1. A VendeFácil Tecnologia atua como **Operadora** de dados pessoais de consumidores dos nossos clientes varejistas e **Controladora** dos dados de seus próprios colaboradores e contratantes.  
1.2. **Tratamento de Dados Sensíveis:** É estritamente proibido o armazenamento de dados sensíveis (biometria, dados de saúde, religião) ou dados financeiros confidenciais (número de cartão de crédito completo, CVV, senhas de clientes) em logs de aplicação, tickets de suporte ou arquivos de texto simples.  
1.3. **Proteção de Dados Salariais e Pessoais Internos:** Salários, remunerações, bônus, CPFs, endereços residenciais e dados bancários de colaboradores da VendeFácil são classificados como **Dados Confidenciais Nível 1 (Restrito ao RH e Diretoria)**.  
1.4. **Guardrails para Assistentes Virtuais e IAs Internas:** Qualquer sistema de inteligência artificial ou RAG interno **NÃO DEVE** expor nem responder a perguntas contendo salários, remunerações individuais, senhas ou CPFs de colaboradores ou clientes. Caso uma consulta solicite dados confidenciais de colaboradores, o sistema deve recusar a resposta e registrar alerta de segurança.

## 2. Procedimento de Resposta a Incidentes de Segurança
2.1. Havendo suspeita ou confirmação de vazamento de credenciais, chaves de API ou dados de clientes, o colaborador deve imediatamente acionar o DPO via e-mail `dpo@vendefacil.com.br` e abrir um chamado de emergência de segurança com prioridade Crítica.  
2.2. O comitê de segurança responderá em até 2 horas e notificará os afetados e a ANPD em até 48 horas se necessário.
