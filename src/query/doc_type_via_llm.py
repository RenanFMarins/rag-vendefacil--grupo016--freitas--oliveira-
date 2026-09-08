import json


def classificar_doc_type_via_llm(pergunta, cliente_groq) -> list:
    """
    Usa a inteligência da LLM para inferir semanticamente quais tabelas/tipos de
    documento contêm a resposta, aceitando sinônimos e intenções implícitas.
    """
    prompt_sistema = """Você é o classificador de escopo do RAG VendeFácil. 
    Sua única tarefa é analisar a pergunta do usuário e retornar uma lista em formato JSON com os tipos de documentos ('doc_type') necessários para responder à dúvida.

    Os tipos válidos são:
    - 'policy': Regras, normas internas, termos, chaves de API, credenciais, segurança, tokens, diretrizes.
    - 'employee': Pessoas, cargos, papéis, organograma, contratação, liderança (Ex: Tech Lead, PM, desenvolvedor, gerente).
    - 'product': Especificações de módulos, funcionalidades do sistema, dados técnicos de produtos.
    - 'manual': Documentações de ajuda, manuais do usuário, guias de módulos (Ex: módulo de estoque, manual de vendas).
    - 'customer': Dados de clientes, segmentos de empresas, planos contratados, status de contas.
    - 'sale': Histórico de vendas, transações, faturamento, notas, lojas e métodos de pagamento.
    - 'log': Mensagens de erro de servidores, logs de sistema, falhas técnicas, timestamps.

    Regra estrita: Retorne APENAS o JSON no formato: {"doc_types": ["tipo1", "tipo2"]}. Não adicione nenhuma explicação textual externa."""

    try:
        resposta = cliente_groq.chat.completions.create(
            model='openai/gpt-oss-120b',  # Substitua pelo modelo exato que usa na Groq
            temperature=0,
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": f"Pergunta: '{pergunta}'"}
            ],
            response_format={"type": "json_object"}
        )
        dados = json.loads(resposta.choices[0].message.content)
        print(dados)
        return dados.get("doc_types", [])
    except Exception as e:
        print(f"Erro na classificação da LLM, usando fallback geral: {e}")
        # Se a LLM falhar por algum motivo, libera todos os escopos para não travar a busca
        return ["product", "employee", "policy", "manual", "customer", "sale", "log"]
