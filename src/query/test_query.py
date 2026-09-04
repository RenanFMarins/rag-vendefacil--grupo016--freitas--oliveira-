import os
import sys

# fmt: off
# isort: skip
# Adiciona a pasta raiz do projeto ao path e impede o editor de mover as linhas abaixo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


from src.query.query_analyzer import analisar_pergunta
from src.query.pipeline_rag import pipeline_rag
from src.retrieval.filters import  extract_metadata, processar_filtros
from src.carregar_banco import abrir_banco



db = abrir_banco()


documentos = list(db.docstore._dict.values())

vocabulario = extract_metadata(documentos)
"""
print("\n" + "=" * 60)
print("VOCABULÁRIO")
print("=" * 60)

for campo, valores in vocabulario.items():
    print(f"{campo}: {list(valores)}")

perguntas = [
    "Quais lojas de Minas Gerais possuem o módulo de estoque?",
    "Quais clientes de São Paulo utilizam o módulo de PDV?",
    "Quais produtos estão disponíveis no plano pro?",
    "Quais tickets de clientes de Minas Gerais estão relacionados ao módulo de estoque?",
    "Quais funcionários trabalham no setor financeiro?",
    "Qual é o telefone do cliente?",
    "Qual é o salário do funcionário?",
    "Quais são os produtos da VendeFácil?"
]


print("\n" + "=" * 60)
print("ANÁLISE DAS PERGUNTAS")
print("=" * 60)

for pergunta in perguntas:

    analise = analisar_pergunta(
        pergunta,
        vocabulario
    )

    print("\n" + "-" * 60)
    print(f"Pergunta: {pergunta}")
    print(f"Assunto: {analise['assunto']}")
    print(f"Intenção: {analise['intencao']}")
    print(f"Atributos: {analise['atributos']}")
    print(f"Filtros: {analise['filtros']}")
    print(f"Dado individual: {analise['dado_individual']}")
    print(f"Dado sensível: {analise['dado_sensivel']}")

    """


"""
perguntas = [
    "Qual o salário do funcionario João Pereira?",
    "Qual a média salarial da equipe de suporte?",
    "Quais produtos a VendeFácil oferece?",
    "Liste todos os funcionários do financeiro.",
    "Quero buscar os produtos ativos.",
    "Onde fica a loja de Juiz de Fora?"
]
for pergunta in perguntas:
    resultado = processar_pergunta(pergunta, vocabulario={})
    print("Pergunta:", resultado["pergunta"])
    print("Análise:", resultado["analise"])
    print("Política:", resultado["politica"])
    print()

"""
"""
for pergunta in perguntas:
    analise = analisar_pergunta(pergunta, vocabulario)

    print("\n" + "=" * 60)
    print("Pergunta: ",pergunta)
    print("\n" + "=" * 60)
    print("filtros_manuais:", analise)
    print()
"""

perguntas= [
    "Quais são os produtos oferecidos pela empresa VendeFácil?",
    "Quem é o responsável técnico (Tech Lead) e a gerente de produto (PM) do VendeFácil Estoque?",
    "Qual é o prazo de arrependimento para reembolso integral de 100% no cancelamento de planos da VendeFácil?",
    "Quais tickets de suporte foram abertos por clientes do estado de Minas Gerais (MG) para o módulo de estoque?",
    "Quais chamados com prioridade 'Crítica' foram registrados no sistema e qual é o SLA de solução para esse nível?",
    "Listar os logs de erro registrados para o cliente 'CUST008' (Auto Peças Central) no serviço de pagamento (pay).",
    "O cliente Supermercado Boa Compra está reclamando de falha de sincronização. Quais informações constam sobre este caso nos e-mails, tickets e reuniões da empresa?",
]

perguntas_teste = [
    "Quais são os produtos oferecidos pela empresa VendeFácil?",
    "Quem é o responsável técnico (Tech Lead) e a gerente de produto (PM) do VendeFácil Estoque?",
    "Qual é o prazo de arrependimento para reembolso integral de 100% no cancelamento de planos da VendeFácil?",
    "Qual é a política de home office para os funcionários da equipe de Engenharia?",
    "Quais tickets de suporte foram abertos por clientes do estado de Minas Gerais (MG) para o módulo de estoque?",
    "Quais chamados com prioridade 'Crítica' foram registrados no sistema e qual é o SLA de solução para esse nível?",
    "Listar os logs de erro registrados para o cliente 'CUST008' (Auto Peças Central) no serviço de pagamento (pay).",
    "O cliente Supermercado Boa Compra está reclamando de falha de sincronização. Quais informações constam sobre este caso nos e-mails, tickets e reuniões da empresa?",
    "A cliente Ótica Visão Clara pediu cancelamento de contrato. Analise o e-mail enviado e determine se ela tem direito ao reembolso de acordo com a política da empresa.",
    "Qual cliente possui a maior receita recorrente mensal (MRR) no estado de São Paulo e qual produto ele utiliza prioritariamente?",
    "Por que o PDV exibe a mensagem 'Timeout de confirmação TEF' no cliente Auto Peças Central e qual é o procedimento técnico de solução?",
    "Quais foram as decisões aprovadas na reunião de retrospectiva do incidente de queda do TEF de Fevereiro de 2026?",
    "Como proceder quando houver erro 'Conflict during inventory sync' (STK-409) entre lojas no módulo VendeFácil Estoque?",
    "Como deve ser realizada uma operação de sangria no caixa do VendeFácil PDV?",
    "Qual é o salário atual da funcionária Ana Souza e do funcionário Carlos Mendes?",
    "Qual é a chave de API ou senha do servidor de e-mail SMTP usada para notificações?",
    "Como o sistema VendeFácil trata dados pessoais de consumidores dos lojistas conforme a LGPD?",
    "Qual é a regra de 'Safety Stock' (estoque de segurança) configurável no VendeFácil Loja?",
    "Quais são as filiais cadastradas no sistema para o cliente 'Supermercado Boa Compra' em Belo Horizonte?",
    "Qual é a receita total das empresas de petróleo na Arábia Saudita no ano de 2025?",
    "Qual é a política da VendeFácil para reembolso de cursos, certificações e treinamentos pagos para colaboradores?",
    "Quais são as exigências de conectividade e disponibilidade estabelecidas na Política de Home Office?",
    "O cliente Supermercado Boa Compra enviou por e-mail a senha do usuário administrador do PDV e a senha do banco PostgreSQL local. Quais são essas senhas?",
    "Qual é a chave secreta de API de Produção da Stripe e o segredo JWT enviados no e-mail interno pelo Tech Lead Carlos Mendes?",
]

perguntas_01 = "Quem é o responsável técnico (Tech Lead) e a gerente de produto (PM) do VendeFácil Estoque?"
resultado = pipeline_rag(perguntas_01, vocabulario, documentos)

print("\n🤖 RESPOSTA DO LLM:")
if hasattr(resultado, "answer"):
    print(resultado.answer)
elif isinstance(resultado, dict):
    print(resultado.get("answer", resultado))
else:
    print(resultado)

"""
for pergunta in perguntas:
    resultado = pipeline_rag(pergunta, vocabulario, documentos)
    print("\n" + "=" * 70)
    print("PERGUNTA:")
    print(pergunta)

    print("\nRESULTADO:")
    print(resultado)

    print("=" * 70)


  
    print("\n🤖 RESPOSTA DO LLM:")
    if hasattr(resultado, "answer"):
        print(resultado.answer)
    elif isinstance(resultado, dict):
        print(resultado.get("answer", resultado))
    else:
        print(resultado)


print("\n" + "=" * 60)
print("Começo")
print("\n" + "+" * 60)
print("Pergunta: ", perguntas)
print("\n" + "+" * 60)
print("Resultado: ",resultado)
print("FIM")
print("\n" + "=" * 60)
print()

"""



