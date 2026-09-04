import os
import sys

# fmt: off
# isort: skip
# Adiciona a pasta raiz do projeto ao path e impede o editor de mover as linhas abaixo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


from src.query.query_analyzer import processar_pergunta
from src.retrieval.filters import  extract_metadata
from src.carregar_banco import abrir_banco




db = abrir_banco()


documentos = list(db.docstore._dict.values())[:10]

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