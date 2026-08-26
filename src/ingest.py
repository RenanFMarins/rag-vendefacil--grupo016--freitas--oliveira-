from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
import datetime
import csv
import json
import glob
import os

#pegar os caminhos dos arquivos 
arquivos_recebidos = glob.glob(os.path.join("../data/", "**/","*"), recursive=True)
dados_arquivos = []

#
for doc in arquivos_recebidos:
    if os.path.isdir(doc):
     continue
         
    if "customers" in doc.lower():
        doc_type = "customer"
        sensitivity = "interno"
    elif "employees" in doc.lower():
          doc_type = "employee"
          sensitivity = "restrito"
    elif "products" in doc.lower():
          doc_type = "product"
          sensitivity = "publico" 
    elif "stores" in doc.lower():
               doc_type = "store"
               sensitivity = "publico" 
    elif "tickets" in doc.lower():
               doc_type = "ticket"
               sensitivity = "interno" 
    elif "logs" in doc.lower():
               doc_type = "log"
               sensitivity = "interno" 
    elif "documentation" in doc.lower():
               doc_type = "manual"
               sensitivity = "publico" 
    elif "policies" in doc.lower():
               doc_type = "policy"
               sensitivity = "interno"
    elif "meetings" in doc.lower():
               doc_type = "email"
               sensitivity = "interno"
    elif "emails" in doc.lower():
               doc_type = "email"
               sensitivity = "restrito" 
    else: 
         doc_type = "outro"
         sensitivity = "outro"     

    dados_arquivos.append({"caminho":doc, "doc_type":doc_type, "sensitivity":sensitivity})

#serializado em linguagem natural
def processar_produtos(dados_json):
       empresa = dados_json["company"]
       data_atualizacao = dados_json["last_updated"]

       arquivo_serializado = []

       for nome_plano, detalhes in dados_json["pricing_plans"].items():
              texto_plano = (
                     f"Na empresa {empresa} (aualizado em {data_atualizacao}),"
                     f"o plano {nome_plano} custa R$ {detalhes["monthly_fee_brl"]} por mês."
                     f"Descrição: {detalhes["description"]}"
              )
              arquivo_serializado.append(texto_plano)

       for produto in dados_json["products"]:
              texto_produto =(
                    f"A empresa {empresa} (atualizado em {data_atualizacao}) oferece o produto {produto["name"]} "
                    f"(ID: {produto['product_id']}) da categoria {produto['category']}. "
                    f"Descrição: {produto['description']}"
              )
              arquivo_serializado.append(texto_produto)

       return arquivo_serializado


def processar_stores(dados_json):
     arquivo_serializado = []
     for loja in dados_json["network_stores"]:
          modulo = ", ".join(loja["active_modules"])
          texto_base = (
               f"A loja {loja["store_name"]} (ID da loja: {loja["store_id"]}),"
               f"pertencente à empresa {loja["company_name"]} (ID do cliente: {loja["customer_id"]}),"
               f"está localizada na cidade de {loja["city"]}, no estado de {loja["state"]}."
               f"Esta unidade possui {loja["pos_terminals_count"]} terminais de PDV em operação e conta com os seguintes módulos ativos no sistema: {modulo}."
          )
          arquivo_serializado.append(texto_base)
     return arquivo_serializado

def processar_system_logs(linha):
     return  (
        f"Em {linha['timestamp']}, o sistema registrou um log de nível {linha['level']} "
        f"no serviço {linha['service']} (módulo: {linha['module']}). "
        f"O evento do tipo {linha['event']} (Código de erro: {linha['error_code']}) "
        f"gerou a mensagem: {linha['message']}."
     )

def processar_customers(linha):
     return (
            f"O cliente {linha['company_name']} (ID: {linha['customer_id']}, CNPJ: {linha['cnpj']}) "
            f"está localizado em {linha['city']} - {linha['state']} e atua no segmento de {linha['segment']}. "
            f"Atualmente, a empresa possui o status {linha['status']}, utiliza o plano {linha['plan']} "
            f"com o produto principal {linha['main_product']} e gera um MRR de R$ {linha['mrr']}. "
            f"O e-mail de contato é {linha['contact_email']}."
        )

def processar_employees(linha):
     return (
            f"O colaborador {linha['name']} (ID: {linha['id']}) atua no departamento de {linha['department']} "
            f"no cargo de {linha['role']}. Foi contratado em {linha['hire_date']}, possui o status {linha['status']} "
            f"e o e-mail de contato profissional é {linha['email']}. O salário atual do funcionário é de R$ {linha['salary']}."
        )
def processar_sales(linha):
     return (
            f"A venda de ID {linha['sale_id']} foi registrada na data de {linha['date']} "
                        f"no estabelecimento {linha['store_name']} (ID da loja: {linha['store_id']}), "
                        f"localizado em {linha['city']} - {linha['state']}. O cliente comprador foi "
                        f"{linha['company_name']} (ID do cliente: {linha['customer_id']}). "
                        f"O produto adquirido foi o {linha['product_name']} (ID do produto: {linha['product_id']}) "
                        f"pelo valor de R$ {linha['amount_brl']}, utilizando o método de pagamento {linha['payment_method']} "
                        f"no terminal {linha['pos_terminal']}. O status atual desta transação é: {linha['status']}."
        )

def criar_documento(texto, caminho, doc_type, sensitivity, chunk_id):
    return Document(
        page_content=texto,
        metadata={
            "source_file": caminho,
            "doc_type": doc_type,
            "chunk_id": chunk_id,
            "sensitivity": sensitivity
        }
    )

def leitor_json():
     for dados in dados_arquivos:
         caminho = dados["caminho"]
         if caminho.endswith(".json"):
           with open(caminho, "r", encoding="utf-8") as j:
               registro = json.load(j)
               if "products" in caminho.lower():
                    produtos_json = processar_produtos(registro)
               elif "stores" in caminho.lower():
                    stores_json = processar_stores(registro)                   
               else:
                    print("-------------") 
                                                
#leitor_json()

def leitor_csv():
    chunks_system_logs = []
    chunks_customers = []
    chunks_employees = []
    chunks_sales = []
    for dados in dados_arquivos:
          caminho = dados["caminho"]  
          if caminho.endswith(".csv"):
             print(caminho)
             with open(caminho, "r", encoding='utf-8') as c:
                  registro = csv.DictReader(c)
                  if "system_logs" in caminho.lower(): 
                    for linha in registro:
                       system_logs = processar_system_logs(linha)
                       documento = criar_documento(
                                   texto=system_logs,
                                   caminho=caminho,
                                   doc_type=dados["doc_type"],
                                   sensitivity=dados["sensitivity"],
                                   chunk_id=f"system_logs_{linha["customer_id"]}"
                                   )
                       chunks_system_logs.append(documento)                       
                  elif "customers" in caminho.lower():
                        for linha in registro: 
                         customers = processar_customers(linha)
                         documento = criar_documento(
                                   texto=customers,
                                   caminho=caminho,
                                   doc_type=dados["doc_type"],
                                   sensitivity=dados["sensitivity"],
                                   chunk_id=f"customers_{linha["customer_id"]}"
                                   )
                         chunks_customers.append(documento)  
                  elif "employees" in caminho.lower():
                        for linha in registro: 
                            employees = processar_employees(linha)
                            documento = criar_documento(
                                   texto=employees,
                                   caminho=caminho,
                                   doc_type=dados["doc_type"],
                                   sensitivity=dados["sensitivity"],
                                   chunk_id=f"employees_{linha["id"]}"
                                   )
                            chunks_employees.append(documento)  
                  elif "sales" in caminho.lower():
                         for linha in registro: 
                              sales = processar_sales(linha)
                              documento = criar_documento(
                                   texto=sales,
                                   caminho=caminho,
                                   doc_type=dados["doc_type"],
                                   sensitivity=dados["sensitivity"],
                                   chunk_id=f"sales_{linha["sale_id"]}"
                                   )
                              chunks_sales.append(documento)
                  else:
                         print("--------")  
    return  chunks_employees 
                                                 
leitor_csv()
         






        

    