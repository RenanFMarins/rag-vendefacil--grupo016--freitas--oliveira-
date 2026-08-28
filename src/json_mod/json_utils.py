import json
import os


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
        arquivo_serializado.append({
            "texto": texto_plano,
            "product_id": nome_plano.lower()
        })

    for produto in dados_json["products"]:
        texto_produto = (
            f"A empresa {empresa} (atualizado em {data_atualizacao}) oferece o produto {produto["name"]} "
            f"(ID: {produto['product_id']}) da categoria {produto['category']}. "
            f"Descrição: {produto['description']}"
        )
        arquivo_serializado.append({
            "texto": texto_produto,
            "product_id": produto["product_id"],
            "category": produto['category'],
            "product_manager": produto['product_manager'],
            "tech_lead": produto['tech_lead'],

        })

    return arquivo_serializado


def processar_stores(dados_json):
    arquivo_serializado = []
    for loja in dados_json["network_stores"]:
        modulo = ", ".join(loja["active_modules"])

        texto_loja = (
            f"A loja {loja["store_name"]} (ID da loja: {loja["store_id"]}),"
            f"pertencente à empresa {loja["company_name"]} (ID do cliente: {loja["customer_id"]}),"
            f"está localizada na cidade de {loja["city"]}, no estado de {loja["state"]}."
            f"Esta unidade possui {loja["pos_terminals_count"]} terminais de PDV em operação e conta com os seguintes módulos ativos no sistema: {modulo}."
        )
        arquivo_serializado.append({
            "texto": texto_loja,
            "store_id": loja["store_id"],
            "customer_id": loja["customer_id"],
            "state": loja["state"],
            "city": loja["city"],
            "active_modules": loja["active_modules"],

        })

    return arquivo_serializado


def leitor_json(dados_arquivos, criar_documento):
    chunks_finais = []
    for dados in dados_arquivos:
        caminho = dados["caminho"]
        nome_arquivo = os.path.basename(caminho)
        if caminho.endswith(".json"):
            with open(caminho, "r", encoding="utf-8") as j:
                registro = json.load(j)

                if "products" in caminho.lower():
                    produtos_json = processar_produtos(registro)
                    for linha in produtos_json:
                        documento = criar_documento(
                            texto=linha['texto'],
                            caminho=nome_arquivo,
                            doc_type=dados["doc_type"],
                            sensitivity=dados["sensitivity"],
                            chunk_id=f"products_{linha['product_id']}",
                            product_id=linha['product_id'],
                            category=linha['category'] if 'category' in linha else "Sem categoria",
                            product_manager=linha['product_manager'] if 'product_manager' in linha else "Não Informado",
                            tech_lead=linha['tech_lead'] if 'tech_lead' in linha else "Não Informado",
                        )
                        chunks_finais.append(documento)

                elif "stores" in caminho.lower():
                    stores_json = processar_stores(registro)
                    for linha in stores_json:
                        documento = criar_documento(
                            texto=linha['texto'],
                            caminho=nome_arquivo,
                            doc_type=dados["doc_type"],
                            sensitivity=dados["sensitivity"],
                            chunk_id=f"stores_{linha['store_id']}",
                            store_id=linha['store_id'],
                            customer_id=linha['customer_id'],
                            state=linha['state'],
                            city=linha['city'],
                            active_modules=linha['active_modules']
                        )
                        chunks_finais.append(documento)
                else:
                    print(f"Arquivo JSON não reconhecido: {caminho}")
    return chunks_finais
