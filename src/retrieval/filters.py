from rapidfuzz import fuzz


def extract_metadata(documentos):

    CAMPOS_NAO_FILTRAVEIS = {
        "chunk_id",
        "store_id",
        "customer_id",
        "product_id",
        "error_code"
    }

    filtros = {}

    for documento in documentos:

        for campo, valor in documento.metadata.items():

            if campo in CAMPOS_NAO_FILTRAVEIS:
                continue

            if campo not in filtros:
                filtros[campo] = set()

            if isinstance(valor, list):
                filtros[campo].update(str(item).lower() for item in valor)
            else:
                filtros[campo].add(str(valor).lower())

    return filtros


def extrair_filtros_manuais(pergunta):

    pergunta = pergunta.lower()

    filtros = {}

    tipos_documento = {
        "tickets": "ticket", "ticket": "ticket", "chamado": "ticket", "chamados": "ticket",
        "clientes": "customer", "cliente": "customer",
        "lojas": "store", "loja": "store",
        "vendas": "sale", "venda": "sale",
        "produtos": "product", "produto": "product",
        "funcionários": "employee", "funcionário": "employee", "funcionario": "employee",
        "logs": "log", "log": "log", "servidor": "log",
        "manual": "manual", "manuais": "manual", "documentação": "manual", "documentacao": "manual", "módulo": "manual", "modulo": "manual",
        "ata": "ata", "atas": "ata", "reunião": "ata", "reuniao": "ata",
        "política": "policy", "politica": "policy", "norma": "policy",
        "email": "email", "e-mail": "email",
    }

    estados = {"acre": "AC",
               "alagoas": "AL",
               "amapá": "AP",
               "amazonas": "AM",
               "bahia": "BA",
               "ceará": "CE",
               "distrito federal": "DF",
               "espírito santo": "ES",
               "goiás": "GO",
               "maranhão": "MA",
               "mato grosso": "MT",
               "mato grosso do sul": "MS",
               "minas gerais": "MG",
               "pará": "PA",
               "paraíba": "PB",
               "paraná": "PR",
               "pernambuco": "PE",
               "piauí": "PI",
               "rio de janeiro": "RJ",
               "rio grande do norte": "RN",
               "rio grande do sul": "RS",
               "rondônia": "RO",
               "roraima": "RR",
               "santa catarina": "SC",
               "são paulo": "SP",
               "sergipe": "SE",
               "tocantins": "TO"}

    modulos = {
        "pdv": "pdv", "ponto de venda": "pdv",
        "estoque": "estoque", "inventário": "estoque", "inventario": "estoque",
        "pay": "pay", "pagamento": "pay",
        "analytics": "analytics", "análise": "analytics", "analise de dados": "analytics",
        "ecommerce": "ecommerce", "e-commerce": "ecommerce", "loja virtual": "ecommerce",
    }

    # documentos
    for termo, valor in tipos_documento.items():
        if termo in pergunta:
            filtros["doc_type"] = valor
            break

    # estado
    for termo, valor in estados.items():
        if termo in pergunta:
            filtros["state"] = valor
            break

    # modulo
    for termo, valor in modulos.items():
        if termo in pergunta:
            filtros["active_modules"] = valor
            break

    return filtros


def extrair_filtros_automatico(pergunta, vocabulario, limiar=75):

    pergunta = pergunta.lower()
    filtros = {}

    for campo, valores in vocabulario.items():

        if campo == "state":
            continue

        if not valores:
            continue

        melhor_resultado = None
        melhor_score = 0

        for valor in valores:

            valor = str(valor).lower().strip()

            if len(valor) < 3:
                continue

            if valor in pergunta.split():
                score = 100

            else:
                score = fuzz.token_set_ratio(
                    valor,
                    pergunta
                )

            if score > melhor_score:
                melhor_score = score
                melhor_resultado = valor

        if melhor_score >= limiar:
            filtros[campo] = melhor_resultado

    return filtros


def combinar_filtros(filtros_manuais, filtros_automaticos):
    filtros_combinados = {}

    filtros_combinados = filtros_automaticos | filtros_manuais

    return filtros_combinados


def filtrar_documento(documentos, filtros):

    documentos_filtrados = []

    for documento in documentos:

        metadata = documento.metadata

        atende_filtros = True

        if "doc_type" in filtros and "doc_type" in metadata:
            if metadata.get("doc_type") != filtros["doc_type"]:
                continue

        for campo, valor_esperado in filtros.items():
            if campo == "doc_type":
                continue  # já tratado no passo 1

            if campo not in metadata:
                continue  # campo não existe nesse tipo de doc -> ignora

            valor_doc = metadata.get(campo)

            if isinstance(valor_doc, list):
                termo = str(valor_esperado).lower()
                encontrou = any(termo in str(item).lower()
                                for item in valor_doc)
                if not encontrou:
                    atende_filtros = False
                    break
            else:
                if str(valor_doc).lower() != str(valor_esperado).lower():
                    atende_filtros = False
                    break

        if atende_filtros:
            documentos_filtrados.append(documento)

    return documentos_filtrados


def processar_filtros(pergunta, documentos):

    vocabulario = extract_metadata(documentos)

    filtros_manuais = extrair_filtros_manuais(pergunta)

    filtros_automaticos = extrair_filtros_automatico(pergunta, vocabulario)

    filtro_combinado = combinar_filtros(filtros_manuais, filtros_automaticos)

    documentos_filtrados = filtrar_documento(documentos, filtro_combinado)

    return {
        "filtros_manuais": filtros_manuais,
        "filtros_automaticos": filtros_automaticos,
        "filtro_combinado": filtro_combinado,
        "documentos_filtrados": documentos_filtrados
    }
