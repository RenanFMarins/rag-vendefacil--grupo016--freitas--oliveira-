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

    tipos_documento = {"tickets": "ticket", "ticket": "ticket", "clientes": "customer", "cliente": "customer", "lojas": "store", "loja": "store",
                       "vendas": "sale", "venda": "sale", "produtos": "product", "produto": "product", "funcionários": "employee", "funcionários": "employee"}

    estados = {"minas gerais": "MG", "minas": "MG",
               "rio de janeiro": "RJ", "são paulo": "SP", "espírito santo": "ES"}

    modulos = {"pdv": "pdv", "ponto de venda": "pdv", "estoque": "estoque",
               "inventário": "estoque", "pay": "pay", "pagamento": "pay"}

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
            filtros["module"] = valor
            break
    print(filtros)

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

        if "doc_type" in filtros:
            if metadata.get("doc_type") != filtros["doc_type"]:
                atende_filtros = False

        if "state" in filtros:
            if metadata.get('state') != filtros["state"]:
                atende_filtros = False

        if "module" in filtros:
            modulos_do_doc = metadata.get(
                'active_modules') or metadata.get('module') or []
            if isinstance(modulos_do_doc, list):

                termo_procurado = filtros['module'].lower()
                encontrou_modulo = any(termo_procurado in mod.lower()
                                       for mod in modulos_do_doc)
                if not encontrou_modulo:
                    atende_filtros = False
            else:

                if filtros['module'].lower() not in str(modulos_do_doc).lower():
                    atende_filtros = False

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
