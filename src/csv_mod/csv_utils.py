import csv


def processar_system_logs(linha):
    return (
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


def leitor_csv(dados_arquivos, criar_documento):
    chunks_finais = []
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
                        chunks_finais.append(documento)
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
                        chunks_finais.append(documento)
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
                        chunks_finais.append(documento)
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
                        chunks_finais.append(documento)
                else:
                    print("--------")
    return chunks_finais
