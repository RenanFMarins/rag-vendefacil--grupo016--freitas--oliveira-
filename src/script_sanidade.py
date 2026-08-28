from ingest import abrir_banco


def script_sanidade():
    db = abrir_banco()

    total_chunks = len(db.docstore._dict)

    print("=" * 60)
    print("SANIDADE DO BANCO FAISS")
    print("=" * 60)

    print(f"\nTotal de chunks: {total_chunks}")

    distribuicao = {}

    for documento in db.docstore._dict.values():
        doc_type = documento.metadata.get("doc_type", "não informado")

        if doc_type not in distribuicao:
            distribuicao[doc_type] = 0

        distribuicao[doc_type] += 1

    print("\nDistribuição por doc_type:")

    for doc_type, quantidade in distribuicao.items():
        print(f"- {doc_type}: {quantidade}")

    perguntas = [
        "Quais produtos a empresa VendeFácil oferece e quais são suas principais funcionalidades?",

        "Quais lojas estão localizadas em Minas Gerais e quais módulos elas possuem?",

        "Quais clientes estão no plano Enterprise e qual é o valor mensal?"
    ]

    print("\n" + "=" * 60)
    print("TESTE DE SIMILARIDADE")
    print("=" * 60)

    for pergunta in perguntas:

        print(f"\nPergunta: {pergunta}")
        print("-" * 60)

    resultados = db.similarity_search_with_score(
        pergunta,
        k=5
    )

    for posicao, (documento, score) in enumerate(resultados, start=1):

        print(f"\n{posicao}. Score: {score:.4f}")
        print(f"doc_type: {documento.metadata.get('doc_type')}")
        print(f"chunk_id: {documento.metadata.get('chunk_id')}")
        print(f"metadado: {documento.metadata}")
        print(f"Texto: {documento.page_content}")


if __name__ == "__main__":
    script_sanidade()
