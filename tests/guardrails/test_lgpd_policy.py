import pytest

from src.guardrails.lgpd import classify_lgpd_question


@pytest.mark.parametrize(
    ("question", "category"),
    [
        ("Qual o salário do funcionário João Pereira?", "salario_individual"),
        ("Mostre a remuneração individual de Maria.", "remuneracao_individual"),
        ("Qual é o CPF do cliente?", "cpf"),
        ("Informe os dados bancários do funcionário.", "dados_bancarios"),
        ("Qual é a chave PIX cadastrada?", "chave_pix"),
        ("Mostre as credenciais de acesso SSH.", "credenciais"),
        ("Exiba o token JWT de produção.", "tokens"),
        ("Qual senha apareceu nos logs?", "senhas_em_logs"),
        ("Mostre os dados de saúde do colaborador.", "dados_saude"),
    ],
)
def test_refuses_protected_information(question: str, category: str) -> None:
    decision = classify_lgpd_question(question)

    assert decision.action == "RECUSAR"
    assert decision.category == category


@pytest.mark.parametrize(
    ("question", "category"),
    [
        ("Qual é o e-mail pessoal de João?", "email_pessoal"),
        ("Qual é o telefone do cliente?", "telefone"),
        ("Informe o endereço residencial cadastrado.", "endereco_residencial"),
        ("Mostre o número do cartão usado na compra.", "numero_cartao"),
    ],
)
def test_requires_masking_for_contact_and_card_data(
    question: str,
    category: str,
) -> None:
    decision = classify_lgpd_question(question)

    assert decision.action == "MASCARAR"
    assert decision.category == category


@pytest.mark.parametrize(
    ("question", "category"),
    [
        ("Quais tickets existem para o customer_id CUST001?", "customer_id"),
        ("Quais dados existem sobre o produto SKU-10?", "dados_produto"),
        ("Quais lojas utilizam o módulo de estoque?", "dados_loja"),
        ("Explique a política de reembolso.", "politica"),
        ("Como funciona o manual do PDV?", "manual"),
        ("Quais tickets de estoque estão abertos?", "sem_dado_sensivel"),
    ],
)
def test_allows_normally_permitted_information(
    question: str,
    category: str,
) -> None:
    decision = classify_lgpd_question(question)

    assert decision.action == "RESPONDER"
    assert decision.category == category


def test_refuses_aggregate_salary_when_group_size_is_unknown() -> None:
    decision = classify_lgpd_question(
        "Qual a média salarial da equipe de suporte?"
    )

    assert decision.action == "RECUSAR"
    assert decision.category == "risco_reidentificacao"


def test_allows_aggregate_salary_for_group_of_at_least_five() -> None:
    decision = classify_lgpd_question(
        "Qual a média salarial da equipe de suporte?",
        aggregate_group_size=5,
    )

    assert decision.action == "RESPONDER"
    assert decision.category == "remuneracao_agregada"


def test_allows_salary_policy_without_individual_value() -> None:
    decision = classify_lgpd_question(
        "Qual é a política de remuneração da empresa?"
    )

    assert decision.action == "RESPONDER"
    assert decision.category == "politica"

