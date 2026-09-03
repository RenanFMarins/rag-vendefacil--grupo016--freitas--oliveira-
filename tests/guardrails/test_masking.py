from src.guardrails.masking import mask_card_numbers
from src.guardrails.masking import mask_personal_data
from src.guardrails.masking import mask_personal_emails
from src.guardrails.masking import mask_phone_numbers
from src.guardrails.masking import mask_residential_addresses


def test_masks_personal_email() -> None:
    text = "Contato: maria@email.com"

    masked = mask_personal_emails(text)

    assert masked == "Contato: ma***@***.com"
    assert "maria@email.com" not in masked


def test_masks_phone_preserving_useful_format() -> None:
    text = "Telefone: (31) 98765-4312"

    masked = mask_phone_numbers(text)

    assert masked == "Telefone: (31) 9****-**12"
    assert "98765-4312" not in masked


def test_masks_card_number_preserving_only_last_four_digits() -> None:
    text = "Cartão: 4111 1111 1111 1111"

    masked = mask_card_numbers(text)

    assert masked == "Cartão: **** **** **** 1111"
    assert "4111 1111 1111 1111" not in masked


def test_keeps_text_without_pii_unchanged() -> None:
    text = "O manual do PDV explica como realizar uma sangria."

    assert mask_personal_data(text) == text


def test_masks_multiple_values_in_the_same_text() -> None:
    text = (
        "Contatos: maria@email.com e joao@empresa.com.br; "
        "telefones (31) 98765-4312 e (11) 3456-7890; "
        "cartão 4111-1111-1111-1111."
    )

    masked = mask_personal_data(text)

    assert "ma***@***.com" in masked
    assert "jo***@***.com.br" in masked
    assert "(31) 9****-**12" in masked
    assert "(11) 3***-**90" in masked
    assert "****-****-****-1111" in masked
    assert "maria@email.com" not in masked
    assert "98765-4312" not in masked
    assert "4111-1111-1111-1111" not in masked


def test_masks_residential_address_conservatively() -> None:
    text = "Entrega em Avenida Brasil, 1500, apartamento 20. Confirmado."

    masked = mask_residential_addresses(text)

    assert masked == "Entrega em Avenida ***. Confirmado."
    assert "Brasil" not in masked
    assert "1500" not in masked

