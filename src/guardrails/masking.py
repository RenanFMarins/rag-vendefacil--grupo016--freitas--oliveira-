import re


EMAIL_PATTERN = re.compile(
    r"\b(?P<local>[A-Za-z0-9._%+-]+)@"
    r"(?P<domain>[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b"
)
PHONE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?P<country>\+?55[\s.-]*)?"
    r"(?P<area>\(\d{2}\)|\d{2})"
    r"(?P<area_sep>[\s.-]*)"
    r"(?P<prefix>\d{4,5})"
    r"(?P<number_sep>[-.\s]?)"
    r"(?P<suffix>\d{4})"
    r"(?!\d)"
)
CARD_PATTERN = re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)")

# Estratégia conservadora: ao reconhecer um prefixo de logradouro, todo o
# trecho até o fim da linha ou da frase é ocultado. Isso reduz o risco de
# expor número/complemento, mas pode ocultar texto demais e não reconhece
# endereços sem prefixos conhecidos.
ADDRESS_PATTERN = re.compile(
    r"\b(?P<kind>Rua|R\.|Avenida|Av\.|Alameda|Travessa|Praça|Estrada|Rodovia)"
    r"\s+[^;\n.!?]{2,120}",
    re.IGNORECASE,
)


def _mask_email(match: re.Match[str]) -> str:
    local = match.group("local")
    domain_labels = match.group("domain").split(".")
    suffix_size = (
        2
        if len(domain_labels) >= 3 and len(domain_labels[-1]) == 2
        else 1
    )
    suffix = ".".join(domain_labels[-suffix_size:])
    visible_local = local[:2]
    return f"{visible_local}***@***.{suffix}"


def mask_personal_emails(text: str) -> str:
    """Preserva parte do identificador e o sufixo do domínio."""
    return EMAIL_PATTERN.sub(_mask_email, text)


def _mask_phone(match: re.Match[str]) -> str:
    prefix = match.group("prefix")
    suffix = match.group("suffix")
    masked_number = (
        f"{prefix[0]}{'*' * (len(prefix) - 1)}"
        f"{match.group('number_sep')}**{suffix[-2:]}"
    )
    return (
        f"{match.group('country') or ''}"
        f"{match.group('area')}"
        f"{match.group('area_sep')}"
        f"{masked_number}"
    )


def mask_phone_numbers(text: str) -> str:
    return PHONE_PATTERN.sub(_mask_phone, text)


def _mask_card(match: re.Match[str]) -> str:
    value = match.group(0)
    digit_count = sum(character.isdigit() for character in value)
    digits_seen = 0
    masked: list[str] = []

    for character in value:
        if not character.isdigit():
            masked.append(character)
            continue
        digits_seen += 1
        masked.append("*" if digits_seen <= digit_count - 4 else character)

    return "".join(masked)


def mask_card_numbers(text: str) -> str:
    return CARD_PATTERN.sub(_mask_card, text)


def _mask_address(match: re.Match[str]) -> str:
    return f"{match.group('kind')} ***"


def mask_residential_addresses(text: str) -> str:
    return ADDRESS_PATTERN.sub(_mask_address, text)


def mask_personal_data(text: str) -> str:
    masked = mask_card_numbers(text)
    masked = mask_personal_emails(masked)
    masked = mask_phone_numbers(masked)
    return mask_residential_addresses(masked)

