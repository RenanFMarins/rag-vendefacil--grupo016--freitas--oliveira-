import unicodedata
from collections.abc import Callable

from src.retrieval.models import QueryFilters

STATE_NAME_TO_CODE = {
    "acre": "AC",
    "alagoas": "AL",
    "amapa": "AP",
    "amazonas": "AM",
    "bahia": "BA",
    "ceara": "CE",
    "distrito federal": "DF",
    "espirito santo": "ES",
    "goias": "GO",
    "maranhao": "MA",
    "mato grosso": "MT",
    "mato grosso do sul": "MS",
    "minas gerais": "MG",
    "para": "PA",
    "paraiba": "PB",
    "parana": "PR",
    "pernambuco": "PE",
    "piaui": "PI",
    "rio de janeiro": "RJ",
    "rio grande do norte": "RN",
    "rio grande do sul": "RS",
    "rondonia": "RO",
    "roraima": "RR",
    "santa catarina": "SC",
    "sao paulo": "SP",
    "sergipe": "SE",
    "tocantins": "TO",
}

MODULE_ALIASES = {
    "e commerce": "ecommerce",
    "vendefacil loja": "ecommerce",
    "vendefacil pay": "pay",
}

EDGE_PUNCTUATION = " \t\r\n,;:{}[]()\"'"


def clean_whitespace(value: str) -> str:
    return " ".join(value.split())


def normalization_key(value: object) -> str:
    cleaned = clean_whitespace(str(value)).strip(EDGE_PUNCTUATION)
    decomposed = unicodedata.normalize("NFKD", cleaned)
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return without_accents.casefold()


def normalize_state(value: str) -> str:
    cleaned = clean_whitespace(value)
    if len(cleaned) == 2 and cleaned.isalpha():
        return cleaned.upper()

    return STATE_NAME_TO_CODE.get(normalization_key(cleaned), cleaned)


def normalize_module(value: str) -> str:
    key = normalization_key(value).replace("-", " ")
    key = clean_whitespace(key)
    return MODULE_ALIASES.get(key, key)


def normalize_categorical_value(value: str) -> str:
    return normalization_key(value)


def normalize_identifier(value: str) -> str:
    return clean_whitespace(value).upper()


FIELD_NORMALIZERS: dict[str, Callable[[str], str]] = {
    "doc_type": normalize_categorical_value,
    "state": normalize_state,
    "module": normalize_module,
    "plan": normalize_categorical_value,
    "priority": normalize_categorical_value,
    "status": normalize_categorical_value,
    "category": normalize_categorical_value,
    "sentiment": normalize_categorical_value,
    "sensitivity": normalize_categorical_value,
    "customer_id": normalize_identifier,
    "ticket_id": normalize_identifier,
}


def normalize_query_filters(filters: QueryFilters) -> QueryFilters:
    normalized_values = {
        field: FIELD_NORMALIZERS[field](value)
        for field, value in filters.model_dump(exclude_none=True).items()
    }
    return QueryFilters.model_validate(normalized_values)
