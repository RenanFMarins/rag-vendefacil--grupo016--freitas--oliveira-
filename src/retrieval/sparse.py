import re
import unicodedata
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.retrieval.models import QueryFilters


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[._@/+-][a-z0-9]+)*")
PORTUGUESE_STOPWORDS = frozenset(
    {
        "a",
        "ao",
        "aos",
        "as",
        "com",
        "como",
        "da",
        "das",
        "de",
        "do",
        "dos",
        "e",
        "em",
        "na",
        "nas",
        "no",
        "nos",
        "o",
        "os",
        "para",
        "por",
        "que",
        "se",
        "um",
        "uma",
    }
)
SEARCHABLE_METADATA_FIELDS = (
    "chunk_id",
    "ticket_id",
    "customer_id",
    "source_file",
)


def tokenize_for_bm25(text: str) -> list[str]:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return [
        token
        for token in TOKEN_PATTERN.findall(without_accents)
        if token not in PORTUGUESE_STOPWORDS
    ]


def _document_search_text(document: Document) -> str:
    identifier_values = (
        str(document.metadata[field])
        for field in SEARCHABLE_METADATA_FIELDS
        if document.metadata.get(field) is not None
    )
    return " ".join((document.page_content, *identifier_values))


def _matches_filters(document: Document, filters: Mapping[str, object]) -> bool:
    def value_matches(actual: object, expected: object) -> bool:
        if isinstance(actual, (list, tuple, set, frozenset)):
            return expected in actual
        return actual == expected

    return all(
        value_matches(document.metadata.get(field), value)
        for field, value in filters.items()
    )


@dataclass(frozen=True)
class SparseSearchResult:
    document: Document
    score: float
    rank: int

    @property
    def chunk_id(self) -> str:
        return str(self.document.metadata["chunk_id"])


class BM25SparseRetriever:
    def __init__(self, documents: Sequence[Document]) -> None:
        if not documents:
            raise ValueError("O BM25 requer pelo menos um Document.")

        self._documents = tuple(documents)
        chunk_ids = [self._read_chunk_id(document) for document in documents]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Os valores de chunk_id devem ser únicos.")

        self._tokenized_corpus = tuple(
            tokenize_for_bm25(_document_search_text(document))
            for document in self._documents
        )
        if any(not tokens for tokens in self._tokenized_corpus):
            raise ValueError("Todo Document precisa gerar ao menos um token.")

        self._token_sets = tuple(map(set, self._tokenized_corpus))
        self._index = BM25Okapi(self._tokenized_corpus)

    @staticmethod
    def _read_chunk_id(document: Document) -> str:
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id is None or not str(chunk_id).strip():
            raise ValueError("Todo Document precisa de um chunk_id.")
        return str(chunk_id)

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        validated_filters: QueryFilters | None = None,
        allowed_sensitivities: Collection[str] | None = None,
    ) -> list[SparseSearchResult]:
        if k < 1:
            raise ValueError("k deve ser maior que zero.")

        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []

        filters = (
            validated_filters.model_dump(exclude_none=True)
            if validated_filters
            else {}
        )
        query_token_set = set(query_tokens)
        scores = self._index.get_scores(query_tokens)

        candidates: list[tuple[float, str, Document]] = []
        for position, document in enumerate(self._documents):
            if (
                allowed_sensitivities is not None
                and document.metadata.get("sensitivity")
                not in allowed_sensitivities
            ):
                continue
            if filters and not _matches_filters(document, filters):
                continue
            if query_token_set.isdisjoint(self._token_sets[position]):
                continue
            candidates.append(
                (
                    float(scores[position]),
                    self._read_chunk_id(document),
                    document,
                )
            )

        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [
            SparseSearchResult(
                document=document,
                score=score,
                rank=rank,
            )
            for rank, (score, _chunk_id, document) in enumerate(
                candidates[:k],
                start=1,
            )
        ]

