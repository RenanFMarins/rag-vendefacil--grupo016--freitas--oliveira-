import logging
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import ValidationError

from src.generation.generator import GenerationConfig, generate_rag_response
from src.guardrails.lgpd import classify_lgpd_question
from src.guardrails.masking import mask_personal_data
from src.retrieval.pipeline import HybridRetriever
from starter.schema import RAGResponse, SourceEvidence
from src.guardrails.scope import build_out_of_scope_refusal
from src.guardrails.scope import classify_question_scope


LOGGER = logging.getLogger(__name__)
ALLOWED_GENERATION_SENSITIVITIES = frozenset({"publico", "interno"})


@dataclass(frozen=True)
class RetrievedChunk:
    """Identifica um resultado recuperado sem expor seu conteúdo."""

    chunk_id: str
    source_file: str
    rank: int


@dataclass(frozen=True)
class RAGPipelineExecution:
    """Resposta final acompanhada do trace seguro usado na avaliação."""

    response: RAGResponse
    retrieved_chunks: tuple[RetrievedChunk, ...] = ()


def _build_lgpd_refusal() -> RAGResponse:
    return RAGResponse(
        answer=(
            "Não posso fornecer essa informação porque ela é protegida "
            "pela política de privacidade e segurança."
        ),
        confidence_level="recusado",
        sources_used=[],
        reasoning="A política LGPD exige a recusa desta categoria de solicitação.",
        is_refusal=True,
        refusal_reason="lgpd",
    )


def _copy_with_masked_content(document: Document) -> Document:
    return Document(
        page_content=mask_personal_data(document.page_content),
        metadata=dict(document.metadata),
    )


def _mask_response(response: RAGResponse) -> RAGResponse:
    sources = [
        SourceEvidence(
            filepath=source.filepath,
            chunk_id=source.chunk_id,
            quotation=mask_personal_data(source.quotation),
        )
        for source in response.sources_used
    ]
    return RAGResponse(
        answer=mask_personal_data(response.answer),
        confidence_level=response.confidence_level,
        sources_used=sources,
        reasoning=mask_personal_data(response.reasoning),
        is_refusal=response.is_refusal,
        refusal_reason=response.refusal_reason,
    )


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        *,
        scope_llm: BaseChatModel | None = None,
        generation_llm: BaseChatModel | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        self._retriever = retriever
        self._scope_llm = scope_llm
        self._generation_llm = generation_llm
        self._generation_config = generation_config or GenerationConfig()

    def _log_response(self, response: RAGResponse, debug: bool) -> None:
        if not debug:
            return
        LOGGER.info("Quantidade de fontes: %d", len(response.sources_used))
        LOGGER.info("confidence_level: %s", response.confidence_level)
        LOGGER.info("is_refusal: %s", response.is_refusal)
        LOGGER.info("refusal_reason: %s", response.refusal_reason)

    def answer_with_trace(
        self,
        question: str,
        *,
        aggregate_group_size: int | None = None,
        debug: bool = False,
    ) -> RAGPipelineExecution:
        """Executa o pipeline e preserva somente IDs seguros do Top-K."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("A pergunta não pode estar vazia.")

        lgpd_decision = classify_lgpd_question(
            normalized_question,
            aggregate_group_size=aggregate_group_size,
        )
        if debug:
            LOGGER.info("Ação LGPD: %s", lgpd_decision.action)

        if lgpd_decision.action == "RECUSAR":
            response = _build_lgpd_refusal()
            self._log_response(response, debug)
            return RAGPipelineExecution(response=response)

        operational_question = (
            mask_personal_data(normalized_question)
            if lgpd_decision.action == "MASCARAR"
            else normalized_question
        )

        try:
            scope_decision = classify_question_scope(
                operational_question,
                self._retriever.metadata_catalog,
                llm=self._scope_llm,
                debug=False,
            )
        except (ValidationError, OutputParserException):
            response = RAGResponse(
                answer=(
                    "Não foi possível confirmar que a pergunta pertence ao "
                    "escopo corporativo da VendeFácil."
                ),
                confidence_level="recusado",
                sources_used=[],
                reasoning="A classificação de escopo não pôde ser validada.",
                is_refusal=True,
                refusal_reason="fora_de_escopo",
            )
            self._log_response(response, debug)
            return RAGPipelineExecution(response=response)

        if debug:
            LOGGER.info(
                "Classificação de escopo: %s",
                scope_decision.classification,
            )

        if not scope_decision.is_in_scope:
            response = build_out_of_scope_refusal(scope_decision)
            self._log_response(response, debug)
            return RAGPipelineExecution(response=response)

        retrieval_response = self._retriever.retrieve(
            operational_question,
            debug=False,
            allowed_sensitivities=ALLOWED_GENERATION_SENSITIVITIES,
        )
        retrieved_ids = [result.chunk_id for result in retrieval_response.results]
        if debug:
            LOGGER.info("chunk_ids recuperados: %s", retrieved_ids)

        allowed_results = [
            result
            for result in retrieval_response.results
            if result.document.metadata.get("sensitivity")
            in ALLOWED_GENERATION_SENSITIVITIES
        ]
        documents = [result.document for result in allowed_results]
        retrieved_chunks = tuple(
            RetrievedChunk(
                chunk_id=result.chunk_id,
                source_file=str(result.document.metadata["source_file"]),
                rank=rank,
            )
            for rank, result in enumerate(allowed_results, start=1)
        )
        if lgpd_decision.action == "MASCARAR":
            generation_documents: Sequence[Document] = [
                _copy_with_masked_content(document)
                for document in documents
            ]
        else:
            generation_documents = documents

        response = generate_rag_response(
            operational_question,
            generation_documents,
            llm=self._generation_llm,
            config=self._generation_config,
            debug=debug,
        )
        if lgpd_decision.action == "MASCARAR":
            response = _mask_response(response)

        validated_response = RAGResponse.model_validate(response)
        self._log_response(validated_response, debug)
        return RAGPipelineExecution(
            response=validated_response,
            retrieved_chunks=retrieved_chunks,
        )

    def answer(
        self,
        question: str,
        *,
        aggregate_group_size: int | None = None,
        debug: bool = False,
    ) -> RAGResponse:
        """Mantém a API pública existente retornando somente RAGResponse."""
        execution = self.answer_with_trace(
            question,
            aggregate_group_size=aggregate_group_size,
            debug=debug,
        )
        return execution.response

