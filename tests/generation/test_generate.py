from langchain_core.documents import Document

from src.generation.generator import GenerationConfig
from src.generation.generator import GenerationDraft
from src.generation.generator import generate_rag_response


class FakeStructuredModel:
    def __init__(self, response: object) -> None:
        self.responses = response if isinstance(response, list) else [response]
        self.messages = []
        self.messages_history = []
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        self.messages = messages
        self.messages_history.append(messages)
        return self.responses[self.calls - 1]


class FakeChatModel:
    def __init__(self, response: object) -> None:
        self.structured_model = FakeStructuredModel(response)
        self.schema = None
        self.method = None
        self.strict = None

    def with_structured_output(self, schema, *, method, strict):
        self.schema = schema
        self.method = method
        self.strict = strict
        return self.structured_model


def make_document(
    chunk_id: str,
    content: str,
    source_file: str,
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source_file": source_file,
            "chunk_id": chunk_id,
            "doc_type": "manual",
            "sensitivity": "publico",
        },
    )


def test_generates_answer_with_strong_evidence() -> None:
    document = make_document(
        "manual:pdv:sangria",
        "A sangria retira valores do gaveteiro por medida de segurança.",
        "manual_pdv.md",
    )
    fake_llm = FakeChatModel(
        GenerationDraft(
            answer="A sangria retira valores do gaveteiro.",
            confidence_level="alta",
            selected_chunk_ids=["manual:pdv:sangria"],
            reasoning="O manual descreve diretamente a finalidade da sangria.",
            has_sufficient_evidence=True,
        )
    )

    response = generate_rag_response(
        "O que é uma sangria?",
        [document],
        llm=fake_llm,
    )

    assert response.answer == "A sangria retira valores do gaveteiro."
    assert response.confidence_level == "alta"
    assert response.is_refusal is False
    assert response.refusal_reason is None
    assert response.sources_used[0].filepath == "manual_pdv.md"
    assert response.sources_used[0].chunk_id == "manual:pdv:sangria"
    assert response.sources_used[0].quotation == document.page_content
    assert fake_llm.schema is GenerationDraft
    assert fake_llm.method == "json_schema"
    assert fake_llm.strict is True
    assert fake_llm.structured_model.calls == 1
    prompt = str(fake_llm.structured_model.messages)
    assert "O que é uma sangria?" in prompt
    assert "manual_pdv.md" in prompt
    assert "manual:pdv:sangria" in prompt
    assert document.page_content in prompt


def test_generates_answer_with_multiple_verified_sources() -> None:
    first = make_document(
        "manual:estoque:001",
        "O inventário cego não mostra o saldo esperado ao operador.",
        "inventario.md",
    )
    second = make_document(
        "policy:estoque:002",
        "Divergências devem ser revisadas pelo supervisor da loja.",
        "politica_estoque.md",
    )
    fake_llm = FakeChatModel(
        GenerationDraft(
            answer="O operador conta sem ver o saldo e o supervisor revisa divergências.",
            confidence_level="alta",
            selected_chunk_ids=[first.metadata["chunk_id"], second.metadata["chunk_id"]],
            reasoning="A resposta combina o manual e a política de revisão.",
            has_sufficient_evidence=True,
        )
    )

    response = generate_rag_response(
        "Como funciona a revisão do inventário cego?",
        [first, second],
        llm=fake_llm,
    )

    assert [source.chunk_id for source in response.sources_used] == [
        "manual:estoque:001",
        "policy:estoque:002",
    ]
    assert [source.quotation for source in response.sources_used] == [
        first.page_content,
        second.page_content,
    ]


def test_refuses_when_llm_reports_insufficient_evidence() -> None:
    document = make_document(
        "manual:pdv:001",
        "O manual descreve a abertura do caixa.",
        "manual_pdv.md",
    )
    fake_llm = FakeChatModel(
        GenerationDraft(
            answer="Não há informação suficiente.",
            confidence_level="baixa",
            selected_chunk_ids=[],
            reasoning="O contexto não trata da pergunta.",
            has_sufficient_evidence=False,
        )
    )

    response = generate_rag_response(
        "Qual é a política de frete?",
        [document],
        llm=fake_llm,
    )

    assert response.is_refusal is True
    assert response.confidence_level == "recusado"
    assert response.sources_used == []
    assert response.refusal_reason == "sem_evidencia"


def test_refuses_instead_of_accepting_invented_chunk_id() -> None:
    document = make_document(
        "manual:pdv:001",
        "O manual descreve a abertura do caixa.",
        "manual_pdv.md",
    )
    fake_llm = FakeChatModel(
        GenerationDraft(
            answer="Resposta supostamente fundamentada.",
            confidence_level="alta",
            selected_chunk_ids=["chunk:inventado"],
            reasoning="Fonte inexistente.",
            has_sufficient_evidence=True,
        )
    )

    response = generate_rag_response(
        "Como abrir o caixa?",
        [document],
        llm=fake_llm,
    )

    assert response.is_refusal is True
    assert response.sources_used == []
    assert response.refusal_reason == "sem_evidencia"


def test_refuses_without_calling_llm_when_no_documents_are_available() -> None:
    response = generate_rag_response("Como abrir o caixa?", [])

    assert response.is_refusal is True
    assert response.confidence_level == "recusado"
    assert response.sources_used == []
    assert response.refusal_reason == "sem_evidencia"


def test_retries_invalid_first_output_and_accepts_valid_second_output() -> None:
    document = make_document(
        "manual:pdv:001",
        "O caixa deve ser aberto com um suprimento inicial.",
        "manual_pdv.md",
    )
    valid_draft = GenerationDraft(
        answer="O caixa é aberto com um suprimento inicial.",
        confidence_level="alta",
        selected_chunk_ids=["manual:pdv:001"],
        reasoning="O procedimento consta no manual.",
        has_sufficient_evidence=True,
    )
    fake_llm = FakeChatModel(
        [
            {"answer": "Saída incompleta"},
            valid_draft,
        ]
    )

    response = generate_rag_response(
        "Como abrir o caixa?",
        [document],
        llm=fake_llm,
        config=GenerationConfig(max_attempts=2),
    )

    assert response.is_refusal is False
    assert response.answer == "O caixa é aberto com um suprimento inicial."
    assert fake_llm.structured_model.calls == 2
    retry_prompt = str(fake_llm.structured_model.messages_history[1])
    assert "não correspondeu à estrutura obrigatória" in retry_prompt
    assert "has_sufficient_evidence" in retry_prompt


def test_returns_controlled_failure_after_all_attempts_are_invalid() -> None:
    document = make_document(
        "manual:pdv:001",
        "O caixa deve ser aberto com um suprimento inicial.",
        "manual_pdv.md",
    )
    fake_llm = FakeChatModel(
        [
            {"answer": "Primeira saída incompleta"},
            {"confidence_level": "valor inválido"},
        ]
    )

    response = generate_rag_response(
        "Como abrir o caixa?",
        [document],
        llm=fake_llm,
        config=GenerationConfig(max_attempts=2),
    )

    assert fake_llm.structured_model.calls == 2
    assert response.is_refusal is True
    assert response.confidence_level == "recusado"
    assert response.sources_used == []
    assert response.refusal_reason == "sem_evidencia"
    assert "validada com segurança" in response.answer

