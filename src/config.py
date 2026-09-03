"""Configurações compartilhadas pela recuperação da Etapa 2."""

from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage"
ENV_FILE = PROJECT_ROOT / ".env"

INDEX_PATHS = {
    "csv": STORAGE_DIR / "faiss_csv",
    "json": STORAGE_DIR / "faiss_json",
    "jsonl": STORAGE_DIR / "faiss_jsonl",
    "markdown": STORAGE_DIR / "faiss_markdown",
    "pdf": STORAGE_DIR / "faiss_pdf",
    "txt": STORAGE_DIR / "faiss_txt",
}

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_EMBEDDING_MODEL_ENV = "OPENAI_EMBEDDING_MODEL"
OPENAI_QUERY_ANALYZER_MODEL_ENV = "OPENAI_QUERY_ANALYZER_MODEL"
OPENAI_SCOPE_MODEL_ENV = "OPENAI_SCOPE_MODEL"
OPENAI_GENERATION_MODEL_ENV = "OPENAI_GENERATION_MODEL"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
CHAT_MODEL_TEMPERATURE = 0


def load_project_environment() -> None:
    """Carrega o .env da raiz sem sobrescrever variáveis já definidas."""
    load_dotenv(ENV_FILE, override=False)
