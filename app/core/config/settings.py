"""
Application settings loaded from environment variables / .env file.

Design notes
------------
- Single Settings object, cached via lru_cache so it's read once per process.
- Every configurable knob from the spec (chunking, retrieval, embeddings,
  reranking, LLM provider, security, MCP) lives here so no other module
  reaches into os.environ directly. This keeps infrastructure config
  decoupled from business logic (Clean Architecture boundary).
"""

from enum import Enum
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    GROQ = "groq"
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


class ChunkStrategy(str, Enum):
    RECURSIVE = "recursive"
    FIXED = "fixed"
    OVERLAP = "overlap"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- App ----
    app_name: str = "Enterprise RAG Assistant"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # ---- LLM ----
    llm_provider: LLMProvider = LLMProvider.GROQ
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    groq_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    anthropic_api_key: str = ""

    # ---- Embeddings ----
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32

    # ---- Reranker ----
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str = "cpu"
    reranker_top_k: int = 5

    # ---- Vector store ----
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "enterprise_docs"

    # ---- Relational DB ----
    database_url: str = "sqlite+aiosqlite:///./data/app.db"

    # ---- Chunking ----
    chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    chunk_size: int = 800
    chunk_overlap: int = 120

    # ---- Retrieval ----
    retrieval_top_k: int = 10
    similarity_threshold: float = 0.3
    hybrid_search_enabled: bool = True
    bm25_weight: float = 0.4
    dense_weight: float = 0.6

    # ---- LangSmith ----
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "enterprise-rag"

    # ---- Security ----
    api_key_header_name: str = "X-API-Key"
    backend_api_key: str = "change-me-to-a-random-secret"
    rate_limit_per_minute: int = 30
    max_upload_size_mb: int = 20
    # Stored as raw comma-separated strings (matching the .env format exactly)
    # rather than List[str] — pydantic-settings tries to JSON-decode List[...]
    # fields straight from the env value, which crashes on a plain
    # comma-separated string like ".pdf,.docx,.txt". Parsed into real lists
    # via the properties below instead.
    allowed_file_extensions: str = ".pdf,.docx,.txt,.md,.csv,.json"

    # ---- MCP ----
    mcp_filesystem_root: str = "./data/mcp_files"
    mcp_github_token: str = ""
    mcp_notion_token: str = ""

    # ---- CORS ----
    cors_allowed_origins: str = "http://localhost:8501,http://localhost:3000"

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [item.strip() for item in self.allowed_file_extensions.split(",") if item.strip()]

    @property
    def cors_origins_list(self) -> List[str]:
        return [item.strip() for item in self.cors_allowed_origins.split(",") if item.strip()]

    @property
    def active_llm_api_key(self) -> str:
        """Return whichever API key corresponds to the configured provider."""
        mapping = {
            LLMProvider.GROQ: self.groq_api_key,
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.GEMINI: self.gemini_api_key,
            LLMProvider.ANTHROPIC: self.anthropic_api_key,
        }
        return mapping[self.llm_provider]


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor — import and call this, never instantiate Settings() directly."""
    return Settings()
