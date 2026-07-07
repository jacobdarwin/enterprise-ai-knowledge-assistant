"""
Domain entities for the RAG system.

These are plain Pydantic models with zero knowledge of FastAPI, ChromaDB,
SQLAlchemy, or any other infrastructure concern. Infrastructure layers
(repositories, vector stores) convert to/from these types. This is what
keeps the "Clean Architecture" boundary real rather than decorative.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class Document(BaseModel):
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    file_type: str
    upload_time: datetime = Field(default_factory=_now)
    status: DocumentStatus = DocumentStatus.UPLOADED
    num_chunks: int = 0
    size_bytes: int = 0
    error_message: Optional[str] = None


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    filename: str
    content: str
    page: Optional[int] = None
    chunk_index: int = 0
    upload_time: datetime = Field(default_factory=_now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk plus its retrieval scores — used post-search, pre-generation."""

    chunk: Chunk
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Citation(BaseModel):
    filename: str
    page: Optional[int] = None
    chunk_id: str
    snippet: str


class ChatMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: str
    role: ChatRole
    content: str
    citations: List[Citation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    query: str
    document_ids: Optional[List[str]] = None  # optional metadata filter
    top_k: Optional[int] = None
    stream: bool = True


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    retrieval_count: int = 0
    latency_ms: float = 0.0
