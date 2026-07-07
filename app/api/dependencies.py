"""
Dependency injection container.

FastAPI's Depends() system calls these functions per-request, but the
expensive singletons (embedding model, vector store, LLM client, the
compiled LangGraph) are built once via lru_cache/module-level globals
and reused across requests. Only the DB session is genuinely
per-request (SQLAlchemy sessions aren't safe to share across concurrent
requests).
"""

from functools import lru_cache
from typing import Optional

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.citation_agent import CitationAgent
from app.agents.critic_agent import CriticAgent
from app.agents.query_agent import QueryAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.writer_agent import WriterAgent
from app.core.config.settings import get_settings
from app.core.interfaces.reranker import Reranker
from app.core.interfaces.vector_store import VectorStore
from app.embeddings.sentence_transformer_provider import get_embedding_provider
from app.graph.workflow import build_rag_graph
from app.rag.vector_store.chroma_store import ChromaVectorStore
from app.repositories.database import get_db_session
from app.repositories.sqlite_chat_repository import SQLiteChatRepository
from app.repositories.sqlite_document_repository import SQLiteDocumentRepository
from app.retrieval.hybrid_retriever import HybridRetriever
from app.services.chat_service import ChatService
from app.services.ingestion_service import IngestionService
from app.services.llm.factory import get_llm_provider


@lru_cache
def get_vector_store() -> VectorStore:
    return ChromaVectorStore(embedding_provider=get_embedding_provider())


@lru_cache
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(vector_store=get_vector_store())


@lru_cache
def get_reranker() -> Optional[Reranker]:
    from app.reranker.cross_encoder_reranker import CrossEncoderReranker

    return CrossEncoderReranker()


@lru_cache
def get_compiled_graph():
    settings = get_settings()
    llm = get_llm_provider()
    query_agent = QueryAgent(llm)
    retriever_agent = RetrieverAgent(get_hybrid_retriever(), reranker=get_reranker())
    critic_agent = CriticAgent(llm)
    writer_agent = WriterAgent(llm)
    citation_agent = CitationAgent()
    return build_rag_graph(
        query_agent,
        retriever_agent,
        critic_agent,
        writer_agent,
        citation_agent,
        default_top_k=settings.retrieval_top_k,
    )


def get_document_repository(session: AsyncSession = Depends(get_db_session)) -> SQLiteDocumentRepository:
    return SQLiteDocumentRepository(session)


def get_chat_repository(session: AsyncSession = Depends(get_db_session)) -> SQLiteChatRepository:
    return SQLiteChatRepository(session)


def get_ingestion_service(
    document_repository: SQLiteDocumentRepository = Depends(get_document_repository),
) -> IngestionService:
    return IngestionService(
        document_repository=document_repository,
        vector_store=get_vector_store(),
        hybrid_retriever=get_hybrid_retriever(),
    )


def get_chat_service(
    chat_repository: SQLiteChatRepository = Depends(get_chat_repository),
) -> ChatService:
    llm = get_llm_provider()
    return ChatService(
        chat_graph=get_compiled_graph(),
        chat_repository=chat_repository,
        query_agent=QueryAgent(llm),
        retriever_agent=RetrieverAgent(get_hybrid_retriever(), reranker=get_reranker()),
        critic_agent=CriticAgent(llm),
        writer_agent=WriterAgent(llm),
        citation_agent=CitationAgent(),
        max_retrieval_attempts=2,
    )
