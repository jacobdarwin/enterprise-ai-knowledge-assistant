"""
Shared state for the RAG LangGraph workflow.

LangGraph nodes are plain functions that take this state dict, do work,
and return a partial dict of updates to merge in. Keeping this as a
TypedDict (not a Pydantic model) is the LangGraph convention — it plays
nicely with the library's state-merging machinery.
"""

from typing import List, Optional, TypedDict

from app.core.domain.models import Citation, RetrievedChunk


class GraphState(TypedDict, total=False):
    # --- input ---
    query: str
    original_query: str
    conversation_id: str
    document_ids: Optional[List[str]]
    history_summary: str
    top_k: Optional[int]

    # --- retrieval ---
    retrieved_chunks: List[RetrievedChunk]
    graded_chunks: List[RetrievedChunk]
    retrieval_attempts: int
    max_retrieval_attempts: int

    # --- generation ---
    answer: str
    citations: List[Citation]

    # --- diagnostics ---
    retrieval_sufficient: bool
