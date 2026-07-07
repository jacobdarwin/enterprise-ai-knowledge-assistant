"""
Prompt construction for the RAG pipeline.

This module is the single place where the "never hallucinate, cite
sources, say I don't know" contract from the spec is actually encoded.
Every prompt sent to the LLM for answer generation goes through
`build_answer_messages` so that contract can't accidentally be
bypassed by some other code path.
"""

from typing import List

from app.core.domain.models import RetrievedChunk

SYSTEM_PROMPT = """You are an enterprise knowledge assistant. You answer questions \
using ONLY the context provided below, retrieved from the organization's own documents.

Rules you must follow:
1. Base your answer strictly on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to answer confidently, \
respond exactly with: "I don't know based on the available documents." Do not guess.
3. Every factual claim must be traceable to a specific source. When you state a fact, \
reference it using the format [filename, page X] immediately after the sentence.
4. Do not fabricate filenames, page numbers, or facts not present in the context.
5. Be concise and direct. Do not pad the answer with generic disclaimers beyond rule 2.
"""

GRADING_SYSTEM_PROMPT = """You are a retrieval quality grader for a RAG system. \
Given a user question and a retrieved document chunk, decide if the chunk is \
relevant enough to help answer the question.

Respond with exactly one word: "relevant" or "irrelevant". No explanation, no punctuation."""


def build_context_block(chunks: List[RetrievedChunk]) -> str:
    """Render retrieved chunks into a numbered, source-tagged context block
    the LLM can cite directly by filename/page."""
    if not chunks:
        return "(no relevant context was retrieved)"

    blocks = []
    for i, rc in enumerate(chunks, start=1):
        page_str = f"page {rc.chunk.page}" if rc.chunk.page is not None else "no page"
        blocks.append(f"[Source {i}: {rc.chunk.filename}, {page_str}]\n{rc.chunk.content}")
    return "\n\n".join(blocks)


def build_answer_messages(query: str, chunks: List[RetrievedChunk], history_summary: str = "") -> list:
    """Returns a list of dicts (role, content) — deliberately plain dicts here
    rather than LLMMessage, since callers may be either LangGraph nodes or
    the agents layer; both convert to LLMMessage right before calling the
    LLMProvider port."""
    context_block = build_context_block(chunks)
    history_note = (
        f"\nConversation so far (for continuity only, do not treat as source material):\n{history_summary}\n"
        if history_summary
        else ""
    )

    user_content = f"""Context:
{context_block}
{history_note}
Question: {query}

Answer the question following all rules in the system prompt."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_grading_messages(query: str, chunk_content: str) -> list:
    return [
        {"role": "system", "content": GRADING_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}\n\nDocument chunk:\n{chunk_content}"},
    ]


def build_query_rewrite_messages(query: str, previous_attempt_note: str = "") -> list:
    """Used by the Query Agent when the grader says retrieval was insufficient
    and we need a reformulated query for a second retrieval pass."""
    system = (
        "You rewrite user questions into better search queries for a document "
        "retrieval system. Keep the rewritten query short, keyword-rich, and "
        "faithful to the original intent. Return ONLY the rewritten query, "
        "nothing else."
    )
    user = f"Original question: {query}"
    if previous_attempt_note:
        user += f"\nNote: the first retrieval attempt did not find enough relevant information. {previous_attempt_note}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
