from typing import List

from app.core.config.logging_config import get_logger
from app.core.domain.models import RetrievedChunk
from app.core.interfaces.llm_provider import LLMMessage, LLMProvider
from app.prompts.rag_prompts import build_grading_messages

log = get_logger(__name__)


class CriticAgent:
    """The 'Retrieval Grader' from the spec's LangGraph workflow. Judges
    whether each retrieved chunk is actually relevant to the question, and
    whether the surviving set is enough to answer from — this is what
    drives the graph's "Need More Retrieval?" conditional edge."""

    def __init__(self, llm_provider: LLMProvider, min_relevant_chunks: int = 1):
        self.llm_provider = llm_provider
        self.min_relevant_chunks = min_relevant_chunks

    async def grade(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        relevant: List[RetrievedChunk] = []
        for rc in chunks:
            messages = [LLMMessage(**m) for m in build_grading_messages(query, rc.chunk.content)]
            verdict = await self.llm_provider.generate(messages, temperature=0.0, max_tokens=5)
            is_relevant = "relevant" in verdict.strip().lower() and "irrelevant" not in verdict.strip().lower()
            if is_relevant:
                relevant.append(rc)
        log.info("retrieval_graded", total=len(chunks), relevant=len(relevant))
        return relevant

    def is_sufficient(self, graded_chunks: List[RetrievedChunk]) -> bool:
        return len(graded_chunks) >= self.min_relevant_chunks
