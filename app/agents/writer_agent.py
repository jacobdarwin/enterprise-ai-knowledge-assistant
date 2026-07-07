from typing import List

from app.core.domain.models import RetrievedChunk
from app.core.interfaces.llm_provider import LLMMessage, LLMProvider
from app.prompts.rag_prompts import build_answer_messages

NO_CONTEXT_ANSWER = "I don't know based on the available documents."


class WriterAgent:
    """Generates the final answer. If the Critic found nothing relevant, we
    short-circuit and return the fixed "I don't know" response without even
    calling the LLM — cheaper, faster, and removes any chance of the model
    hallucinating an answer from its own training data instead."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def write(self, query: str, graded_chunks: List[RetrievedChunk], history_summary: str = "") -> str:
        if not graded_chunks:
            return NO_CONTEXT_ANSWER

        messages = [LLMMessage(**m) for m in build_answer_messages(query, graded_chunks, history_summary)]
        return await self.llm_provider.generate(messages)

    async def write_stream(self, query: str, graded_chunks: List[RetrievedChunk], history_summary: str = ""):
        if not graded_chunks:
            yield NO_CONTEXT_ANSWER
            return

        messages = [LLMMessage(**m) for m in build_answer_messages(query, graded_chunks, history_summary)]
        async for delta in self.llm_provider.stream(messages):
            yield delta
