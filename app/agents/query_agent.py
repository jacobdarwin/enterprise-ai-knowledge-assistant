from app.core.interfaces.llm_provider import LLMMessage, LLMProvider
from app.prompts.rag_prompts import build_query_rewrite_messages


class QueryAgent:
    """Owns query understanding. On the first pass it just forwards the raw
    query; if the Critic Agent later judges retrieval insufficient, this
    agent rewrites the query for a second, hopefully better, retrieval pass."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    def analyze(self, query: str) -> str:
        """First-pass: light normalization only. Kept as its own step (rather
        than skipping straight to retrieval) so future work — intent
        classification, PII scrubbing, language detection — has a home."""
        return query.strip()

    async def rewrite_for_retry(self, original_query: str, note: str = "") -> str:
        messages = [LLMMessage(**m) for m in build_query_rewrite_messages(original_query, note)]
        rewritten = await self.llm_provider.generate(messages, temperature=0.0, max_tokens=64)
        return rewritten.strip().strip('"')
