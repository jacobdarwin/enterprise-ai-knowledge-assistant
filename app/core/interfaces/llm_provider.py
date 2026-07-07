from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Optional

from pydantic import BaseModel


class LLMMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMProvider(ABC):
    """
    Port for chat-completion LLMs. Infra impl: app/services/llm/
    (one adapter per provider, selected in a factory keyed off
    settings.llm_provider — see Phase 5).
    """

    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Non-streaming completion."""

    @abstractmethod
    async def stream(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Streaming completion — yields text deltas."""
