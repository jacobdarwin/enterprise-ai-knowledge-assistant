"""
Adapts any LangChain BaseChatModel to our LLMProvider port.

One adapter class, parameterized by whichever concrete chat model the
factory constructs (ChatGroq, ChatOpenAI, ChatGoogleGenerativeAI,
ChatAnthropic). This is the payoff of the interface from Phase 1: the
LangGraph nodes and agents only ever talk to `LLMProvider`, so switching
LLM_PROVIDER in .env from groq to anthropic requires zero changes here
or anywhere upstream.
"""

from typing import AsyncIterator, List, Optional

from app.core.interfaces.llm_provider import LLMMessage, LLMProvider


class LangChainLLMAdapter(LLMProvider):
    def __init__(self, chat_model, default_temperature: float, default_max_tokens: int):
        self._model = chat_model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens

    @staticmethod
    def _to_langchain_messages(messages: List[LLMMessage]):
        # Lazy import so this module doesn't force langchain-core onto
        # anything that just wants the LLMMessage type.
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        role_map = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
        return [role_map[m.role](content=m.content) for m in messages]

    def _bind(self, temperature: Optional[float], max_tokens: Optional[int]):
        return self._model.bind(
            temperature=temperature if temperature is not None else self._default_temperature,
            max_tokens=max_tokens if max_tokens is not None else self._default_max_tokens,
        )

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        lc_messages = self._to_langchain_messages(messages)
        model = self._bind(temperature, max_tokens)
        response = await model.ainvoke(lc_messages)
        return response.content

    async def stream(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        lc_messages = self._to_langchain_messages(messages)
        model = self._bind(temperature, max_tokens)
        async for chunk in model.astream(lc_messages):
            if chunk.content:
                yield chunk.content
