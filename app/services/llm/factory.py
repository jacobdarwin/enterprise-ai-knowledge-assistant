"""
LLM provider factory. Reads settings.llm_provider and constructs the
matching LangChain chat model, wrapped in LangChainLLMAdapter so the
rest of the app only ever depends on the LLMProvider port.

Each branch lazy-imports its package so installing, say, only
langchain-groq (the free-tier default) doesn't require openai/gemini/
anthropic SDKs to be present too.
"""

from functools import lru_cache

from app.core.config.langsmith_setup import configure_langsmith
from app.core.config.logging_config import get_logger
from app.core.config.settings import LLMProvider as LLMProviderEnum
from app.core.config.settings import get_settings
from app.core.interfaces.llm_provider import LLMProvider
from app.services.llm.langchain_adapter import LangChainLLMAdapter

log = get_logger(__name__)


def _build_groq_model(settings):
    from langchain_groq import ChatGroq

    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is not set — get a free key at console.groq.com")
    return ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key)


def _build_openai_model(settings):
    from langchain_openai import ChatOpenAI

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    return ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key)


def _build_gemini_model(settings):
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set — get a free key at aistudio.google.com")
    return ChatGoogleGenerativeAI(model=settings.llm_model, google_api_key=settings.gemini_api_key)


def _build_anthropic_model(settings):
    from langchain_anthropic import ChatAnthropic

    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set")
    return ChatAnthropic(model=settings.llm_model, api_key=settings.anthropic_api_key)


_BUILDERS = {
    LLMProviderEnum.GROQ: _build_groq_model,
    LLMProviderEnum.OPENAI: _build_openai_model,
    LLMProviderEnum.GEMINI: _build_gemini_model,
    LLMProviderEnum.ANTHROPIC: _build_anthropic_model,
}


@lru_cache
def get_llm_provider() -> LLMProvider:
    configure_langsmith()
    settings = get_settings()
    builder = _BUILDERS.get(settings.llm_provider)
    if builder is None:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")

    log.info("llm_provider_initializing", provider=settings.llm_provider, model=settings.llm_model)
    chat_model = builder(settings)
    return LangChainLLMAdapter(
        chat_model=chat_model,
        default_temperature=settings.llm_temperature,
        default_max_tokens=settings.llm_max_tokens,
    )
