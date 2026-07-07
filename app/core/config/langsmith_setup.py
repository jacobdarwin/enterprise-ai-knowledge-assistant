"""
LangSmith tracing activation.

Bug this fixes: pydantic-settings reads LANGCHAIN_* values from .env into
our Settings object, but LangChain/LangGraph's tracing machinery checks
os.environ directly — it has no idea our Settings object exists. Without
this, a person could set LANGCHAIN_TRACING_V2=true in .env and see zero
traces show up in LangSmith, with no error anywhere. Call this before
constructing any LangChain chat model or graph.
"""

import os

from app.core.config.settings import get_settings

_configured = False


def configure_langsmith() -> None:
    global _configured
    if _configured:
        return

    settings = get_settings()
    tracing_enabled = bool(settings.langchain_tracing_v2 and settings.langchain_api_key)
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if tracing_enabled else "false"
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    if settings.langchain_project:
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    elif settings.langchain_tracing_v2:
        from app.core.config.logging_config import get_logger

        get_logger(__name__).warning(
            "langsmith_tracing_enabled_without_api_key",
            hint="Set LANGCHAIN_API_KEY in .env, or set LANGCHAIN_TRACING_V2=false to silence this.",
        )

    _configured = True
