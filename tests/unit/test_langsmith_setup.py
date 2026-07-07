import os
from types import SimpleNamespace

from app.core.config import langsmith_setup


def test_configure_langsmith_disables_tracing_without_api_key(monkeypatch):
    monkeypatch.setattr(langsmith_setup, "_configured", False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)

    monkeypatch.setattr(
        langsmith_setup,
        "get_settings",
        lambda: SimpleNamespace(
            langchain_tracing_v2=True,
            langchain_project="test-project",
            langchain_api_key="",
        ),
    )

    langsmith_setup.configure_langsmith()

    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
    assert os.environ["LANGCHAIN_PROJECT"] == "test-project"
    assert "LANGCHAIN_API_KEY" not in os.environ
