import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("BACKEND_API_KEY", "test-api-key")


@pytest.fixture()
def temp_db_url() -> Generator[str, None, None]:
    """Isolated SQLite file per test so tests never share state."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        yield f"sqlite+aiosqlite:///{db_path}"


@pytest.fixture()
def api_headers() -> dict:
    return {"X-API-Key": "test-api-key"}


@pytest.fixture()
def test_client(temp_db_url, monkeypatch) -> Generator[TestClient, None, None]:
    """
    A TestClient wired to an isolated DB, with heavy singletons (embedding
    model, LLM, vector store) left untouched at the dependency layer —
    individual tests override get_ingestion_service / get_chat_service
    with fakes as needed rather than this fixture doing it globally, since
    different tests need different fake behavior.
    """
    monkeypatch.setenv("DATABASE_URL", temp_db_url)

    # get_settings() is lru_cache'd — a Settings object built by an earlier
    # test (or module import) would otherwise keep its stale DATABASE_URL
    # forever, silently ignoring monkeypatch.setenv above. Clear it so this
    # test's env actually takes effect.
    from app.core.config.settings import get_settings

    get_settings.cache_clear()

    # Reset the cached engine/session factory so the new DATABASE_URL takes effect
    import app.repositories.database as db_module

    db_module._engine = None
    db_module._session_factory = None

    from app.main import app

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
    get_settings.cache_clear()
