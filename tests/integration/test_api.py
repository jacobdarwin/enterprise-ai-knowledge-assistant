from app.api.dependencies import get_ingestion_service


def test_health_check(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_requires_api_key(test_client):
    response = test_client.post("/chat", json={"query": "test"})
    assert response.status_code == 401


def test_documents_requires_api_key(test_client):
    response = test_client.get("/documents")
    assert response.status_code == 401


def test_list_documents_empty_initially(test_client, api_headers):
    response = test_client.get("/documents", headers=api_headers)
    assert response.status_code == 200
    assert response.json() == []


def test_delete_nonexistent_document_returns_404(test_client, api_headers):
    from app.main import app

    class FakeIngestionService:
        async def delete_document(self, document_id):
            pass

    app.dependency_overrides[get_ingestion_service] = lambda: FakeIngestionService()
    response = test_client.delete("/documents/does-not-exist", headers=api_headers)
    assert response.status_code == 404


def test_history_empty_initially(test_client, api_headers):
    response = test_client.get("/history", headers=api_headers)
    assert response.status_code == 200
    assert response.json() == {"conversations": []}
