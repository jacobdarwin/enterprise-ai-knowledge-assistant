from app.api.dependencies import get_chat_service
from app.core.domain.models import ChatResponse, Citation, Document, DocumentStatus
from app.services.ingestion_service import IngestionService


def test_upload_with_faked_ingestion(test_client, api_headers, monkeypatch):
    import app.api.routes.upload as upload_module

    monkeypatch.setattr(upload_module, "get_vector_store", lambda: object())
    monkeypatch.setattr(upload_module, "get_hybrid_retriever", lambda: object())

    async def fake_ingest(self, file_path, original_filename):
        return Document(
            filename=original_filename,
            file_type="txt",
            size_bytes=42,
            status=DocumentStatus.INDEXED,
            num_chunks=2,
        )

    monkeypatch.setattr(IngestionService, "ingest", fake_ingest)

    files = {"file": ("policy.txt", b"Employees get 18 days leave.", "text/plain")}
    response = test_client.post("/upload", headers=api_headers, files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "indexed"
    assert body["num_chunks"] == 2


def test_upload_rejects_unsupported_extension(test_client, api_headers):
    files = {"file": ("malware.exe", b"binary content", "application/octet-stream")}
    response = test_client.post("/upload", headers=api_headers, files=files)
    assert response.status_code == 400


def test_chat_non_streaming_with_faked_service(test_client, api_headers):
    from app.main import app

    class FakeChatService:
        async def handle_chat(self, request):
            return ChatResponse(
                conversation_id="conv-1",
                answer="Employees get 18 days of leave per year.",
                citations=[Citation(filename="handbook.txt", page=1, chunk_id="c1", snippet="18 days...")],
                retrieval_count=1,
                latency_ms=10.0,
            )

    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()

    response = test_client.post(
        "/chat", headers=api_headers, json={"query": "How many leave days?", "stream": False}
    )
    assert response.status_code == 200
    body = response.json()
    assert "18 days" in body["answer"]
    assert len(body["citations"]) == 1
