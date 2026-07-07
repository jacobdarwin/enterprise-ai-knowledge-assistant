from fastapi import APIRouter, Depends

from app.api.dependencies import get_document_repository, get_vector_store
from app.mcp.registry import get_all_server_specs
from app.repositories.sqlite_document_repository import SQLiteDocumentRepository

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def get_metrics(document_repository: SQLiteDocumentRepository = Depends(get_document_repository)):
    documents = await document_repository.list_all()
    status_breakdown: dict = {}
    total_chunks = 0
    for doc in documents:
        status_breakdown[doc.status.value] = status_breakdown.get(doc.status.value, 0) + 1
        total_chunks += doc.num_chunks

    vector_store = get_vector_store()
    vector_count = vector_store._collection.count()  # Chroma-specific; fine for a metrics endpoint

    mcp_servers = {
        name: {"enabled": spec.enabled, "description": spec.description}
        for name, spec in get_all_server_specs().items()
    }

    return {
        "total_documents": len(documents),
        "documents_by_status": status_breakdown,
        "total_chunks_indexed": total_chunks,
        "vectors_in_store": vector_count,
        "mcp_servers": mcp_servers,
    }
