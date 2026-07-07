from typing import List

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_document_repository, get_ingestion_service
from app.api.middleware.security import require_api_key
from app.core.domain.models import Document
from app.repositories.sqlite_document_repository import SQLiteDocumentRepository
from app.services.ingestion_service import IngestionService

router = APIRouter(tags=["documents"], dependencies=[Depends(require_api_key)])


@router.get("/documents", response_model=List[Document])
async def list_documents(document_repository: SQLiteDocumentRepository = Depends(get_document_repository)):
    return await document_repository.list_all()


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    document_repository: SQLiteDocumentRepository = Depends(get_document_repository),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
):
    existing = await document_repository.get(document_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")

    await ingestion_service.delete_document(document_id)
    return {"deleted": True, "document_id": document_id}
