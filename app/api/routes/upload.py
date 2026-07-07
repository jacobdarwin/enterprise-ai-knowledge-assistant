import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, Request, UploadFile

from app.api.dependencies import get_document_repository, get_hybrid_retriever, get_vector_store
from app.api.middleware.file_validation import validate_upload
from app.api.middleware.rate_limit import limiter
from app.api.middleware.security import require_api_key
from app.core.domain.models import Document
from app.repositories.sqlite_document_repository import SQLiteDocumentRepository
from app.services.ingestion_service import IngestionService

router = APIRouter(tags=["upload"], dependencies=[Depends(require_api_key)])


@router.post("/upload", response_model=Document)
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    file: UploadFile,
    document_repository: SQLiteDocumentRepository = Depends(get_document_repository),
):
    # Validate BEFORE constructing the ingestion service — get_vector_store()/
    # get_hybrid_retriever() lazily load the embedding model on first call,
    # which is expensive and pointless work for a file we're about to reject.
    contents = await validate_upload(file)

    ingestion_service = IngestionService(
        document_repository=document_repository,
        vector_store=get_vector_store(),
        hybrid_retriever=get_hybrid_retriever(),
    )

    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        document = await ingestion_service.ingest(tmp_path, original_filename=file.filename)
    finally:
        tmp_path.unlink(missing_ok=True)

    return document
