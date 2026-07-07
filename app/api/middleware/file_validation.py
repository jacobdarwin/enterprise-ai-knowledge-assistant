from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from app.core.config.settings import get_settings


async def validate_upload(file: UploadFile) -> bytes:
    """Reads and validates an uploaded file. Returns the raw bytes if valid,
    raises HTTPException otherwise. Reading here (rather than trusting
    Content-Length) means we enforce the real size limit even if a client
    lies about the header."""
    settings = get_settings()

    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {settings.allowed_extensions_list}",
        )

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    contents = await file.read()
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.max_upload_size_mb}MB limit.",
        )
    if len(contents) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    return contents
