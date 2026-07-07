from typing import List, Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.domain.models import Document, DocumentStatus
from app.core.interfaces.repositories import DocumentRepository
from app.repositories.database import DocumentRow


def _row_to_domain(row: DocumentRow) -> Document:
    return Document(
        document_id=row.document_id,
        filename=row.filename,
        file_type=row.file_type,
        upload_time=row.upload_time,
        status=DocumentStatus(row.status),
        num_chunks=row.num_chunks,
        size_bytes=row.size_bytes,
        error_message=row.error_message,
    )


class SQLiteDocumentRepository(DocumentRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, document: Document) -> Document:
        row = DocumentRow(
            document_id=document.document_id,
            filename=document.filename,
            file_type=document.file_type,
            upload_time=document.upload_time,
            status=document.status.value,
            num_chunks=document.num_chunks,
            size_bytes=document.size_bytes,
            error_message=document.error_message,
        )
        self.session.add(row)
        await self.session.commit()
        return document

    async def get(self, document_id: str) -> Optional[Document]:
        result = await self.session.execute(select(DocumentRow).where(DocumentRow.document_id == document_id))
        row = result.scalar_one_or_none()
        return _row_to_domain(row) if row else None

    async def list_all(self) -> List[Document]:
        result = await self.session.execute(select(DocumentRow).order_by(DocumentRow.upload_time.desc()))
        return [_row_to_domain(row) for row in result.scalars().all()]

    async def update_status(
        self, document_id: str, status: str, num_chunks: int = 0, error_message: Optional[str] = None
    ) -> None:
        result = await self.session.execute(select(DocumentRow).where(DocumentRow.document_id == document_id))
        row = result.scalar_one_or_none()
        if row is None:
            return
        row.status = status
        if num_chunks:
            row.num_chunks = num_chunks
        if error_message is not None:
            row.error_message = error_message
        await self.session.commit()

    async def delete(self, document_id: str) -> None:
        await self.session.execute(delete(DocumentRow).where(DocumentRow.document_id == document_id))
        await self.session.commit()
