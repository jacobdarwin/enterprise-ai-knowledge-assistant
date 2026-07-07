from abc import ABC, abstractmethod
from typing import List, Optional

from app.core.domain.models import ChatMessage, Document


class DocumentRepository(ABC):
    """Port for document metadata persistence (repository pattern)."""

    @abstractmethod
    async def create(self, document: Document) -> Document: ...

    @abstractmethod
    async def get(self, document_id: str) -> Optional[Document]: ...

    @abstractmethod
    async def list_all(self) -> List[Document]: ...

    @abstractmethod
    async def update_status(
        self, document_id: str, status: str, num_chunks: int = 0, error_message: Optional[str] = None
    ) -> None: ...

    @abstractmethod
    async def delete(self, document_id: str) -> None: ...


class ChatRepository(ABC):
    """Port for conversation history persistence."""

    @abstractmethod
    async def add_message(self, message: ChatMessage) -> None: ...

    @abstractmethod
    async def get_conversation(self, conversation_id: str) -> List[ChatMessage]: ...

    @abstractmethod
    async def list_conversations(self) -> List[str]: ...
