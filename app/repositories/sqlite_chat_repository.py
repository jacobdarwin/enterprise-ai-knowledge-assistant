import json
from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.domain.models import ChatMessage, ChatRole, Citation
from app.core.interfaces.repositories import ChatRepository
from app.repositories.database import ChatMessageRow


def _row_to_domain(row: ChatMessageRow) -> ChatMessage:
    return ChatMessage(
        message_id=row.message_id,
        conversation_id=row.conversation_id,
        role=ChatRole(row.role),
        content=row.content,
        citations=[Citation(**c) for c in json.loads(row.citations_json)],
        created_at=row.created_at,
    )


class SQLiteChatRepository(ChatRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_message(self, message: ChatMessage) -> None:
        row = ChatMessageRow(
            message_id=message.message_id,
            conversation_id=message.conversation_id,
            role=message.role.value,
            content=message.content,
            citations_json=json.dumps([c.model_dump() for c in message.citations]),
            created_at=message.created_at,
        )
        self.session.add(row)
        await self.session.commit()

    async def get_conversation(self, conversation_id: str) -> List[ChatMessage]:
        result = await self.session.execute(
            select(ChatMessageRow)
            .where(ChatMessageRow.conversation_id == conversation_id)
            .order_by(ChatMessageRow.created_at.asc())
        )
        return [_row_to_domain(row) for row in result.scalars().all()]

    async def list_conversations(self) -> List[str]:
        result = await self.session.execute(select(ChatMessageRow.conversation_id).distinct())
        return [row[0] for row in result.all()]
