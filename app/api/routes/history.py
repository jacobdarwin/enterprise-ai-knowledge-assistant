from typing import List, Optional

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_chat_repository
from app.api.middleware.security import require_api_key
from app.core.domain.models import ChatMessage
from app.repositories.sqlite_chat_repository import SQLiteChatRepository

router = APIRouter(tags=["history"], dependencies=[Depends(require_api_key)])


@router.get("/history")
async def get_history(
    conversation_id: Optional[str] = Query(default=None),
    chat_repository: SQLiteChatRepository = Depends(get_chat_repository),
):
    if conversation_id:
        messages: List[ChatMessage] = await chat_repository.get_conversation(conversation_id)
        return {"conversation_id": conversation_id, "messages": messages}

    conversations = await chat_repository.list_conversations()
    return {"conversations": conversations}
