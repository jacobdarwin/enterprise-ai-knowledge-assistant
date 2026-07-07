from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_chat_service
from app.api.middleware.input_sanitization import flag_suspicious_query, sanitize_query
from app.api.middleware.rate_limit import limiter
from app.api.middleware.security import require_api_key
from app.core.domain.models import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(tags=["chat"], dependencies=[Depends(require_api_key)])


@router.post("/chat")
@limiter.limit("30/minute")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    chat_request.query = sanitize_query(chat_request.query)
    flag_suspicious_query(chat_request.query)

    if chat_request.stream:
        return StreamingResponse(
            chat_service.handle_chat_stream(chat_request),
            media_type="text/event-stream",
        )

    response: ChatResponse = await chat_service.handle_chat(chat_request)
    return response
