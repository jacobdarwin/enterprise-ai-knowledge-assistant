import json
import time
from uuid import uuid4

from app.agents.citation_agent import CitationAgent
from app.agents.critic_agent import CriticAgent
from app.agents.query_agent import QueryAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.writer_agent import WriterAgent
from app.core.config.logging_config import get_logger
from app.core.domain.models import ChatMessage, ChatRequest, ChatResponse, ChatRole
from app.core.interfaces.repositories import ChatRepository

log = get_logger(__name__)

MAX_HISTORY_MESSAGES_FOR_SUMMARY = 6


class ChatService:
    def __init__(
        self,
        chat_graph,
        chat_repository: ChatRepository,
        query_agent: QueryAgent,
        retriever_agent: RetrieverAgent,
        critic_agent: CriticAgent,
        writer_agent: WriterAgent,
        citation_agent: CitationAgent,
        max_retrieval_attempts: int = 2,
    ):
        self.chat_graph = chat_graph
        self.chat_repository = chat_repository
        self.query_agent = query_agent
        self.retriever_agent = retriever_agent
        self.critic_agent = critic_agent
        self.writer_agent = writer_agent
        self.citation_agent = citation_agent
        self.max_retrieval_attempts = max_retrieval_attempts

    async def _build_history_summary(self, conversation_id: str) -> str:
        history = await self.chat_repository.get_conversation(conversation_id)
        recent = history[-MAX_HISTORY_MESSAGES_FOR_SUMMARY:]
        return "\n".join(f"{m.role.value}: {m.content}" for m in recent)

    async def handle_chat(self, request: ChatRequest) -> ChatResponse:
        start = time.perf_counter()
        conversation_id = request.conversation_id or str(uuid4())

        history_summary = ""
        if request.conversation_id:
            history_summary = await self._build_history_summary(conversation_id)

        await self.chat_repository.add_message(
            ChatMessage(conversation_id=conversation_id, role=ChatRole.USER, content=request.query)
        )

        result = await self.chat_graph.ainvoke(
            {
                "query": request.query,
                "document_ids": request.document_ids,
                "history_summary": history_summary,
                "top_k": request.top_k,
                "retrieval_attempts": 0,
                "max_retrieval_attempts": self.max_retrieval_attempts,
            }
        )

        answer = result.get("answer", "I don't know based on the available documents.")
        citations = result.get("citations", [])

        await self.chat_repository.add_message(
            ChatMessage(
                conversation_id=conversation_id, role=ChatRole.ASSISTANT, content=answer, citations=citations
            )
        )

        latency_ms = (time.perf_counter() - start) * 1000
        log.info("chat_handled", conversation_id=conversation_id, latency_ms=round(latency_ms, 1))

        return ChatResponse(
            conversation_id=conversation_id,
            answer=answer,
            citations=citations,
            retrieval_count=len(result.get("graded_chunks", [])),
            latency_ms=latency_ms,
        )

    async def handle_chat_stream(self, request: ChatRequest):
        """
        Server-Sent-Events generator for POST /chat when stream=True.
        Runs retrieval + grading (with the same retry-once policy as the
        graph) directly via the agents, then streams the Writer Agent's
        token deltas, and finishes with a `done` event carrying citations —
        so the frontend can render sources only once the full answer (and
        therefore the full citation set) is known.
        """
        conversation_id = request.conversation_id or str(uuid4())
        history_summary = await self._build_history_summary(conversation_id) if request.conversation_id else ""

        await self.chat_repository.add_message(
            ChatMessage(conversation_id=conversation_id, role=ChatRole.USER, content=request.query)
        )

        query = self.query_agent.analyze(request.query)
        original_query = query
        graded: list = []

        for attempt in range(1, self.max_retrieval_attempts + 1):
            candidates = await self.retriever_agent.retrieve(
                query=query, top_k=request.top_k or 10, document_ids=request.document_ids
            )
            graded = await self.critic_agent.grade(original_query, candidates)
            if self.critic_agent.is_sufficient(graded) or attempt == self.max_retrieval_attempts:
                break
            query = await self.query_agent.rewrite_for_retry(original_query, "Try different or broader keywords.")

        full_answer = ""
        async for delta in self.writer_agent.write_stream(original_query, graded, history_summary):
            full_answer += delta
            yield f"event: token\ndata: {json.dumps({'delta': delta})}\n\n"

        citations = self.citation_agent.build_citations(graded)

        await self.chat_repository.add_message(
            ChatMessage(
                conversation_id=conversation_id, role=ChatRole.ASSISTANT, content=full_answer, citations=citations
            )
        )

        done_payload = {
            "conversation_id": conversation_id,
            "citations": [c.model_dump() for c in citations],
        }
        yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
