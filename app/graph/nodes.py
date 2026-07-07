"""
LangGraph node functions.

Each node is an async function of (GraphState) -> partial state update,
the signature LangGraph expects. All actual logic lives in the agents
(app/agents/) — nodes exist purely to adapt agent calls to the graph's
state-in/state-out contract, and to log the workflow's progress step by
step (useful for the LangSmith trace view).

Nodes are built by `build_nodes(...)` so they close over agent instances
without relying on globals — keeps this composable and testable.
"""

from typing import Callable, Dict

from app.agents.citation_agent import CitationAgent
from app.agents.critic_agent import CriticAgent
from app.agents.query_agent import QueryAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.writer_agent import WriterAgent
from app.core.config.logging_config import get_logger
from app.graph.state import GraphState

log = get_logger(__name__)


def build_nodes(
    query_agent: QueryAgent,
    retriever_agent: RetrieverAgent,
    critic_agent: CriticAgent,
    writer_agent: WriterAgent,
    citation_agent: CitationAgent,
    default_top_k: int,
) -> Dict[str, Callable]:
    async def query_analysis_node(state: GraphState) -> dict:
        analyzed = query_agent.analyze(state["query"])
        log.info("graph_node", node="query_analysis", query=analyzed)
        return {"query": analyzed, "original_query": state.get("original_query", analyzed)}

    async def retriever_node(state: GraphState) -> dict:
        chunks = await retriever_agent.retrieve(
            query=state["query"],
            top_k=state.get("top_k") or default_top_k,
            document_ids=state.get("document_ids"),
        )
        log.info("graph_node", node="retriever", num_candidates=len(chunks))
        return {"retrieved_chunks": chunks}

    async def retrieval_grader_node(state: GraphState) -> dict:
        graded = await critic_agent.grade(state["query"], state.get("retrieved_chunks", []))
        sufficient = critic_agent.is_sufficient(graded)
        attempts = state.get("retrieval_attempts", 0) + 1
        log.info(
            "graph_node", node="retrieval_grader", graded=len(graded), sufficient=sufficient, attempts=attempts
        )
        return {"graded_chunks": graded, "retrieval_sufficient": sufficient, "retrieval_attempts": attempts}

    async def query_rewrite_node(state: GraphState) -> dict:
        """Runs only when the grader says retrieval was insufficient AND we
        haven't hit max_retrieval_attempts yet (see routing function below)."""
        rewritten = await query_agent.rewrite_for_retry(
            state.get("original_query", state["query"]),
            note="Try different or broader keywords.",
        )
        log.info("graph_node", node="query_rewrite", rewritten_query=rewritten)
        return {"query": rewritten}

    async def context_builder_node(state: GraphState) -> dict:
        # Context assembly itself happens inside build_answer_messages at
        # generation time; this node is a named pass-through step so the
        # graph's shape matches the spec 1:1 and LangSmith traces show it
        # as a distinct stage.
        log.info("graph_node", node="context_builder", chunks=len(state.get("graded_chunks", [])))
        return {}

    async def answer_generator_node(state: GraphState) -> dict:
        answer = await writer_agent.write(
            query=state.get("original_query", state["query"]),
            graded_chunks=state.get("graded_chunks", []),
            history_summary=state.get("history_summary", ""),
        )
        log.info("graph_node", node="answer_generator", answer_len=len(answer))
        return {"answer": answer}

    async def citation_generator_node(state: GraphState) -> dict:
        citations = citation_agent.build_citations(state.get("graded_chunks", []))
        log.info("graph_node", node="citation_generator", num_citations=len(citations))
        return {"citations": citations}

    return {
        "query_analysis": query_analysis_node,
        "retriever": retriever_node,
        "retrieval_grader": retrieval_grader_node,
        "query_rewrite": query_rewrite_node,
        "context_builder": context_builder_node,
        "answer_generator": answer_generator_node,
        "citation_generator": citation_generator_node,
    }


def route_after_grading(state: GraphState) -> str:
    """Conditional edge: the spec's 'Need More Retrieval?' decision point.
    Retries at most once (max_retrieval_attempts, default 2 total attempts)
    to avoid infinite loops when the corpus genuinely lacks the answer."""
    if state.get("retrieval_sufficient", False):
        return "context_builder"
    if state.get("retrieval_attempts", 0) >= state.get("max_retrieval_attempts", 2):
        return "context_builder"  # give up gracefully — Writer will emit "I don't know"
    return "query_rewrite"
