"""
Assembles the LangGraph workflow:

    query_analysis -> retriever -> retrieval_grader --(sufficient?)--> context_builder -> answer_generator -> citation_generator -> END
                                          ^                    |
                                          |                    v (insufficient, attempts left)
                                          +---------- query_rewrite

This matches the spec's diagram 1:1: Query Analysis -> Retriever ->
Retrieval Grader -> Need More Retrieval? -> Context Builder -> Answer
Generator -> Citation Generator -> Final Response.
"""

from langgraph.graph import END, StateGraph

from app.agents.citation_agent import CitationAgent
from app.agents.critic_agent import CriticAgent
from app.agents.query_agent import QueryAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.writer_agent import WriterAgent
from app.graph.nodes import build_nodes, route_after_grading
from app.graph.state import GraphState


def build_rag_graph(
    query_agent: QueryAgent,
    retriever_agent: RetrieverAgent,
    critic_agent: CriticAgent,
    writer_agent: WriterAgent,
    citation_agent: CitationAgent,
    default_top_k: int = 10,
):
    nodes = build_nodes(
        query_agent=query_agent,
        retriever_agent=retriever_agent,
        critic_agent=critic_agent,
        writer_agent=writer_agent,
        citation_agent=citation_agent,
        default_top_k=default_top_k,
    )

    graph = StateGraph(GraphState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.set_entry_point("query_analysis")
    graph.add_edge("query_analysis", "retriever")
    graph.add_edge("retriever", "retrieval_grader")
    graph.add_conditional_edges(
        "retrieval_grader",
        route_after_grading,
        {"context_builder": "context_builder", "query_rewrite": "query_rewrite"},
    )
    graph.add_edge("query_rewrite", "retriever")  # loop back for a second retrieval pass
    graph.add_edge("context_builder", "answer_generator")
    graph.add_edge("answer_generator", "citation_generator")
    graph.add_edge("citation_generator", END)

    return graph.compile()
