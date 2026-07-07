# Architecture Deep Dive

See the main [README.md](../README.md) for the high-level diagrams. This document covers
the reasoning behind key design decisions.

## Why Clean Architecture / ports-and-adapters

Every external dependency (LLM provider, vector database, relational database, reranker)
is defined as an abstract interface in `app/core/interfaces/` before any concrete
implementation exists. This was validated concretely during development: the `VectorStore`
interface was written once in Phase 1, then `ChromaVectorStore` (Phase 2) implemented it
without any changes to the interface — and the retrieval/agent/graph layers built in
Phases 3-4 never had to know Chroma existed, only that *some* `VectorStore` did.

## Why LangGraph instead of a linear chain

The spec's retrieval-grading step needs a conditional retry loop: if the Critic Agent
judges the retrieved context insufficient, the Query Agent rewrites the query and
retrieval runs again — but only once, to avoid infinite loops when the corpus genuinely
lacks the answer. LangGraph's conditional edges (`route_after_grading` in
`app/graph/nodes.py`) express this naturally as a graph rather than nested if/else logic
in a single function.

## Why citations are never LLM-generated

`app/agents/citation_agent.py` builds citations directly from retrieval metadata
(filename, page, chunk_id) — never by asking the LLM to report its own sources. This
guarantees citations can't be fabricated: they are a deterministic transformation of
what was actually retrieved, not a claim the model makes about itself.

## Why Reciprocal Rank Fusion over score normalization

Dense retrieval produces cosine similarities (0-1, bounded) while BM25 produces
unbounded term-frequency scores. Naively averaging or weighting these two scales
produces meaningless numbers. RRF (`app/retrieval/rrf.py`) sidesteps this entirely by
only considering *rank position* in each list — a chunk that's #1 in both lists always
outranks a chunk that's #1 in only one, regardless of the underlying score scales.
