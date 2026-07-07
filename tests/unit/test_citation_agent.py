from app.agents.citation_agent import CitationAgent
from app.core.domain.models import Chunk, RetrievedChunk


def _mk_retrieved(filename: str, page: int, content: str, chunk_id: str) -> RetrievedChunk:
    chunk = Chunk(chunk_id=chunk_id, document_id="doc1", filename=filename, page=page, content=content)
    return RetrievedChunk(chunk=chunk)


def test_builds_one_citation_per_chunk():
    agent = CitationAgent()
    chunks = [
        _mk_retrieved("handbook.txt", 1, "Leave policy details here.", "c1"),
        _mk_retrieved("handbook.txt", 3, "Remote work details here.", "c2"),
    ]
    citations = agent.build_citations(chunks)
    assert len(citations) == 2
    assert citations[0].filename == "handbook.txt"
    assert citations[0].page == 1


def test_deduplicates_same_filename_and_page():
    agent = CitationAgent()
    chunks = [
        _mk_retrieved("handbook.txt", 1, "First chunk on page 1.", "c1"),
        _mk_retrieved("handbook.txt", 1, "Second chunk also on page 1.", "c2"),
    ]
    citations = agent.build_citations(chunks)
    assert len(citations) == 1  # same (filename, page) key


def test_snippet_is_truncated_for_long_content():
    agent = CitationAgent()
    long_content = "A" * 500
    chunks = [_mk_retrieved("f.txt", None, long_content, "c1")]
    citations = agent.build_citations(chunks)
    assert len(citations[0].snippet) <= 165  # SNIPPET_MAX_CHARS + "..."
    assert citations[0].snippet.endswith("...")


def test_empty_input_returns_empty():
    agent = CitationAgent()
    assert agent.build_citations([]) == []
