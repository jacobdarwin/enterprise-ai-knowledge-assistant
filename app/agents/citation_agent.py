from typing import List

from app.core.domain.models import Citation, RetrievedChunk

SNIPPET_MAX_CHARS = 160


class CitationAgent:
    """Builds structured Citation objects from the chunks the Writer Agent
    was given. Deliberately NOT LLM-based — citations are derived directly
    from retrieval metadata (filename, page, chunk_id), so they can never
    be fabricated or drift from what was actually retrieved. This is the
    piece of the "never hallucinate... cite filenames and page numbers"
    requirement that the LLM is never trusted to get exactly right."""

    def build_citations(self, graded_chunks: List[RetrievedChunk]) -> List[Citation]:
        citations: List[Citation] = []
        seen = set()
        for rc in graded_chunks:
            key = (rc.chunk.filename, rc.chunk.page)
            if key in seen:
                continue
            seen.add(key)
            snippet = rc.chunk.content[:SNIPPET_MAX_CHARS].strip()
            if len(rc.chunk.content) > SNIPPET_MAX_CHARS:
                snippet += "..."
            citations.append(
                Citation(
                    filename=rc.chunk.filename,
                    page=rc.chunk.page,
                    chunk_id=rc.chunk.chunk_id,
                    snippet=snippet,
                )
            )
        return citations
