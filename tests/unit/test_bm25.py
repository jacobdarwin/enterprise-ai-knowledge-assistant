from app.core.domain.models import Chunk
from app.retrieval.bm25_retriever import BM25Index


def _mk(content: str, chunk_id: str = "c") -> Chunk:
    return Chunk(chunk_id=chunk_id, document_id="doc1", filename="f.txt", content=content)


def test_exact_keyword_match_ranks_first():
    # Note: BM25's IDF term collapses to exactly zero for a word appearing
    # in 1 of just 2 documents (log((N-n+0.5)/(n+0.5)) = log(1) = 0 when
    # N=2, n=1) — a known property of BM25Okapi's math with tiny corpora,
    # not a bug. A realistic minimum-size corpus (4 docs) avoids the
    # degenerate case and reflects real usage.
    chunks = [
        _mk("Employees are entitled to 18 days of leave per year.", "leave"),
        _mk("The office coffee machine is broken.", "coffee"),
        _mk("Remote work requires manager approval.", "remote"),
        _mk("The parking garage closes at 9pm.", "parking"),
    ]
    index = BM25Index(chunks)
    results = index.search("leave days", top_k=2)
    assert results[0].chunk.chunk_id == "leave"


def test_zero_overlap_chunks_are_excluded():
    chunks = [_mk("Completely unrelated content about parking spaces.", "parking")]
    index = BM25Index(chunks)
    results = index.search("employee sick leave policy", top_k=5)
    assert results == []


def test_empty_index_returns_empty():
    index = BM25Index([])
    assert index.is_empty
    assert index.search("anything", top_k=5) == []


def test_top_k_is_respected():
    chunks = [_mk(f"leave policy detail number {i}", f"c{i}") for i in range(10)]
    index = BM25Index(chunks)
    results = index.search("leave policy", top_k=3)
    assert len(results) <= 3
