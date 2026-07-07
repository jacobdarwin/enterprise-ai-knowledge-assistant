from app.core.domain.models import Chunk, RetrievedChunk
from app.retrieval.rrf import reciprocal_rank_fusion


def _mk(chunk_id: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, document_id="doc1", filename="f.txt", content=f"content {chunk_id}")


def test_chunk_ranked_first_in_both_lists_wins():
    a, b = _mk("A"), _mk("B")
    dense = [RetrievedChunk(chunk=a, dense_score=0.9), RetrievedChunk(chunk=b, dense_score=0.8)]
    bm25 = [RetrievedChunk(chunk=a, bm25_score=5.0), RetrievedChunk(chunk=b, bm25_score=3.0)]

    fused = reciprocal_rank_fusion(dense, bm25, top_k=2)
    assert fused[0].chunk.chunk_id == "A"


def test_high_bm25_rank_can_overcome_lower_dense_rank():
    a, b = _mk("A"), _mk("B")
    # A ranks 1st in dense, 3rd(absent) in bm25; B ranks 2nd in dense, 1st in bm25
    dense = [RetrievedChunk(chunk=a, dense_score=0.9), RetrievedChunk(chunk=b, dense_score=0.8)]
    bm25 = [RetrievedChunk(chunk=b, bm25_score=5.0)]

    fused = reciprocal_rank_fusion(dense, bm25, dense_weight=0.5, bm25_weight=0.5, top_k=2)
    assert fused[0].chunk.chunk_id == "B"


def test_top_k_is_respected():
    chunks = [RetrievedChunk(chunk=_mk(str(i)), dense_score=1.0 / (i + 1)) for i in range(10)]
    fused = reciprocal_rank_fusion(chunks, [], top_k=3)
    assert len(fused) == 3


def test_chunk_only_in_one_list_still_included():
    a = _mk("A")
    dense = [RetrievedChunk(chunk=a, dense_score=0.5)]
    fused = reciprocal_rank_fusion(dense, [], top_k=5)
    assert len(fused) == 1
    assert fused[0].chunk.chunk_id == "A"


def test_empty_inputs_return_empty():
    assert reciprocal_rank_fusion([], [], top_k=5) == []
