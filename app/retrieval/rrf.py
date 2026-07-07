"""
Reciprocal Rank Fusion (RRF).

RRF combines two differently-scaled ranked lists (cosine similarity from
dense search vs. BM25's unbounded term-frequency scores) without needing
to normalize either score to a common scale — a classic problem with
naive weighted-score fusion. Instead, RRF only looks at *rank position*
in each list:

    fused_score(doc) = sum over lists L containing doc of:  weight_L / (k + rank_L(doc))

k=60 is the standard constant from the original RRF paper (Cormack et al.),
chosen because it de-emphasizes the exact top rank mattering too much
while still rewarding high positions. We additionally apply the
dense_weight/bm25_weight from settings so the person can bias fusion
toward whichever retrieval mode performs better for their document set.
"""

from typing import Dict, List

from app.core.domain.models import RetrievedChunk

RRF_K = 60


def reciprocal_rank_fusion(
    dense_results: List[RetrievedChunk],
    bm25_results: List[RetrievedChunk],
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4,
    top_k: int = 10,
) -> List[RetrievedChunk]:
    fused: Dict[str, RetrievedChunk] = {}
    scores: Dict[str, float] = {}

    def _accumulate(results: List[RetrievedChunk], weight: float, score_field: str):
        for rank, retrieved in enumerate(results, start=1):
            chunk_id = retrieved.chunk.chunk_id
            contribution = weight / (RRF_K + rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution

            if chunk_id not in fused:
                fused[chunk_id] = retrieved
            else:
                # Merge in whichever score (dense_score/bm25_score) this
                # list carries, so the final object shows both signals.
                existing = fused[chunk_id]
                if score_field == "dense_score":
                    existing.dense_score = retrieved.dense_score
                else:
                    existing.bm25_score = retrieved.bm25_score

    _accumulate(dense_results, dense_weight, "dense_score")
    _accumulate(bm25_results, bm25_weight, "bm25_score")

    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]

    output: List[RetrievedChunk] = []
    for chunk_id in ranked_ids:
        item = fused[chunk_id]
        item.fused_score = scores[chunk_id]
        output.append(item)
    return output
