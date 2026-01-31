"""Hybrid retrieval (BM25 + Dense)."""

from __future__ import annotations

from typing import Dict, List, Tuple


def rrf_merge(
    bm25_results: List[Tuple[str, float]],
    dense_results: List[Tuple[str, float]],
    top_k: int,
    k: int = 60,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}

    for rank, (doc_id, _) in enumerate(bm25_results, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + bm25_weight * (1.0 / (k + rank))

    for rank, (doc_id, _) in enumerate(dense_results, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + dense_weight * (1.0 / (k + rank))

    merged = list(scores.items())
    merged.sort(key=lambda item: item[1], reverse=True)
    return merged[:top_k]
