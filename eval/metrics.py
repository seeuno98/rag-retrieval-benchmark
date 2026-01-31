"""Evaluation metrics for retrieval."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple


def recall_at_k(results: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = results[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def mrr_at_k(results: Sequence[str], relevant: Set[str], k: int) -> float:
    for rank, doc_id in enumerate(results[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def compute_metrics(
    per_query_results: Dict[str, List[str]],
    qrels: Dict[str, Set[str]],
    k_values: Iterable[int],
    mrr_k: int,
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    counts = 0
    for qid, results in per_query_results.items():
        relevant = qrels.get(qid, set())
        for k in k_values:
            key = f"recall@{k}"
            totals[key] = totals.get(key, 0.0) + recall_at_k(results, relevant, k)
        totals["mrr"] = totals.get("mrr", 0.0) + mrr_at_k(results, relevant, mrr_k)
        counts += 1

    if counts == 0:
        return {key: 0.0 for key in totals}

    return {key: value / counts for key, value in totals.items()}
