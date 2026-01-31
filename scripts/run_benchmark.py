#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from eval.metrics import compute_metrics  # noqa: E402
from retrieval.bm25 import BM25Index  # noqa: E402
from retrieval.dense import DenseIndex  # noqa: E402
from retrieval.hybrid import rrf_merge  # noqa: E402
from scripts.prepare_data import prepare_dataset  # noqa: E402


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def load_corpus(path: Path) -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id") or obj.get("_id")
            text = obj.get("text") or obj.get("contents")
            if doc_id is None or text is None:
                raise ValueError(f"Malformed corpus line: {line!r}")
            docs.append((str(doc_id), str(text)))
    return docs


def load_queries(path: Path) -> List[Tuple[str, str]]:
    queries: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = obj.get("qid") or obj.get("_id")
            query = obj.get("query") or obj.get("text")
            if qid is None or query is None:
                raise ValueError(f"Malformed query line: {line!r}")
            queries.append((str(qid), str(query)))
    return queries


def load_qrels(path: Path) -> Dict[str, Set[str]]:
    qrels: Dict[str, Set[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            qid = row.get("qid") or row.get("query-id") or row.get("query_id")
            doc_id = row.get("doc_id") or row.get("corpus-id") or row.get("corpus_id") or row.get("doc-id")
            rel = row.get("relevance") or row.get("score") or row.get("rel")
            if not qid or not doc_id:
                continue
            if rel is not None and float(rel) <= 0:
                continue
            qrels.setdefault(str(qid), set()).add(str(doc_id))
    return qrels


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def format_markdown_table(rows: List[Dict[str, str]]) -> str:
    headers = ["method", "recall@5", "recall@10", "mrr@10", "p50_ms", "p90_ms"]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run retrieval benchmark end-to-end.")
    parser.add_argument("--dataset", default="mini")
    parser.add_argument("--methods", default="bm25,dense,hybrid")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dense_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--hybrid_k", type=int, default=60)
    parser.add_argument("--rerank", action="store_true", help="Placeholder flag for reranking stage.")
    args = parser.parse_args()

    set_seeds(args.seed)

    data_dir = ROOT_DIR / "data"
    dataset_dir = prepare_dataset(args.dataset, data_dir)
    corpus = load_corpus(dataset_dir / "corpus.jsonl")
    queries = load_queries(dataset_dir / "queries.jsonl")
    qrels = load_qrels(dataset_dir / "qrels.tsv")

    print(f"Loaded docs={len(corpus)}, queries={len(queries)}, qrels={sum(len(v) for v in qrels.values())}")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bm25_index = None
    dense_index = None

    if "bm25" in methods or "hybrid" in methods:
        bm25_index = BM25Index.build(corpus)
    if "dense" in methods or "hybrid" in methods:
        dense_index = DenseIndex.build(corpus, model_name=args.dense_model)

    results_rows: List[Dict[str, str]] = []
    results_json: Dict[str, Dict[str, float]] = {}

    for method in methods:
        per_query_results: Dict[str, List[str]] = {}
        retrieval_times: List[float] = []
        total_times: List[float] = []
        rerank_times: List[float] = []

        for qid, query in queries:
            start_total = time.perf_counter()
            start_retrieval = time.perf_counter()

            if method == "bm25":
                assert bm25_index is not None
                scored = bm25_index.retrieve(query, args.top_k)
            elif method == "dense":
                assert dense_index is not None
                scored = dense_index.retrieve(query, args.top_k)
            elif method == "hybrid":
                assert bm25_index is not None and dense_index is not None
                bm25_scored = bm25_index.retrieve(query, args.top_k * 5)
                dense_scored = dense_index.retrieve(query, args.top_k * 5)
                scored = rrf_merge(
                    bm25_scored,
                    dense_scored,
                    top_k=args.top_k,
                    k=args.hybrid_k,
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            retrieval_ms = (time.perf_counter() - start_retrieval) * 1000.0
            rerank_ms = 0.0
            if args.rerank:
                rerank_ms = 0.0
            total_ms = (time.perf_counter() - start_total) * 1000.0

            retrieval_times.append(retrieval_ms)
            rerank_times.append(rerank_ms)
            total_times.append(total_ms)

            per_query_results[qid] = [doc_id for doc_id, _ in scored]

        metrics = compute_metrics(per_query_results, qrels, k_values=[5, 10], mrr_k=10)
        p50 = percentile(total_times, 50)
        p90 = percentile(total_times, 90)

        results_json[method] = {
            "recall@5": metrics.get("recall@5", 0.0),
            "recall@10": metrics.get("recall@10", 0.0),
            "mrr@10": metrics.get("mrr", 0.0),
            "latency_p50_ms": p50,
            "latency_p90_ms": p90,
            "retrieval_p50_ms": percentile(retrieval_times, 50),
            "retrieval_p90_ms": percentile(retrieval_times, 90),
            "rerank_p50_ms": percentile(rerank_times, 50),
            "rerank_p90_ms": percentile(rerank_times, 90),
        }

        results_rows.append(
            {
                "method": method,
                "recall@5": f"{results_json[method]['recall@5']:.4f}",
                "recall@10": f"{results_json[method]['recall@10']:.4f}",
                "mrr@10": f"{results_json[method]['mrr@10']:.4f}",
                "p50_ms": f"{results_json[method]['latency_p50_ms']:.2f}",
                "p90_ms": f"{results_json[method]['latency_p90_ms']:.2f}",
            }
        )

    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "dataset": args.dataset,
        "methods": methods,
        "top_k": args.top_k,
        "seed": args.seed,
        "dense_model": args.dense_model,
        "hybrid_k": args.hybrid_k,
        "rerank": args.rerank,
    }

    results_payload = {
        "config": config,
        "metrics": results_json,
    }

    with (results_dir / "results_latest.json").open("w", encoding="utf-8") as handle:
        json.dump(results_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with (results_dir / "results_latest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results_rows[0].keys())
        writer.writeheader()
        writer.writerows(results_rows)

    markdown_table = format_markdown_table(results_rows)
    with (results_dir / "results_latest.md").open("w", encoding="utf-8") as handle:
        handle.write(markdown_table + "\n")

    print(markdown_table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
