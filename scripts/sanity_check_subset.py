#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Set, Tuple


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def count_lines(path: Path, max_lines: int) -> Tuple[int, bool]:
    count = 0
    capped = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line:
                continue
            count += 1
            if count >= max_lines:
                capped = True
                break
    return count, capped


def count_qrels_lines(path: Path, max_lines: int) -> Tuple[int, bool]:
    count = 0
    capped = False
    with path.open("r", encoding="utf-8") as handle:
        first = True
        for line in handle:
            if not line:
                continue
            if first:
                first = False
                cols = line.rstrip("\n").split("\t")
                if cols and cols[0].strip().lower() == "query-id":
                    continue
            count += 1
            if count >= max_lines:
                capped = True
                break
    return count, capped


def load_query_ids(path: Path, max_lines: int) -> Tuple[Set[str], bool]:
    ids: Set[str] = set()
    capped = False
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = obj.get("_id")
            if qid is None:
                raise ValueError(f"Missing _id in queries file {path}")
            ids.add(str(qid))
            if idx >= max_lines:
                capped = True
                break
    return ids, capped


def load_corpus_ids(path: Path, max_lines: int) -> Tuple[Set[str], bool]:
    ids: Set[str] = set()
    capped = False
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("_id")
            if cid is None:
                raise ValueError(f"Missing _id in corpus file {path}")
            ids.add(str(cid))
            if idx >= max_lines:
                capped = True
                break
    return ids, capped


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanity check MSMARCO subset files.")
    parser.add_argument("--subset_dir", default="data/msmarco_subset")
    parser.add_argument("--max_lines", type=int, default=200000)
    args = parser.parse_args()

    subset_dir = Path(args.subset_dir)
    corpus_path = subset_dir / "corpus.jsonl"
    queries_path = subset_dir / "queries.jsonl"
    qrels_path = subset_dir / "qrels.tsv"

    missing = [p.name for p in (corpus_path, queries_path, qrels_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing subset files in {subset_dir}: {', '.join(missing)}")

    corpus_count, corpus_capped = count_lines(corpus_path, args.max_lines)
    query_count, query_capped = count_lines(queries_path, args.max_lines)
    qrels_count, qrels_capped = count_qrels_lines(qrels_path, args.max_lines)

    if corpus_capped or query_capped or qrels_capped:
        warn("Line counts capped by --max_lines; results may be incomplete.")

    query_ids, query_ids_capped = load_query_ids(queries_path, args.max_lines)
    corpus_ids, corpus_ids_capped = load_corpus_ids(corpus_path, args.max_lines)
    if query_ids_capped or corpus_ids_capped:
        warn("ID loading capped by --max_lines; qrels validation may be incomplete.")

    broken_rows = 0
    qrels_queries: Set[str] = set()
    qrels_rows = 0
    score_counts: Dict[str, int] = Counter()
    with qrels_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            qrels_rows += 1
            qid = row.get("query-id")
            cid = row.get("corpus-id")
            score = row.get("score")
            if not qid or not cid:
                broken_rows += 1
                continue
            if qid not in query_ids or cid not in corpus_ids:
                broken_rows += 1
            qrels_queries.add(qid)
            if score is not None:
                score_counts[str(score)] += 1

    if qrels_count == 0 or broken_rows > 0:
        print(f"corpus_count={corpus_count}, query_count={query_count}, qrels_count={qrels_count}")
        print(f"broken_qrels_rows={broken_rows}")
        raise SystemExit(1)

    qrels_query_count = len(qrels_queries)
    fraction_with_qrels = qrels_query_count / query_count if query_count else 0.0
    avg_qrels = qrels_rows / qrels_query_count if qrels_query_count else 0.0
    score_items = sorted(score_counts.items(), key=lambda item: (-item[1], item[0]))
    top_scores = ", ".join(f"{score}={count}" for score, count in score_items[:10]) or "none"

    print(f"corpus_count={corpus_count}, query_count={query_count}, qrels_count={qrels_count}")
    print(f"broken_qrels_rows={broken_rows}")
    print(f"unique_qrels_queries={qrels_query_count}")
    print(f"fraction_queries_with_qrels={fraction_with_qrels:.4f}")
    print(f"avg_qrels_per_qrels_query={avg_qrels:.4f}")
    print(f"score_counts_top={top_scores}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
