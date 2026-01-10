#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def find_single_file(root: Path, name: str) -> Path:
    direct = root / name
    if direct.exists():
        return direct
    matches = sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {root}")
    if len(matches) > 1:
        warn(f"Multiple {name} files found; using {matches[0]}")
    return matches[0]


def find_qrels_file(root: Path, split: str) -> Tuple[Path, str]:
    qrels_dir = root / "qrels"
    if qrels_dir.exists():
        split_path = qrels_dir / f"{split}.tsv"
        if split_path.exists():
            return split_path, split
        candidates = sorted(qrels_dir.glob("*.tsv"))
        if candidates:
            chosen = candidates[0]
            return chosen, chosen.stem
    candidates = sorted(root.rglob("*.tsv"))
    qrels_candidates = [p for p in candidates if "qrels" in p.name.lower()]
    if qrels_candidates:
        chosen = qrels_candidates[0]
        return chosen, chosen.stem
    if candidates:
        warn(f"No qrels TSV named {split}.tsv found; using {candidates[0]}")
        return candidates[0], candidates[0].stem
    raise FileNotFoundError(f"Could not find qrels TSV under {root}")


def read_query_ids(path: Path) -> List[str]:
    ids: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("_id")
            if qid is None:
                raise ValueError(f"Missing _id in queries file {path}")
            ids.append(str(qid))
    return ids


def iter_queries(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = obj.get("_id")
            if qid is None:
                raise ValueError(f"Missing _id in queries file {path}")
            yield str(qid), obj


def read_qrels(path: Path):
    header: Optional[str] = None
    rows: List[Tuple[str, str, str]] = []
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if header is None:
                cols = line.rstrip("\n").split("\t")
                if cols and cols[0].strip().lower() == "query-id":
                    header = line.rstrip("\n")
                    continue
            total += 1
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                raise ValueError(f"Malformed qrels line: {line!r}")
            qid, cid = cols[0], cols[1]
            rows.append((qid, cid, line.rstrip("\n")))
    return header, rows, total


def read_corpus_ids(path: Path) -> List[str]:
    ids: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("_id")
            if cid is None:
                raise ValueError(f"Missing _id in corpus file {path}")
            ids.append(str(cid))
    return ids


def write_queries_subset(path: Path, queries_path: Path, sampled_ids: Set[str]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for qid, obj in iter_queries(queries_path):
            if qid in sampled_ids:
                handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1
    return count


def write_corpus_subset(path: Path, corpus_path: Path, corpus_ids: Set[str]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        with corpus_path.open("r", encoding="utf-8") as corpus_handle:
            for line in corpus_handle:
                if not line.strip():
                    continue
                obj = json.loads(line)
                cid = obj.get("_id")
                if cid is None:
                    raise ValueError(f"Missing _id in corpus file {corpus_path}")
                if str(cid) in corpus_ids:
                    handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    count += 1
    return count


def write_qrels_subset(
    path: Path,
    header: Optional[str],
    rows: List[Tuple[str, str, str]],
    sampled_queries: Set[str],
    corpus_ids: Set[str],
) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        if header is not None:
            handle.write(header + "\n")
        for qid, cid, line in rows:
            if qid in sampled_queries and cid in corpus_ids:
                handle.write(line + "\n")
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Subsample BEIR MSMARCO passage dataset.")
    parser.add_argument("--input_dir", required=True, help="Path to extracted MSMARCO dataset dir")
    parser.add_argument("--output_dir", default="data/msmarco_subset")
    parser.add_argument("--corpus_target", type=int, default=100000)
    parser.add_argument("--queries_target", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = find_single_file(input_dir, "corpus.jsonl")
    queries_path = find_single_file(input_dir, "queries.jsonl")
    qrels_path, split_used = find_qrels_file(input_dir, args.split)

    qrels_header, qrels_rows, total_qrels = read_qrels(qrels_path)
    qrels_query_ids: List[str] = []
    qrels_query_seen: Set[str] = set()
    for qid, _, _ in qrels_rows:
        if qid not in qrels_query_seen:
            qrels_query_seen.add(qid)
            qrels_query_ids.append(qid)
    if not qrels_query_ids:
        raise ValueError(f"No queries found in qrels for split {split_used}; cannot sample queries.")

    all_query_ids = read_query_ids(queries_path)
    total_queries = len(all_query_ids)
    if args.queries_target > len(qrels_query_ids):
        warn(
            f"queries_target {args.queries_target} exceeds qrels queries {len(qrels_query_ids)}; "
            f"using {len(qrels_query_ids)}"
        )
        queries_target = len(qrels_query_ids)
    else:
        queries_target = args.queries_target

    rng = random.Random(args.seed)
    sampled_queries = set(rng.sample(qrels_query_ids, queries_target))

    qrels_after_query = [(qid, cid, line) for qid, cid, line in qrels_rows if qid in sampled_queries]
    if not qrels_after_query:
        raise ValueError(
            "Filtered qrels rows is 0; split may have no overlap or queries_target too small."
        )
    required_docs = {cid for _, cid, _ in qrels_after_query}

    all_corpus_ids = read_corpus_ids(corpus_path)
    total_corpus = len(all_corpus_ids)

    corpus_target = args.corpus_target
    if len(required_docs) > corpus_target:
        warn(
            f"Required docs {len(required_docs)} exceed corpus_target {corpus_target}; "
            f"using corpus_target={len(required_docs)}"
        )
        corpus_target = len(required_docs)

    if len(required_docs) == corpus_target:
        final_corpus = set(required_docs)
    else:
        remaining = [cid for cid in all_corpus_ids if cid not in required_docs]
        needed = corpus_target - len(required_docs)
        if needed > len(remaining):
            warn(
                f"corpus_target {corpus_target} exceeds total corpus {total_corpus}; using {total_corpus}"
            )
            needed = len(remaining)
            corpus_target = len(required_docs) + needed
        sampled_extra = rng.sample(remaining, needed)
        final_corpus = set(required_docs) | set(sampled_extra)

    queries_written = write_queries_subset(output_dir / "queries.jsonl", queries_path, sampled_queries)
    corpus_written = write_corpus_subset(output_dir / "corpus.jsonl", corpus_path, final_corpus)
    qrels_written = write_qrels_subset(
        output_dir / "qrels.tsv",
        qrels_header,
        qrels_rows,
        sampled_queries,
        final_corpus,
    )

    meta = {
        "seed": args.seed,
        "corpus_target": corpus_target,
        "queries_target": queries_target,
        "split_used": split_used,
        "input_counts": {
            "corpus": total_corpus,
            "queries": total_queries,
            "qrels": total_qrels,
        },
        "output_counts": {
            "corpus": corpus_written,
            "queries": queries_written,
            "qrels": qrels_written,
        },
        "qrels_filtered": {
            "after_query_filter": len(qrels_after_query),
            "after_corpus_filter": qrels_written,
        },
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("Subsample complete")
    print(
        "Qrels queries: "
        f"{len(qrels_query_ids)}, sampled queries: {len(sampled_queries)}, "
        f"qrels kept after query filter: {len(qrels_after_query)}"
    )
    print(f"Input: corpus={total_corpus}, queries={total_queries}, qrels={total_qrels}")
    print(f"Output: corpus={corpus_written}, queries={queries_written}, qrels={qrels_written}")
    print(f"Split used: {split_used}")
    print(f"Output dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
