#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def build_mini_dataset(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    docs: List[Tuple[str, str]] = [
        ("d1", "Apple fruit is sweet and crisp."),
        ("d2", "Bananas are yellow and rich in potassium."),
        ("d3", "Oranges contain vitamin C and have a citrus flavor."),
        ("d4", "Grapes are used to make wine."),
        ("d5", "Strawberries grow in the spring."),
        ("d6", "Blueberries are often used in muffins."),
        ("d7", "Cats are small domestic animals."),
        ("d8", "Dogs are loyal pets and enjoy walks."),
        ("d9", "Parrots can mimic human speech."),
        ("d10", "Goldfish are popular in home aquariums."),
        ("d11", "The sun is a star at the center of the solar system."),
        ("d12", "Earth orbits the sun once a year."),
        ("d13", "Mars is known as the red planet."),
        ("d14", "Jupiter is the largest planet."),
        ("d15", "Saturn has prominent rings."),
        ("d16", "Mercury is the closest planet to the sun."),
        ("d17", "Venus has a thick atmosphere."),
        ("d18", "Neptune is far from the sun."),
        ("d19", "Pluto is classified as a dwarf planet."),
        ("d20", "Black holes have extremely strong gravity."),
        ("d21", "Python is a popular programming language."),
        ("d22", "Java is used for enterprise applications."),
        ("d23", "Rust focuses on memory safety."),
        ("d24", "JavaScript runs in the browser."),
        ("d25", "Go is designed for concurrency."),
        ("d26", "Machine learning models learn from data."),
        ("d27", "Neural networks are used for deep learning."),
        ("d28", "Support vector machines are classic classifiers."),
        ("d29", "Decision trees split data by features."),
        ("d30", "Reinforcement learning uses rewards."),
    ]

    queries: List[Tuple[str, str, List[str]]] = [
        ("q1", "Which fruit is sweet and crisp", ["d1"]),
        ("q2", "yellow fruit with potassium", ["d2"]),
        ("q3", "citrus fruit with vitamin C", ["d3"]),
        ("q4", "pet that enjoys walks", ["d8"]),
        ("q5", "domestic animal cat", ["d7"]),
        ("q6", "largest planet", ["d14"]),
        ("q7", "planet with rings", ["d15"]),
        ("q8", "red planet", ["d13"]),
        ("q9", "programming language for browsers", ["d24"]),
        ("q10", "language focused on memory safety", ["d23"]),
        ("q11", "what orbits the sun yearly", ["d12"]),
        ("q12", "dwarf planet name", ["d19"]),
        ("q13", "what is a star at center of solar system", ["d11"]),
        ("q14", "deep learning uses", ["d27"]),
        ("q15", "models learn from data", ["d26"]),
        ("q16", "classic classifier support vector machine", ["d28"]),
        ("q17", "trees split data by features", ["d29"]),
        ("q18", "learning with rewards", ["d30"]),
        ("q19", "browser scripting language", ["d24", "d21"]),
        ("q20", "pet bird that can mimic speech", ["d9"]),
    ]

    with (output_dir / "corpus.jsonl").open("w", encoding="utf-8") as handle:
        for doc_id, text in docs:
            handle.write(json.dumps({"doc_id": doc_id, "text": text}, ensure_ascii=True) + "\n")

    with (output_dir / "queries.jsonl").open("w", encoding="utf-8") as handle:
        for qid, query, _ in queries:
            handle.write(json.dumps({"qid": qid, "query": query}, ensure_ascii=True) + "\n")

    with (output_dir / "qrels.tsv").open("w", encoding="utf-8") as handle:
        handle.write("qid\tdoc_id\trelevance\n")
        for qid, _, rel_docs in queries:
            for doc_id in rel_docs:
                handle.write(f"{qid}\t{doc_id}\t1\n")

    print(f"Prepared mini dataset at {output_dir}")
    print(f"docs={len(docs)}, queries={len(queries)}, qrels={sum(len(q[2]) for q in queries)}")


def prepare_dataset(dataset: str, data_dir: Path) -> Path:
    if dataset != "mini":
        raise ValueError(f"Unsupported dataset {dataset}; only 'mini' is supported today.")
    output_dir = data_dir / "mini"
    if (output_dir / "corpus.jsonl").exists() and (output_dir / "queries.jsonl").exists():
        print(f"Dataset already prepared at {output_dir}")
        return output_dir
    build_mini_dataset(output_dir)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare dataset for the benchmark.")
    parser.add_argument("--dataset", default="mini")
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    prepare_dataset(args.dataset, data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
