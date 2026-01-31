# rag-retrieval-benchmark

Evaluation-first benchmark for RAG retrieval, comparing BM25, dense, and hybrid retrieval methods using standard IR datasets. This repository focuses on reproducible experiments, clear retrieval metrics, and a production-style structure before any generation components are added.

---

## Datasets

MSMARCO (BEIR passage) provides natural-language queries over a large-scale passage corpus with relevance judgments (qrels). It is a standard retrieval benchmark for metrics such as MRR and nDCG.

Reproducible MSMARCO subset commands (data/ is generated locally and not committed):

```bash
python3 scripts/subsample_msmarco.py \
  --input_dir data/beir_raw/msmarco/msmarco \
  --output_dir data/msmarco_subset \
  --corpus_target 100000 \
  --queries_target 2000 \
  --seed 42 \
  --split train
```

Sanity check the subset:

```bash
python3 scripts/sanity_check_subset.py --subset_dir data/msmarco_subset
```

---

## Quickstart

Create a virtual environment and install requirements:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run the benchmark (mini dataset, bm25 + dense + hybrid):

```bash
python3 scripts/run_benchmark.py --dataset mini --methods bm25,dense,hybrid --top_k 10
```

Results are written to `results/results_latest.json`, `results/results_latest.csv`, and `results/results_latest.md`.

## Results (sample)

Command used:

```bash
python3 scripts/run_benchmark.py --dataset mini --methods bm25,dense,hybrid --top_k 10
```

| method | recall@5 | recall@10 | mrr@10 | p50_ms | p90_ms |
| --- | --- | --- | --- | --- | --- |
| bm25 | 0.9500 | 0.9500 | 0.9250 | 0.01 | 0.01 |
| dense | 1.0000 | 1.0000 | 1.0000 | 5.34 | 6.07 |
| hybrid | 0.9500 | 1.0000 | 0.9306 | 5.26 | 5.73 |

Metrics are averaged over queries: Recall@K and MRR@10 reflect retrieval quality, while p50/p90 are end-to-end latencies (ms). Retrieval, rerank (if enabled later), and total latencies are tracked in the JSON artifact.
