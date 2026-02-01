# rag-retrieval-benchmark

Evaluation-first benchmark for RAG retrieval, comparing BM25, dense, and hybrid retrieval methods using standard IR datasets. This repository focuses on reproducible experiments, clear retrieval metrics, and a production-style structure before any generation components are added.

---

## Datasets

### mini
Tiny synthetic dataset for fast local testing (<5s). Useful for smoke tests and CI.

### msmarco_subset
100k passages, 2k queries. A realistic retrieval benchmark stored under:
data/msmarco_subset/

Normalized format:
- corpus.jsonl
- queries.jsonl
- qrels.tsv

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

Run a quick smoke test:
```bash
python3 scripts/run_benchmark.py --dataset mini --methods bm25,dense,hybrid --top_k 10
```

Realistic benchmark (MSMARCO):

```bash
python3 scripts/run_benchmark.py --dataset msmarco_subset --methods bm25,dense,hybrid --top_k 10 --limit_queries 200
```

Note: mini is for quick smoke tests; msmarco_subset gives realistic metrics.

Results are written to `results/results_latest.json`, `results/results_latest.csv`, and `results/results_latest.md`.

## Artifacts

- `results/`: latest metrics outputs (JSON/CSV/Markdown).
- `artifacts/`: optional future artifacts (indexes, cached embeddings, etc.).


## Results (MSMARCO subset, 100k docs, 200 queries)

| method | recall@5 | recall@10 | mrr@10 | p50_ms | p90_ms |
| --- | --- | --- | --- | --- | --- |
| bm25 | 0.7950 | 0.8375 | 0.6879 | 170.39 | 279.12 |
| dense | 0.9300 | 0.9600 | 0.8447 | 12.61 | 13.75 |
| hybrid | 0.8800 | 0.9325 | 0.7976 | 182.68 | 300.48 |

Metrics are averaged over queries. Recall@K and MRR@10 measure retrieval quality; p50/p90 report median and tail latency (ms). Higher Recall/MRR is better; lower latency is better.


Insights:
- Dense retrieval improves Recall@10 by +12% and reduces latency ~10× vs BM25 on CPU.
- Hybrid trades extra latency for combining signals
- First request may be slower due to model warmup and artifact loading.




## Results (mini, toy example)

Used only for smoke testing; metrics are near-perfect due to tiny corpus.

<details>
  <summary>Click to expand to see the detailed results</summary>
  
  Command used:

```bash
python3 scripts/run_benchmark.py --dataset mini --methods bm25,dense,hybrid --top_k 10
```

| method | recall@5 | recall@10 | mrr@10 | p50_ms | p90_ms |
| --- | --- | --- | --- | --- | --- |
| bm25 | 0.9500 | 0.9500 | 0.9250 | 0.01 | 0.01 |
| dense | 1.0000 | 1.0000 | 1.0000 | 5.82 | 6.57 |
| hybrid | 0.9500 | 1.0000 | 0.9306 | 5.78 | 6.05 |



Notes:
- On the tiny mini dataset (30 docs, 20 queries), BM25 latency can be in microseconds; values are reported in milliseconds and may round to ~0.01ms.

- First run can be slightly slower due to model warmup and caching. For meaningful latency comparisons, use the MSMARCO subset.

</details>



## Serving (FastAPI)

Local run:

```bash
python3 -m uvicorn service.app:app --reload --port 8000
```

Docker run:

```bash
docker build -t rag-retrieval-benchmark .
docker run -p 8000:8000 rag-retrieval-benchmark
```

Example curl:

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"dataset":"mini","method":"hybrid","query":"planet with rings", "top_k":10}'
```

First request may be slower due to model/index initialization. Subsequent requests reuse in-memory caches and are significantly faster.
