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

The same retrieval implementations used in the offline benchmark power the live API endpoints.

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

MSMARCO subset example:

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"dataset":"msmarco_subset","method":"dense","query":"what causes diabetes", "top_k":5}'
```

First request may be slower due to model/index initialization. Subsequent requests reuse in-memory caches and are significantly faster.

### RAG Generation (LangChain)

#### Architecture
LangChain is used only as an orchestration layer for prompt construction and LLM calls; all retrieval logic is implemented and benchmarked independently in this repo.


```text 
Client
  |
  |  POST /rag  (dataset, method=bm25|dense|hybrid, query, top_k)
  v
FastAPI (service.app)
  |
  v
RetrievalEngine (service.engine)
  |
  +--> BM25Index (sparse)         \
  +--> DenseIndex + FAISS (dense)  ---> Top-K passages (doc_id, score, snippet)
  +--> Hybrid (RRF merge)         /
  |
  v
LangChain orchestration
  |
  +--> Prompt (Question + Retrieved Context)
  |
  +--> LLM backend
        - default: deterministic mock LLM
        - optional: OpenAI Chat model (via OPENAI_API_KEY)
  |
  v
Answer + sources + latency breakdown
```



#### Example (mini):

```bash
curl -X POST http://localhost:8000/rag -H "Content-Type: application/json" \
  -d '{"dataset":"mini","method":"dense","query":"cats domestic animals","top_k":3}'
```

#### Example response (MSMARCO subset, LangChain + OpenAI)

```bash
curl -X POST http://localhost:8000/rag -H "Content-Type: application/json" \
  -d '{"dataset":"msmarco_subset","method":"dense","query":"what causes diabetes","top_k":5}'
```

```json
{
  "answer": "Type 1 Diabetes is caused by an autoimmune destruction of insulin-producing beta cells...",
  "sources": [
    {"rank": 1, "doc_id": "4645402", "score": 0.76, "...": "..."},
    {"rank": 2, "doc_id": "5268000", "score": 0.73, "...": "..."}
  ],
  "latency_ms": {
    "retrieval_ms": 12.4,
    "generation_ms": 1500.6,
    "total_ms": 1515.2
  },
  "meta": {
    "llm": "openai:gpt-4o-mini"
  }
}
```

Insights:

* **Grounded generation**: The answer is synthesized directly from the top retrieved passages (not hallucinated), and supporting sources are returned for transparency.

* **Separation of concerns**:
  * Retrieval (BM25/Dense/Hybrid) handles relevance and speed (~10–15 ms).
  * LangChain + LLM handles natural-language synthesis (~0.5–2 s).

* **Latency profile**: Retrieval is negligible compared to LLM generation; optimizing retrieval improves scalability while LLM latency dominates user experience.

* **Production pattern**: Retrieval is evaluated offline with Recall/MRR, while generation is evaluated by groundedness, citations, and latency rather than traditional IR metrics.

Notes:

* Generation latency and cost scale with token usage; retrieval remains constant-time after indexing.

* By default, /rag uses a deterministic mock LLM (no API keys required). If `OPENAI_API_KEY` is set, it will use an OpenAI chat model via LangChain instead.

To enable OpenAI-backed generation locally, create a `.env` file in the repo root:

```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

The `.env` file is gitignored. Start the server normally; `/rag` will use OpenAI automatically when the key is present.

Note: the first request may be slower due to model/index initialization; subsequent requests are faster because artifacts remain cached.

**Generation metrics**:
Retrieval quality is evaluated offline using Information Retrieval (IR) metrics such as Recall@K and MRR.  
Generation quality, however, is assessed by groundedness to retrieved context, returned citations, latency (retrieval vs. generation), and token cost rather than IR ranking metrics.


| metric | typical latency |
|---|---|
| retrieval (warm, in-memory index) | 5–15 ms |
| generation (OpenAI LLM) | 500–2000 ms |

## Developer shortcuts (Makefile)

Common tasks are wrapped in a simple Makefile for reproducibility:

```bash
make bench-mini        # quick benchmark on mini dataset
make bench-msmarco     # realistic benchmark on MSMARCO subset
make serve             # run FastAPI locally
make smoke-local       # health + query + rag smoke tests
make docker-build
make docker-run
```

These commands simply wrap the Python and curl commands shown above.

---
