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
