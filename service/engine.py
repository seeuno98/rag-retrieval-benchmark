from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from retrieval.bm25 import BM25Index
from retrieval.dense import DenseIndex
from retrieval.hybrid import rrf_merge
from scripts.prepare_data import prepare_dataset

ROOT_DIR = Path(__file__).resolve().parents[1]


def _load_corpus(path: Path) -> List[Tuple[str, str]]:
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


@dataclass
class DatasetCache:
    dataset: str
    corpus: List[Tuple[str, str]]
    doc_lookup: Dict[str, str]
    bm25_index: Optional[BM25Index] = None
    dense_index: Optional[DenseIndex] = None
    dense_model: Optional[str] = None
    dense_loaded_from_disk: bool = False
    dense_initialized: bool = False


class RetrievalEngine:
    def __init__(
        self,
        data_dir: Path = ROOT_DIR / "data",
        artifacts_dir: Path = ROOT_DIR / "artifacts",
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self._data_dir = data_dir
        self._artifacts_dir = artifacts_dir
        self._dense_model = dense_model
        self._cache: Dict[str, DatasetCache] = {}

    def _load_dataset(self, dataset: str) -> DatasetCache:
        if dataset in self._cache:
            return self._cache[dataset]

        dataset_dir = prepare_dataset(dataset, self._data_dir)
        corpus = _load_corpus(dataset_dir / "corpus.jsonl")
        doc_lookup = {doc_id: text for doc_id, text in corpus}
        cache = DatasetCache(dataset=dataset, corpus=corpus, doc_lookup=doc_lookup)
        self._cache[dataset] = cache
        return cache

    def _ensure_bm25(self, cache: DatasetCache) -> None:
        if cache.bm25_index is not None:
            return
        print(f"[engine] Building BM25 index for dataset={cache.dataset}")
        cache.bm25_index = BM25Index.build(cache.corpus)

    def _dense_artifact_paths(self, dataset: str) -> Tuple[Path, Path]:
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        index_path = self._artifacts_dir / f"dense_{dataset}.faiss"
        doc_ids_path = self._artifacts_dir / f"dense_{dataset}_doc_ids.json"
        return index_path, doc_ids_path

    def _ensure_dense(self, cache: DatasetCache) -> None:
        if cache.dense_index is not None:
            return

        index_path, doc_ids_path = self._dense_artifact_paths(cache.dataset)
        if index_path.exists() and doc_ids_path.exists():
            print(f"[engine] Loading dense index from disk for dataset={cache.dataset}")
            with doc_ids_path.open("r", encoding="utf-8") as handle:
                doc_ids = json.load(handle)
            cache.dense_index = DenseIndex.load(
                str(index_path),
                doc_ids=doc_ids,
                model_name=self._dense_model,
                normalize=True,
            )
            cache.dense_model = self._dense_model
            cache.dense_loaded_from_disk = True
            cache.dense_initialized = True
            return

        print(f"[engine] Building dense index for dataset={cache.dataset}")
        cache.dense_index = DenseIndex.build(cache.corpus, model_name=self._dense_model)
        cache.dense_model = self._dense_model
        cache.dense_loaded_from_disk = False
        cache.dense_initialized = True
        cache.dense_index.save(str(index_path), str(doc_ids_path))
        print(f"[engine] Saved dense index to {index_path}")

    def query(self, dataset: str, method: str, query: str, top_k: int) -> Dict[str, object]:
        cache = self._load_dataset(dataset)
        start_total = time.perf_counter()
        start_retrieval = time.perf_counter()

        if method == "bm25":
            self._ensure_bm25(cache)
            assert cache.bm25_index is not None
            scored = cache.bm25_index.retrieve(query, top_k)
        elif method == "dense":
            self._ensure_dense(cache)
            assert cache.dense_index is not None
            scored = cache.dense_index.retrieve(query, top_k)
        elif method == "hybrid":
            self._ensure_bm25(cache)
            self._ensure_dense(cache)
            assert cache.bm25_index is not None
            assert cache.dense_index is not None
            bm25_scored = cache.bm25_index.retrieve(query, top_k * 5)
            dense_scored = cache.dense_index.retrieve(query, top_k * 5)
            scored = rrf_merge(bm25_scored, dense_scored, top_k=top_k)
        else:
            raise ValueError(f"Unknown method: {method}")

        retrieval_ms = (time.perf_counter() - start_retrieval) * 1000.0
        total_ms = (time.perf_counter() - start_total) * 1000.0

        hits: List[Dict[str, object]] = []
        for rank, (doc_id, score) in enumerate(scored, start=1):
            text = cache.doc_lookup.get(doc_id, "")
            snippet = text[:300]
            hits.append(
                {
                    "rank": rank,
                    "doc_id": doc_id,
                    "score": float(score),
                    "text_snippet": snippet,
                }
            )

        meta = {
            "cache": {
                "in_memory": cache.dense_initialized if method in {"dense", "hybrid"} else False,
                "loaded_from_disk": cache.dense_loaded_from_disk if method in {"dense", "hybrid"} else False,
            },
            "model": cache.dense_model if method in {"dense", "hybrid"} else None,
        }

        return {
            "dataset": dataset,
            "method": method,
            "top_k": top_k,
            "hits": hits,
            "latency_ms": {"retrieval_ms": retrieval_ms, "total_ms": total_ms},
            "meta": meta,
        }
