"""BM25 retrieval implementation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    doc_ids: List[str]
    doc_len: List[int]
    avgdl: float
    idf: Dict[str, float]
    tf: List[Dict[str, int]]
    k1: float = 1.2
    b: float = 0.75

    @classmethod
    def build(cls, docs: Sequence[Tuple[str, str]], k1: float = 1.2, b: float = 0.75) -> "BM25Index":
        doc_ids: List[str] = []
        doc_len: List[int] = []
        tf: List[Dict[str, int]] = []
        df: Dict[str, int] = {}

        for doc_id, text in docs:
            tokens = tokenize(text)
            doc_ids.append(doc_id)
            doc_len.append(len(tokens))
            counts: Dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            tf.append(counts)
            for token in counts:
                df[token] = df.get(token, 0) + 1

        total_docs = len(doc_ids)
        avgdl = sum(doc_len) / total_docs if total_docs else 0.0
        idf: Dict[str, float] = {}
        for token, freq in df.items():
            idf[token] = math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))

        return cls(
            doc_ids=doc_ids,
            doc_len=doc_len,
            avgdl=avgdl,
            idf=idf,
            tf=tf,
            k1=k1,
            b=b,
        )

    def score(self, query_tokens: Iterable[str]) -> List[Tuple[str, float]]:
        scores = [0.0 for _ in self.doc_ids]
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf = self.idf[token]
            for idx, doc_tf in enumerate(self.tf):
                freq = doc_tf.get(token, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * (self.doc_len[idx] / (self.avgdl or 1.0)))
                scores[idx] += idf * (freq * (self.k1 + 1)) / denom
        return list(zip(self.doc_ids, scores))

    def retrieve(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        tokens = tokenize(query)
        scored = self.score(tokens)
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
