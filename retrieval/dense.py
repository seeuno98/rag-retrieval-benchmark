"""Dense retrieval with FAISS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class DenseIndex:
    doc_ids: List[str]
    embeddings: np.ndarray
    index: "faiss.Index"
    model_name: str
    normalize: bool
    model: "SentenceTransformer"

    @classmethod
    def build(
        cls,
        docs: Sequence[Tuple[str, str]],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
        batch_size: int = 64,
    ) -> "DenseIndex":
        from sentence_transformers import SentenceTransformer
        import faiss

        doc_ids = [doc_id for doc_id, _ in docs]
        texts = [text for _, text in docs]
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        ).astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim) if normalize else faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return cls(
            doc_ids=doc_ids,
            embeddings=embeddings,
            index=index,
            model_name=model_name,
            normalize=normalize,
            model=model,
        )

    def retrieve(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        query_emb = self.model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        ).astype("float32")
        scores, indices = self.index.search(query_emb, top_k)
        results: List[Tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            results.append((self.doc_ids[idx], float(score)))
        return results
