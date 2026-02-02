from __future__ import annotations

import time
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from service.engine import RetrievalEngine
from service.rag import build_context, generate_answer

load_dotenv(override=False)

app = FastAPI(title="RAG Retrieval Benchmark Service", version="0.1.0")
engine = RetrievalEngine()


class QueryRequest(BaseModel):
    dataset: Literal["mini", "msmarco_subset"] = Field(default="mini")
    method: Literal["bm25", "dense", "hybrid"] = Field(default="hybrid")
    query: str
    top_k: int = Field(default=10, ge=1, le=100)


class RagRequest(BaseModel):
    dataset: Literal["mini", "msmarco_subset"] = Field(default="mini")
    method: Literal["bm25", "dense", "hybrid"] = Field(default="hybrid")
    query: str
    top_k: int = Field(default=5, ge=1, le=100)
    max_context_chars: Optional[int] = Field(default=2000, ge=200, le=20000)


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/query")
def query(request: QueryRequest):
    dataset = request.dataset
    method = request.method
    query_text = request.query
    top_k = request.top_k

    if not query_text.strip():
        raise HTTPException(status_code=400, detail="query must be a non-empty string")

    if dataset not in {"mini", "msmarco_subset"}:
        raise HTTPException(status_code=400, detail=f"Unsupported dataset: {dataset}")

    try:
        response = engine.query(dataset=dataset, method=method, query=query_text, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return response


@app.post("/rag")
def rag(request: RagRequest):
    dataset = request.dataset
    method = request.method
    query_text = request.query
    top_k = request.top_k
    max_context_chars = request.max_context_chars or 2000

    if not query_text.strip():
        raise HTTPException(status_code=400, detail="query must be a non-empty string")

    if dataset not in {"mini", "msmarco_subset"}:
        raise HTTPException(status_code=400, detail=f"Unsupported dataset: {dataset}")

    start_total = time.perf_counter()
    try:
        retrieval = engine.query(dataset=dataset, method=method, query=query_text, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    hits = retrieval.get("hits", [])
    context = build_context(hits, max_context_chars)
    answer, llm_id, generation_ms = generate_answer(query_text, context)
    total_ms = (time.perf_counter() - start_total) * 1000.0

    return {
        "dataset": dataset,
        "method": method,
        "query": query_text,
        "top_k": top_k,
        "answer": answer,
        "sources": hits,
        "latency_ms": {
            "retrieval_ms": retrieval["latency_ms"]["retrieval_ms"],
            "generation_ms": generation_ms,
            "total_ms": total_ms,
        },
        "meta": {
            "llm": llm_id,
            "cache": retrieval["meta"]["cache"],
        },
    }
