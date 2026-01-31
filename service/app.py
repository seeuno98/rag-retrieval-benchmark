from __future__ import annotations

from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from service.engine import RetrievalEngine

app = FastAPI(title="RAG Retrieval Benchmark Service", version="0.1.0")
engine = RetrievalEngine()


class QueryRequest(BaseModel):
    dataset: str = Field(default="mini")
    method: Literal["bm25", "dense", "hybrid"] = Field(default="hybrid")
    query: str
    top_k: int = Field(default=10, ge=1, le=100)


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

    if dataset != "mini":
        raise HTTPException(status_code=400, detail=f"Unsupported dataset: {dataset}")

    try:
        response = engine.query(dataset=dataset, method=method, query=query_text, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return response
