SHELL := /bin/bash

PY ?= python3
HOST ?= http://127.0.0.1:8000
DATASET_MINI ?= mini
DATASET_MSMARCO ?= msmarco_subset
METHODS ?= bm25,dense,hybrid
TOPK ?= 10
LIMIT ?= 200

.PHONY: help venv install bench-mini bench-msmarco serve health query-mini query-msmarco rag-mini rag-msmarco \
        docker-build docker-run docker-stop smoke-local smoke-docker

help:
	@echo "Targets:"
	@echo "  make venv/install"
	@echo "  make bench-mini | bench-msmarco"
	@echo "  make serve | health | query-mini | query-msmarco | rag-mini | rag-msmarco"
	@echo "  make docker-build | docker-run | docker-stop"
	@echo "  make smoke-local | smoke-docker"

venv:
	$(PY) -m venv .venv
	@echo "Run: . .venv/bin/activate"

install:
	$(PY) -m pip install -r requirements.txt

bench-mini:
	$(PY) scripts/run_benchmark.py --dataset $(DATASET_MINI) --methods $(METHODS) --top_k $(TOPK)

bench-msmarco:
	$(PY) scripts/run_benchmark.py --dataset $(DATASET_MSMARCO) --methods $(METHODS) --top_k $(TOPK) --limit_queries $(LIMIT)

serve:
	$(PY) -m uvicorn service.app:app --reload --port 8000

health:
	@curl -s $(HOST)/health && echo

query-mini:
	@curl -s -X POST $(HOST)/query \
		-H "Content-Type: application/json" \
		-d '{"dataset":"$(DATASET_MINI)","method":"dense","query":"cats domestic animals","top_k":5}' \
	| $(PY) -c "import sys,json; d=json.load(sys.stdin); print(d['dataset'], d['method'], d['hits'][0]['doc_id'], d['latency_ms'], d['meta'])"

query-msmarco:
	@curl -s -X POST $(HOST)/query \
		-H "Content-Type: application/json" \
		-d '{"dataset":"$(DATASET_MSMARCO)","method":"dense","query":"what causes diabetes","top_k":5}' \
	| $(PY) -c "import sys,json; d=json.load(sys.stdin); print(d['dataset'], d['method'], len(d['hits']), d['latency_ms'], d['meta'])"

rag-mini:
	@curl -s -X POST $(HOST)/rag \
		-H "Content-Type: application/json" \
		-d '{"dataset":"$(DATASET_MINI)","method":"dense","query":"cats domestic animals","top_k":3}' \
	| $(PY) -c "import sys,json; d=json.load(sys.stdin); print(d['meta'], d['latency_ms']);"

rag-msmarco:
	@curl -s -X POST $(HOST)/rag \
		-H "Content-Type: application/json" \
		-d '{"dataset":"$(DATASET_MSMARCO)","method":"dense","query":"what causes diabetes","top_k":5}' \
	| $(PY) -c "import sys,json; d=json.load(sys.stdin); print(d['meta'], d['latency_ms']);"

docker-build:
	docker build -t rag-retrieval-benchmark .

docker-run:
	@docker rm -f rag-rag >/dev/null 2>&1 || true
	docker run --name rag-rag -p 8000:8000 \
		-v "$$(pwd)/data:/app/data" \
		-v "$$(pwd)/artifacts:/app/artifacts" \
		-v "$$(pwd)/results:/app/results" \
		rag-retrieval-benchmark

docker-stop:
	@docker rm -f rag-rag >/dev/null 2>&1 || true

# Smoke tests assume server is already running (local or docker)
smoke-local: health query-mini query-msmarco rag-mini

smoke-docker: health query-mini query-msmarco rag-mini
