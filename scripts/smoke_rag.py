#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for /rag endpoint.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/rag")
    parser.add_argument("--dataset", default="mini")
    parser.add_argument("--method", default="dense")
    parser.add_argument("--query", default="cats domestic animals")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_context_chars", type=int, default=2000)
    args = parser.parse_args()

    payload = {
        "dataset": args.dataset,
        "method": args.method,
        "query": args.query,
        "top_k": args.top_k,
        "max_context_chars": args.max_context_chars,
    }

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        args.url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        body = response.read().decode("utf-8")
    print(json.dumps(json.loads(body), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
