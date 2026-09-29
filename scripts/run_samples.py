#!/usr/bin/env python3
"""
Run 6 sample queries against the live API and save the results to
docs/sample_outputs.json.  The server must be running before this script.

Usage:
    uvicorn src.rag.api:app &
    python scripts/run_samples.py
"""

import json
import sys
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8000"

QUERIES = [
    "What is agentic AI and how does it differ from traditional AI?",
    "What are the core components of an agentic AI system?",
    "How do multi-agent systems coordinate tasks?",
    "What role does memory play in agentic AI pipelines?",
    "What are the main risks or limitations of agentic AI?",
    # Out-of-scope question to demonstrate grounded refusal:
    "Who won the FIFA World Cup in 2022?",
]


def run():
    results = []
    for q in QUERIES:
        resp = httpx.post(f"{BASE_URL}/chat", json={"question": q}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        results.append(
            {
                "question": q,
                "answer": data["answer"],
                "confidence": data["confidence"],
                "top_source_page": data["sources"][0]["page"] if data["sources"] else None,
            }
        )
        print(f"Q: {q}")
        print(f"A: {data['answer'][:120]}...")
        print(f"   confidence={data['confidence']:.3f}\n")

    out = Path("docs/sample_outputs.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    run()
