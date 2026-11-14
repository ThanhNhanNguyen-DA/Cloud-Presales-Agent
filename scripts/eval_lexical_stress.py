"""
Keyword-sensitive retrieval check: abbreviations, product names, unaccented typing and exact
prices — the cases embeddings handle poorly (evals/retrieval/lexical_stress.json).

A query passes when an accepted passage (top 4, after reranking and the cutoff) belongs to an
expected service (`expect`, substring of service_name) or contains `expect_content`.
Runs the production retrieve_documents path against the live database.

    python -m scripts.eval_lexical_stress            # vector-only vs hybrid
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.tools.rag_tools import COMPETITOR_CATEGORY, document_scope, retrieve_documents


def passes(item: dict, documents: list[dict]) -> list[bool]:
    marks = []
    for doc in documents:
        name = (doc.get("service_name") or "").lower()
        ok = any(e.lower() in name for e in item.get("expect", []))
        ok |= bool(item.get("expect_content")) and item["expect_content"] in (doc.get("content") or "")
        marks.append(ok)
    return marks


def run(items: list[dict], hybrid: bool) -> dict:
    hit1 = hit4 = 0
    seconds, misses = [], []
    for item in items:
        scope = ({"category": COMPETITOR_CATEGORY} if item["scope"] == "competitor"
                 else document_scope())
        started = time.perf_counter()
        result = retrieve_documents(item["question"], hybrid=hybrid, **scope)
        seconds.append(time.perf_counter() - started)
        marks = passes(item, result.documents)
        hit1 += bool(marks) and marks[0]
        hit4 += any(marks)
        if not any(marks):
            misses.append(f"{item['id']} {item['question']!r}")
    n = len(items)
    warm = seconds[1:] or seconds
    return {"hit_at_1": hit1 / n, "hit_at_4": hit4 / n, "mean_seconds_warm": sum(warm) / len(warm),
            "misses": misses}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=Path("evals/retrieval/lexical_stress.json"))
    args = parser.parse_args()
    items = json.loads(args.eval.read_text(encoding="utf-8"))
    for label, hybrid in (("vector only", False), ("hybrid", True)):
        r = run(items, hybrid)
        print(f"{label:12s} Hit@1 {r['hit_at_1']:.0%}  Hit@4 {r['hit_at_4']:.0%}  "
              f"{r['mean_seconds_warm']:.2f}s/query")
        for miss in r["misses"]:
            print("    miss:", miss)


if __name__ == "__main__":
    main()
