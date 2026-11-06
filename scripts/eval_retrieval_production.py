"""Measure the deployed retrieval path and collect passage scores for human review.

    python -m scripts.eval_retrieval_production --out /tmp/hybrid.json
    python -m scripts.eval_retrieval_production --no-hybrid --out /tmp/vector.json

No LLM generation, no DB writes. Requires the embedding and reranker models.
The old single-gold evaluation set measures ranking, not selective precision: all
other candidate relevance labels remain null until a reviewer judges them.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from app.config import EMBEDDING_DIM, EMBEDDING_MODEL
from app.tools.rag_tools import CANDIDATES, COMPETITOR_CATEGORY, TOP_K, VECTOR_THRESHOLD, retrieve_documents
from app.tools.reranker import RERANK_MIN_SCORE, RERANK_MODEL


def evaluate_queries(samples: list[dict], *, hybrid: bool = True, progress: bool = False) -> dict:
    queries, ranks, elapsed = [], [], []
    for index, sample in enumerate(samples):
        scope = sample.get("scope", "own")
        if scope not in {"own", "competitor"}:
            raise ValueError("scope must be own or competitor.")
        started = time.perf_counter()
        # Score the SAME candidate pool as production, retaining below-cutoff passages
        # for review. The report's ranking metrics apply the runtime cutoff/top-k.
        result = retrieve_documents(
            sample["question"], category=COMPETITOR_CATEGORY if scope == "competitor" else None,
            exclude=[COMPETITOR_CATEGORY] if scope == "own" else None,
            provider=sample.get("provider"), hybrid=hybrid,
            top_k=CANDIDATES * 2, min_score=0.0,
        )
        elapsed.append(time.perf_counter() - started)
        if result.rerank_status != "scored":
            raise RuntimeError("Scored evaluation requires an available, enabled reranker.")
        gold = sample.get("gold_ids") or ([sample["gold_id"]] if sample.get("gold_id") else [])
        def is_gold(document):
            return bool(set([document.get("id"), *document.get("duplicate_ids", [])]) & set(gold))
        accepted = [d for d in result.documents if d["rerank_score"] >= RERANK_MIN_SCORE][:TOP_K]
        ranks.append(next((i for i, d in enumerate(accepted, 1) if is_gold(d)), None))
        queries.append({
            "id": sample.get("id") or f"query-{index + 1}", "question": sample["question"],
            "scope": scope, "provider": sample.get("provider"), "gold_ids": gold,
            "split": sample.get("split"), "answerable": sample.get("answerable", True if gold else None),
            "method": result.method, "notes": result.notes, "seconds": elapsed[-1],
            "candidates": [{"id": d.get("id"), "score": d["rerank_score"],
                            "duplicate_ids": d.get("duplicate_ids", []),
                            "relevant": True if is_gold(d) else None,
                            "service_name": d.get("service_name"), "content": d["content"],
                            "source": (d.get("metadata") or {}).get("source")}
                           for d in result.documents],
        })
        if progress and (index == 0 or (index + 1) % 10 == 0 or index + 1 == len(samples)):
            print(f"Scored {index + 1}/{len(samples)} queries", flush=True)
    labelled = [r for r, q in zip(ranks, queries, strict=True) if q["gold_ids"]]
    n = len(labelled)
    return {
        "run_at": datetime.now(UTC).isoformat(), "top_k": TOP_K,
        "pipeline": {"embedding_model": EMBEDDING_MODEL, "embedding_dim": EMBEDDING_DIM,
                     "rerank_model": RERANK_MODEL, "activation": "sigmoid",
                     "candidates_per_channel": CANDIDATES, "vector_threshold": VECTOR_THRESHOLD,
                     "hybrid_requested": hybrid, "runtime_cutoff": RERANK_MIN_SCORE},
        "metrics": {"queries": len(queries), "queries_with_gold": n,
                    "hit_at_1": sum(r == 1 for r in labelled) / n if n else None,
                    "hit_at_k": sum(r is not None for r in labelled) / n if n else None,
                    "mrr_at_k": sum(1 / r for r in labelled if r) / n if n else None,
                    "mean_seconds": sum(elapsed) / len(elapsed) if elapsed else None,
                    "degraded_queries": sum(bool(q["notes"]) for q in queries)},
        "queries": queries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=Path("evals/retrieval/ranking.json"))
    parser.add_argument("--hybrid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.eval.read_text(encoding="utf-8"))
    report = evaluate_queries(data.get("queries", []) if isinstance(data, dict) else data,
                              hybrid=args.hybrid, progress=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print("Review every candidate label and assign disjoint calibration/validation splits before calibration.")


if __name__ == "__main__":
    main()
