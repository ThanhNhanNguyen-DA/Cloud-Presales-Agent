"""
Offline batch evaluation on a labelled ground-truth set.

Runs the FULL agent graph on each labelled question, captures the answer and the raw
retrieved context, then scores the standard six RAGAS-style metrics — organised into the
three groups — using the project's own building blocks (Groq judge + local Qwen
embeddings, no external `ragas` package).

Metric groups
-------------
1. Retrieval (evaluates the retriever / vector DB):
     * context_precision  — signal-to-noise of the retrieved context.
     * context_recall     — did retrieval fetch everything the correct answer needs?
2. Generation (evaluates the LLM response):
     * faithfulness       — is the answer grounded in the context (hallucination check)?
     * answer_relevancy   — how directly does the answer address the question?
3. End-to-end (compares against the golden ground truth):
     * answer_correctness — factual accuracy of the answer vs the ground truth.
     * answer_similarity  — semantic resemblance (embedding) to the ground truth.

Usage
-----
    python -m scripts.run_ragas_eval                       # uses data/eval/ground_truth.json
    python -m scripts.run_ragas_eval path/to/labelled.json # custom set

Input JSON: a list of {"question": ..., "ground_truth": ...}.
Output: prints a grouped scorecard and writes data/eval/latest_ragas.json for review.

Requires live Groq + Supabase (this is an intentional offline job, not part of a request).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]

from app.graph.builder import build_graph
from app.evaluation.metrics import (
    answer_correctness,
    answer_relevancy,
    answer_similarity,
    context_precision,
    context_recall,
    faithfulness,
)


DEFAULT_SET = ROOT / "data" / "eval" / "ground_truth.json"
OUTPUT_PATH = ROOT / "data" / "eval" / "latest_ragas.json"

# The six metrics, grouped exactly as the evaluation criteria define them.
GROUPS: dict[str, list[str]] = {
    "Retrieval": ["context_precision", "context_recall"],
    "Generation": ["faithfulness", "answer_relevancy"],
    "End-to-End": ["answer_correctness", "answer_similarity"],
}
METRIC_KEYS = [m for keys in GROUPS.values() for m in keys]

# Gentle pacing between samples to stay under the Groq free-tier RPM (each sample fires
# several LLM calls). The LLM wrapper also waits out rate limits, but pacing avoids churn.
THROTTLE_SECONDS = float(os.getenv("RAGAS_EVAL_THROTTLE", "3"))


def _run_graph(graph, question: str) -> dict:
    """Invoke the graph on one question and return the final answer + retrieved context + route."""
    state = {
        "messages": [],
        "session_id": None,
        "user_query": question,
        "route": None,
        "retrieved_experience": None,
        "raw_tool_output": None,
        "agent_response": None,
        "final_response": None,
        "grading_passed": None,
        "retry_count": 0,
    }
    result = graph.invoke(state)
    return {
        "answer": result.get("final_response") or "",
        "context": result.get("raw_tool_output") or "",
        "route": result.get("route"),
    }


def _score(question: str, answer: str, context: str, ground_truth: str) -> dict[str, float | None]:
    """Compute all six metrics for one sample."""
    return {
        # Retrieval
        "context_precision": context_precision(question, context),
        "context_recall": context_recall(ground_truth, context),
        # Generation
        "faithfulness": faithfulness(answer, context),
        "answer_relevancy": answer_relevancy(question, answer),
        # End-to-end
        "answer_correctness": answer_correctness(answer, ground_truth),
        "answer_similarity": answer_similarity(answer, ground_truth),
    }


def _mean_or_none(values: list[float | None]) -> float | None:
    present = [v for v in values if isinstance(v, (int, float))]
    return round(mean(present), 3) if present else None


def evaluate_set(samples: list[dict]) -> dict:
    """Run the graph + score every labelled sample, returning grouped aggregate + per-sample results."""
    graph = build_graph()
    per_sample = []

    for i, sample in enumerate(samples, 1):
        question = sample["question"]
        ground_truth = sample.get("ground_truth", "")
        print(f"[{i}/{len(samples)}] {question[:70]}…")

        try:
            out = _run_graph(graph, question)
            scores = _score(question, out["answer"], out["context"], ground_truth)
            per_sample.append({"question": question, "route": out["route"],
                               "answer": out["answer"], **scores})
        except Exception as e:  # noqa: BLE001 - one bad sample must not lose the whole run
            print(f"  ! sample failed, recording as empty: {e}")
            per_sample.append({"question": question, "route": None, "answer": "",
                               **{k: None for k in METRIC_KEYS}, "error": str(e)})

        if i < len(samples) and THROTTLE_SECONDS > 0:
            time.sleep(THROTTLE_SECONDS)

    aggregate = {k: _mean_or_none([r[k] for r in per_sample]) for k in METRIC_KEYS}
    groups = {g: {k: aggregate[k] for k in keys} for g, keys in GROUPS.items()}

    return {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "num_samples": len(samples),
        "groups": groups,
        "aggregate": aggregate,
        "per_sample": per_sample,
    }


def _print_scorecard(report: dict) -> None:
    print("\n" + "=" * 64)
    print(f"BATCH EVALUATION — {report['num_samples']} samples @ {report['run_at']}")
    print("=" * 64)
    for group, metrics in report["groups"].items():
        print(f"\n{group}")
        for metric, value in metrics.items():
            bar = "█" * int((value or 0) * 20)
            shown = f"{value:.2f}" if isinstance(value, (int, float)) else "  — "
            print(f"  {metric:<20} {shown}  {bar}")
    print("=" * 64)


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SET
    if not input_path.exists():
        print(f"Ground-truth set not found: {input_path}")
        sys.exit(1)

    samples = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list) or not samples:
        print("Ground-truth set must be a non-empty JSON list of {question, ground_truth}.")
        sys.exit(1)

    report = evaluate_set(samples)
    _print_scorecard(report)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
