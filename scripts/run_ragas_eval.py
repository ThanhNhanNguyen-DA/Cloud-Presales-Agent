"""
Offline batch RAGAS evaluation on a labelled ground-truth set.

Runs the FULL agent graph on each labelled question, captures the answer and the raw
retrieved context, then scores four RAGAS-style metrics using the project's own judge
(Groq) and local embeddings (Qwen) — the same building blocks as the live per-answer
scoring, so live and batch numbers are comparable. No external `ragas` package needed.

Metrics
-------
  * faithfulness       — LLM-judged fraction of answer claims supported by the context
                         (reference-free; same as the live metric).
  * answer_relevancy   — cosine(question, answer). Reference-free.
  * answer_correctness — cosine(answer, ground_truth). Ground-truth based.
  * context_recall     — cosine(ground_truth, retrieved_context). A proxy for "did we
                         retrieve the information the correct answer needs?".

Usage
-----
    python -m scripts.run_ragas_eval                       # uses data/eval/ground_truth.json
    python -m scripts.run_ragas_eval path/to/labelled.json # custom set

Input JSON: a list of {"question": ..., "ground_truth": ...}.
Output: prints a scorecard and writes data/eval/latest_ragas.json (read by the
Streamlit monitoring dashboard).

Requires live Groq + Supabase (this is an intentional offline job, not part of a request).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]

from app.graph.builder import build_graph
from app.tools.embedding_tool import embed_text
from app.evaluation.ragas_live import _cosine, answer_relevancy, faithfulness


DEFAULT_SET = ROOT / "data" / "eval" / "ground_truth.json"
OUTPUT_PATH = ROOT / "data" / "eval" / "latest_ragas.json"


def _run_graph(graph, question: str) -> dict:
    """Invoke the graph on one question and return the final answer + retrieved context."""
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


def _cos_texts(a: str, b: str) -> Optional[float]:
    """Cosine similarity between two texts' embeddings, clamped to [0, 1]; None if empty."""
    if not a.strip() or not b.strip():
        return None
    score = _cosine(embed_text(a), embed_text(b))
    return round(max(0.0, min(1.0, score)), 3)


def _mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    present = [v for v in values if isinstance(v, (int, float))]
    return round(mean(present), 3) if present else None


def evaluate_set(samples: List[dict]) -> dict:
    """Run the graph + score every labelled sample, returning aggregate + per-sample results."""
    graph = build_graph()
    per_sample = []

    for i, sample in enumerate(samples, 1):
        question = sample["question"]
        ground_truth = sample.get("ground_truth", "")
        print(f"[{i}/{len(samples)}] {question[:70]}…")

        out = _run_graph(graph, question)
        row = {
            "question": question,
            "route": out["route"],
            "answer": out["answer"],
            "faithfulness": faithfulness(out["answer"], out["context"]),
            "answer_relevancy": answer_relevancy(question, out["answer"]),
            "answer_correctness": _cos_texts(out["answer"], ground_truth),
            "context_recall": _cos_texts(ground_truth, out["context"]),
        }
        per_sample.append(row)

    metric_keys = ["faithfulness", "answer_relevancy", "answer_correctness", "context_recall"]
    aggregate = {k: _mean_or_none([r[k] for r in per_sample]) for k in metric_keys}

    return {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "num_samples": len(samples),
        "aggregate": aggregate,
        "per_sample": per_sample,
    }


def _print_scorecard(report: dict) -> None:
    print("\n" + "=" * 60)
    print(f"BATCH RAGAS — {report['num_samples']} samples @ {report['run_at']}")
    print("=" * 60)
    for metric, value in report["aggregate"].items():
        bar = "█" * int((value or 0) * 20)
        shown = f"{value:.2f}" if isinstance(value, (int, float)) else "  — "
        print(f"  {metric:<20} {shown}  {bar}")
    print("=" * 60)


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
    print(f"\nWrote {OUTPUT_PATH} (shown on the Streamlit 📊 Evaluation dashboard).")


if __name__ == "__main__":
    main()
