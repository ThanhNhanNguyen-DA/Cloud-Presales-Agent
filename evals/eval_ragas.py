"""
RAGAS-style scoring of an end-to-end run, plus a cross-check against the Grounding Grader.

Metrics come from app/evaluation/metrics.py: the judge is the app's own fallback chain
(`get_chat_llm(role="judge")`, JUDGE_MODELS or LLM_MODELS) and embeddings are the local
Qwen3-Embedding-0.6B. No OpenAI key, no external `ragas` package. Scoring runs on
MAX_WORKERS = 2 threads to stay under free-tier rate limits.

  document / competitor  faithfulness, answer_relevancy, context_precision, context_recall
  calculator             faithfulness (vs `tool(args) -> result`), answer_relevancy
  general                not scored (no data to ground against)

context_recall's reference = the case's `required_numbers_or_facts`.

Cross-check: the grader said PASS but faithfulness < 0.8, or the grader's final verdict was
FAIL/UNVERIFIED while faithfulness >= 0.8. Either the grader or the judge is wrong: read both.

    python -m evals.eval_ragas                              # latest evals/results/e2e_*.json
    python -m evals.eval_ragas evals/results/e2e_<ts>.json  # a given run (updated in place)
    python -m evals.eval_e2e --live --ragas                 # run + score in one go
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

from evals import common

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
BY_ROUTE = {"document": METRICS, "competitor": METRICS, "calculator": METRICS[:2]}
MAX_WORKERS = 2
FAITHFULNESS_ALERT = 0.8


def _reference(record: dict) -> str:
    facts = [f if isinstance(f, str) else " / ".join(f) for f in record.get("required_facts") or []]
    return "; ".join(facts)


def score_record(record: dict) -> dict:
    from app.evaluation import metrics as m  # loads the judge / embedder lazily, only when scoring

    wanted = BY_ROUTE.get(record.get("route"), ())
    q, a, ctx = record["question"], record["answer"], record.get("raw_tool_output") or ""
    compute = {
        "faithfulness": lambda: m.faithfulness(a, ctx),
        "answer_relevancy": lambda: m.answer_relevancy(q, a),
        "context_precision": lambda: m.context_precision(q, ctx),
        "context_recall": lambda: m.context_recall(_reference(record), ctx),
    }
    return {name: compute[name]() if name in wanted else None for name in METRICS}


def discrepancy(record: dict) -> str | None:
    faith = (record.get("ragas") or {}).get("faithfulness")
    if faith is None or not record.get("verdicts"):
        return None
    grader_pass = record["verdicts"][-1] == "PASS"
    if grader_pass and faith < FAITHFULNESS_ALERT:
        return f"grader PASS, faithfulness {faith:.2f}"
    if not grader_pass and faith >= FAITHFULNESS_ALERT:
        return f"grader {record['verdicts'][-1]}, faithfulness {faith:.2f}"
    return None


def score_records(records: list[dict]) -> dict:
    """Adds `ragas` and `discrepancy` to every record; returns {metric: mean} for the run."""
    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        for record, scores in zip(records, pool.map(score_record, records), strict=True):
            record["ragas"] = scores
            record["discrepancy"] = discrepancy(record)
    return {name: (round(mean(vals), 3) if (vals := [r["ragas"][name] for r in records
                                                    if r["ragas"][name] is not None]) else None)
            for name in METRICS}


def markdown(records: list[dict], means: dict) -> str:
    cross = [{"id": r["id"], "route": r["route"], "verdicts": r["verdicts"], "discrepancy": r["discrepancy"]}
             for r in records if r.get("discrepancy")]
    return ("## RAGAS-style metrics\n\n" + common.md_table([{"metric": k, "mean": v} for k, v in means.items()])
            + "\n## Cross-check: grader vs faithfulness\n\n" + common.md_table(cross))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", nargs="?", help="An evals/results/e2e_*.json file (default: the latest).")
    args = parser.parse_args()
    runs = sorted(common.RESULTS.glob("e2e_*.json"))
    path = Path(args.run) if args.run else (runs[-1] if runs else None)
    if path is None or not path.exists():
        parser.error("no e2e run found: run `python -m evals.eval_e2e --live` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["summary"]["ragas"] = means = score_records(payload["results"])
    md = markdown(payload["results"], means)
    print(md)
    common.write_report(path, payload, path.with_suffix(".md").read_text(encoding="utf-8").split("## RAGAS")[0] + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
