"""
Stress-test the Grounding Grader (app/graders/self_rag_grader.py) on adversarial cases.

Each case of evals/datasets/adversarial_grader.jsonl is run through the REAL
`grounding_grader_node` (same prompt, parser, reviewed-corrections injection and
"could not verify" fallback as production), as the first attempt of an answer.

Positive class = a wrong answer (expected FAIL):
  FNR  = wrong answers the grader let through          target 0%
  FPR  = correct answers flagged FAIL (useless retry)  target < 5%
  unverified = grader errored / gave no verdict -> answer shipped with the ⚠️ caveat
  malformed  = reply was not the requested JSON (the bare-word fallback may still save it)

    python -m evals.eval_grader            # offline: lexical baseline grader, no API call
    python -m evals.eval_grader --live     # the configured GRADER_MODELS chain (app/llm.py)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from unittest.mock import patch

from evals import common

common.configure(live="--live" in sys.argv)  # before any app import (see tests/conftest.py)

from app.agents.tool_agent import message_text  # noqa: E402
from app.graders import self_rag_grader as grader  # noqa: E402

FNR_TARGET, FPR_TARGET = 0.0, 0.05


def _is_json_verdict(reply: str) -> bool:
    match = re.search(r"\{.*\}", reply or "", re.S)
    try:
        return str(json.loads(match.group()).get("verdict", "")).upper() in {"PASS", "FAIL"}
    except (AttributeError, json.JSONDecodeError):
        return False


def grade_case(case: dict, llm) -> dict:
    replies: list[str] = []

    class Recorder:  # keeps the raw reply so malformed JSON can be counted
        def invoke(self, prompt, *args, **kwargs):
            reply = llm.invoke(prompt, *args, **kwargs)
            replies.append(message_text(reply))
            return reply

    state = {"user_query": case["question"], "route": "calculator", "retry_count": 0,
             "raw_tool_output": case["raw_tool_output"], "final_response": case["candidate_answer"],
             "retrieved_experience": case.get("retrieved_experience")}
    started = time.perf_counter()
    with patch.object(grader, "grader_llm", Recorder()):
        update = grader.grounding_grader_node(state)
    if update.get("grading_passed"):
        got = "PASS"
    elif any(m in (update.get("final_response") or "") for m in grader.UNVERIFIED_MARKERS):
        got = "UNVERIFIED"
    else:
        got = "FAIL"
    return {"id": case["id"], "category": case["category"], "expected": case["expected_verdict"],
            "got": got, "issues": update.get("grader_feedback") or "",
            "json_ok": _is_json_verdict(replies[0] if replies else ""),
            "latency_s": round(time.perf_counter() - started, 3),
            "reply": (replies[0] if replies else "")[:500]}


def summarize(results: list[dict]) -> dict:
    bad = [r for r in results if r["expected"] == "FAIL"]
    good = [r for r in results if r["expected"] == "PASS"]
    lat = [r["latency_s"] for r in results]
    caught = sum(r["got"] == "FAIL" for r in bad)
    flagged = sum(r["got"] == "FAIL" for r in results)
    # F1 for "detects a wrong answer": an UNVERIFIED verdict counts as not detected.
    f1 = round(2 * caught / (flagged + len(bad)), 3) if flagged + len(bad) else None
    return {
        "n": len(results), "n_expected_fail": len(bad), "n_expected_pass": len(good),
        "fnr": common.rate(sum(r["got"] == "PASS" for r in bad), len(bad)),
        "fpr": common.rate(sum(r["got"] == "FAIL" for r in good), len(good)),
        "unverified_rate": common.rate(sum(r["got"] == "UNVERIFIED" for r in results), len(results)),
        "malformed_json_rate": common.rate(sum(not r["json_ok"] for r in results), len(results)),
        "accuracy": common.rate(sum(r["got"] == r["expected"] for r in results), len(results)), "f1": f1,
        "p50_s": common.percentile(lat, 50), "p95_s": common.percentile(lat, 95),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--live", action="store_true", help="Call the real grader LLM chain.")
    parser.add_argument("--dataset", default=str(common.DATASETS / "adversarial_grader.jsonl"))
    parser.add_argument("--out", default=common.default_out("grader"))
    args = parser.parse_args()

    llm = grader.grader_llm if args.live else common.HeuristicGraderLLM()
    results = []
    for case in common.load_jsonl(args.dataset):
        results.append(grade_case(case, llm))
        r = results[-1]
        print(f"{'✓' if r['got'] == r['expected'] else '✗'} {r['id']:32s} expected={r['expected']} got={r['got']}",
              flush=True)

    s = summarize(results)
    gates = {
        "FNR (missed wrong answers)": (s["fnr"], "= 0%", s["fnr"] is not None and s["fnr"] <= FNR_TARGET),
        "FPR (false alarms -> retry)": (s["fpr"], "< 5%", s["fpr"] is not None and s["fpr"] < FPR_TARGET),
        "Unverified / fallback rate": (s["unverified_rate"], "report", True),
        "Malformed JSON rate": (s["malformed_json_rate"], "report", True),
        "Accuracy / F1": (f"{s['accuracy']} / {s['f1']}", "report", True),
        "Latency p50 / p95 (s)": (f"{s['p50_s']} / {s['p95_s']}", "report", True),
    }
    passed = all(ok for _, _, ok in gates.values())
    mistakes = [{"id": r["id"], "category": r["category"], "expected": r["expected"], "got": r["got"],
                 "grader issues": r["issues"] or "—"} for r in results if r["got"] != r["expected"]]
    mode = "live" if args.live else "offline (lexical baseline grader — gates are informational)"
    md = (f"# Grounding grader eval — {mode}\n\n" + common.md_table(common.gate_rows(gates))
          + "\n## Mistakes\n\n" + common.md_table(mistakes))
    print("\n" + md)
    common.write_report(args.out, {"mode": "live" if args.live else "offline",
                                   "run_at": datetime.now().isoformat(timespec="seconds"), "summary": s,
                                   "passed": passed, "results": results}, md)
    # Only a live run measures the real grader; the offline baseline must not break CI.
    return 0 if passed or not args.live else 1


if __name__ == "__main__":
    sys.exit(main())
