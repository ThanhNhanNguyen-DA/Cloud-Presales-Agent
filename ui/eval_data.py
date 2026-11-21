"""
Data behind the "Eval & Quality Dashboard" page: reads the runs written by evals/
(evals/results/e2e_*.json, grader_*.json) and derives each chart's input.
No Streamlit or Plotly here, so it can be unit-tested. When no run exists yet, `demo_runs()`
provides clearly-labelled sample data.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "evals" / "results"
RADAR_AXES = ("Routing Acc", "Faithfulness", "Answer Relevancy", "Context Recall", "Grader F1")
FILTERS = ("FAIL", "Warning", "Grader ↔ RAGAS lệch pha")


def load_runs(directory: str | Path | None = None) -> dict[str, list[dict]]:
    """{"e2e": [...], "grader": [...]}, oldest first; unreadable files are skipped."""
    directory = Path(directory or RESULTS_DIR)
    runs: dict[str, list[dict]] = {"e2e": [], "grader": []}
    for kind, items in runs.items():
        for path in sorted(directory.glob(f"{kind}_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(payload, dict) and isinstance(payload.get("results"), list):
                items.append({**payload, "name": path.stem})
    return runs


def load_audits(directory: str | Path | None = None) -> list[dict]:
    """End-to-end audits (evals/audit.py), oldest first; unreadable files are skipped."""
    audits = []
    for path in sorted(Path(directory or RESULTS_DIR).glob("audit_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("layers"), dict):
            audits.append({**payload, "name": path.stem})
    return audits


LAYER_LABELS = {"L1_store": "Kho vector / DB", "L2_retrieval": "Truy xuất", "L3_generation": "Sinh câu trả lời",
                "L4_agents": "Các agent", "L5_system": "Toàn hệ thống", "L6_evaluation": "Bộ đánh giá"}


def grader_for(e2e_run: dict, grader_runs: list[dict]) -> dict | None:
    """The latest grader run made before this e2e run (else the oldest one), for its F1."""
    before = [g for g in grader_runs if (g.get("run_at") or "") <= (e2e_run.get("run_at") or "")]
    return (before or grader_runs or [None])[-1]


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def kpis(run: dict) -> dict:
    records = run["results"]
    return {
        "cases": len(records),
        "pass_rate": sum(bool(r.get("passed")) for r in records) / len(records) if records else None,
        "recovery_rate": run["summary"].get("tier4_self_correction", {}).get("heal_rate"),
        "p95_latency_s": _percentile([r["latency_s"] for r in records if r.get("latency_s") is not None], 95),
    }


def radar_values(run: dict, grader_run: dict | None) -> dict[str, float | None]:
    summary = run["summary"]
    ragas = summary.get("ragas") or {}
    return {
        "Routing Acc": summary.get("tier1_routing", {}).get("accuracy"),
        "Faithfulness": ragas.get("faithfulness"),
        "Answer Relevancy": ragas.get("answer_relevancy"),
        "Context Recall": ragas.get("context_recall"),
        "Grader F1": ((grader_run or {}).get("summary") or {}).get("f1"),
    }


def final_outcome(record: dict) -> str:
    verdicts = record.get("verdicts") or []
    if not verdicts:
        return "Không cần kiểm chứng"
    return " Verified" if verdicts[-1] == "PASS" else " Warning"


def sankey(records: list[dict]) -> tuple[list[str], list[int], list[int], list[int]]:
    """Route -> grader attempt 1 -> retry -> final outcome, as Plotly Sankey arrays."""
    links: Counter = Counter()
    for r in records:
        route, verdicts = f"Route: {r.get('route') or '?'}", r.get("verdicts") or []
        if not verdicts:
            links[(route, final_outcome(r))] += 1
            continue
        first = f"Grader #1: {verdicts[0]}"
        links[(route, first)] += 1
        if verdicts[0] == "FAIL":
            links[(first, "Retry (≤1)")] += 1
            links[("Retry (≤1)", final_outcome(r))] += 1
        else:
            links[(first, final_outcome(r))] += 1
    labels = list(dict.fromkeys(node for pair in links for node in pair))
    return (labels, [labels.index(a) for a, _ in links], [labels.index(b) for _, b in links],
            list(links.values()))


def grader_verdict(record: dict) -> str:
    verdicts = record.get("verdicts") or []
    return verdicts[-1] if verdicts else "—"


def scatter_points(records: list[dict]) -> list[dict]:
    """Cases with both a rerank score and a faithfulness score (document route)."""
    points = []
    for r in records:
        x, y = r.get("top_rerank_score"), (r.get("ragas") or {}).get("faithfulness")
        if x is not None and y is not None:
            points.append({"id": r["id"], "x": x, "y": y, "verdict": grader_verdict(r)})
    return points


def flags(record: dict) -> list[str]:
    out = []
    if not record.get("passed"):
        out.append(FILTERS[0])
    if record.get("warned"):
        out.append(FILTERS[1])
    if record.get("discrepancy"):
        out.append(FILTERS[2])
    return out


def explorer_rows(records: list[dict], selected: set[str]) -> list[dict]:
    rows = []
    for r in records:
        hit = [f for f in flags(r) if f in selected]
        if hit:
            rows.append({"id": r["id"], "route": r.get("route"), "flags": ", ".join(hit),
                         "grader": " → ".join(r.get("verdicts") or []) or "—",
                         "faithfulness": (r.get("ragas") or {}).get("faithfulness"),
                         "top_rerank": r.get("top_rerank_score"), "question": r.get("question", "")})
    return rows


# --------------------------------------------------------------------------- demo data
_DEMO_QUESTIONS = {
    "general": "Xin chào, bạn giúp được gì?",
}


def _demo_run(name: str, run_at: str, quality: float, seed: int) -> dict:
    rng = random.Random(seed)
    routes = ["calculator"] * 8 + ["document"] * 10 + ["general"] * 4
    records = []
    for i, route in enumerate(routes):
        verdicts: list[str] = []
        if route != "general":
            verdicts = ["PASS"] if rng.random() < quality else ["FAIL", "PASS" if rng.random() < 0.7 else "FAIL"]
        rerank = round(rng.uniform(0.02, 0.98), 2) if route == "document" else None
        base = 0.4 + 0.55 * (rerank if rerank is not None else 0.9)
        faith = None if route == "general" else round(min(1.0, max(0.0, base + rng.uniform(-0.15, 0.1))), 2)
        ok = not verdicts or verdicts[-1] == "PASS"
        discrepancy = None
        if faith is not None and ok and faith < 0.8:
            discrepancy = f"grader PASS, faithfulness {faith:.2f}"
        records.append({
            "id": f"demo-{route}-{i}", "route": route, "expected_route": route,
            "route_ok": rng.random() < quality + 0.15, "passed": ok and rng.random() < quality + 0.1,
            "verdicts": verdicts, "warned": not ok, "discrepancy": discrepancy,
            "grader_issues": ["- (demo) 99.99% không khớp 99.95% trong dữ liệu gốc"] if "FAIL" in verdicts else [],
            "top_rerank_score": rerank,
            "ragas": {"faithfulness": faith, "answer_relevancy": None if route == "general" else round(rng.uniform(0.6, 0.95), 2),
                      "context_recall": round(rng.uniform(0.5, 1.0), 2) if rerank is not None else None},
            "latency_s": round(rng.uniform(1, 3) if route == "general" else rng.uniform(4, 14) * len(verdicts or [1]), 2),
            "question": _DEMO_QUESTIONS[route], "answer": "(dữ liệu mẫu) Câu trả lời của agent.",
            "raw_tool_output": "" if route == "general" else "tool({'demo': True}) -> {'demo_result': 1}",
        })

    def mean(key):
        vals = [r["ragas"][key] for r in records if r["ragas"][key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    retried = [r for r in records if r["verdicts"][:1] == ["FAIL"]]
    return {"name": name, "mode": "demo", "run_at": run_at, "rerank_min_score": 0.1, "results": records,
            "summary": {"tier1_routing": {"accuracy": round(sum(r["route_ok"] for r in records) / len(records), 3)},
                        "tier4_self_correction": {"heal_rate": round(sum(r["verdicts"][1] == "PASS" for r in retried)
                                                                     / len(retried), 3) if retried else None},
                        "ragas": {k: mean(k) for k in ("faithfulness", "answer_relevancy", "context_recall")}}}


def demo_runs() -> dict[str, list[dict]]:
    return {
        "e2e": [_demo_run("demo_baseline", "2026-11-01T09:00:00", 0.6, 1),
                _demo_run("demo_current", "2026-11-14T09:00:00", 0.8, 2)],
        "grader": [{"name": "demo_grader_1", "run_at": "2026-11-01T08:00:00", "summary": {"f1": 0.78}, "results": []},
                   {"name": "demo_grader_2", "run_at": "2026-11-14T08:00:00", "summary": {"f1": 0.9}, "results": []}],
    }
