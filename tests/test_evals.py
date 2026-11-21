"""The offline evaluation suite (evals/) runs green AND can still fail: a harness that cannot fail measures nothing."""

import json
import sys

from evals import common, eval_e2e, eval_grader


def test_number_aware_matching():
    assert common.contains("uptime **99,9 %**", "99.9")
    assert not common.contains("uptime 99.95%", "99.9")          # a drifted SLA is not a match
    assert common.contains("khoảng 1.198,81 GB", ["1198.81", "1.198,81"])
    assert common.contains("26 vCPU / 44 GB", "26")
    assert common.unsupported_numbers("giả định 100 Mbps: mất 8,33 giờ", "Backup 300GB 0 Mbps",
                                      "{'error': 'bandwidth_mbps must be greater than 0'}") == ["100", "8.33"]


def test_offline_e2e_passes_and_catches_an_isolation_leak(monkeypatch, tmp_path):
    out = tmp_path / "e2e.json"
    monkeypatch.setattr(sys, "argv", ["eval_e2e", "--out", str(out)])
    assert eval_e2e.main() == 0
    memory = json.loads(out.read_text())["summary"]["tier1_routing"]["memory"]
    assert memory["pending_correction_excluded"] and memory["injection_tags_neutralised"]

    # Remove the Python-side scope re-check: partner case studies now reach our document search.
    monkeypatch.setattr("app.tools.rag_tools.eligible_documents", lambda rows, **_: rows)
    assert eval_e2e.main() == 1
    assert "case study chunk" in out.with_suffix(".md").read_text()


def test_ragas_metrics_per_route_and_grader_cross_check(monkeypatch):
    from app.evaluation import metrics
    from evals import eval_ragas

    monkeypatch.setattr(metrics, "faithfulness", lambda answer, ctx: 0.5 if "sai" in answer else 0.95)
    monkeypatch.setattr(metrics, "answer_relevancy", lambda q, a: 0.8)
    monkeypatch.setattr(metrics, "context_precision", lambda q, ctx: 0.7)
    monkeypatch.setattr(metrics, "context_recall", lambda ref, ctx: 1.0 if "99.9" in ref else 0.0)
    records = [
        {"id": "calc", "route": "calculator", "question": "q", "answer": "đúng", "verdicts": ["PASS"]},
        {"id": "doc", "route": "document", "question": "q", "answer": "sai", "verdicts": ["PASS"],
         "required_facts": ["99.9", ["10%", "mười"]]},
        {"id": "comp", "route": "document", "question": "q", "answer": "đúng", "verdicts": ["FAIL", "FAIL"]},
        {"id": "gen", "route": "general", "question": "q", "answer": "hi", "verdicts": []},
    ]
    means = eval_ragas.score_records(records)
    calc, doc, comp, gen = records
    assert calc["ragas"]["context_precision"] is None and calc["ragas"]["faithfulness"] == 0.95
    assert doc["ragas"]["context_recall"] == 1.0 and all(v is None for v in gen["ragas"].values())
    assert doc["discrepancy"] == "grader PASS, faithfulness 0.50"
    assert comp["discrepancy"] == "grader FAIL, faithfulness 0.95" and calc["discrepancy"] is None
    assert means["faithfulness"] == 0.8


def test_offline_grader_reports_metrics(monkeypatch, tmp_path):
    out = tmp_path / "grader.json"
    monkeypatch.setattr(sys, "argv", ["eval_grader", "--out", str(out)])
    assert eval_grader.main() == 0                                 # offline gates are informational
    summary = json.loads(out.read_text())["summary"]
    assert summary["n_expected_fail"] == 10 and summary["n_expected_pass"] == 10
    assert summary["fnr"] is not None and summary["malformed_json_rate"] == 0
