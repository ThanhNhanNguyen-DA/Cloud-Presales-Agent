"""Eval & Quality Dashboard: data derivation (pure) and an AppTest render of the admin page."""

import json
import sys
from pathlib import Path

UI_DIR = Path(__file__).resolve().parents[1] / "ui"
sys.path.insert(0, str(UI_DIR))

import eval_data as ed  # noqa: E402


def _record(id_, route, verdicts, passed=True, **extra):
    return {"id": id_, "route": route, "verdicts": verdicts, "passed": passed,
            "warned": bool(verdicts) and verdicts[-1] != "PASS", "latency_s": 1.0, **extra}


RECORDS = [
    _record("a", "calculator", ["PASS"]),
    _record("b", "document", ["FAIL", "PASS"], top_rerank_score=0.7, ragas={"faithfulness": 0.9}),
    _record("c", "document", ["FAIL", "FAIL"], passed=False, top_rerank_score=0.2, ragas={"faithfulness": 0.95},
            discrepancy="grader FAIL, faithfulness 0.95"),
    _record("d", "general", []),
]


def test_sankey_follows_route_grader_retry_outcome():
    labels, source, target, value = ed.sankey(RECORDS)
    links = {(labels[s], labels[t]): v for s, t, v in zip(source, target, value, strict=True)}
    assert links[("Route: document", "Grader #1: FAIL")] == 2
    assert links[("Grader #1: FAIL", "Retry (≤1)")] == 2
    assert links[("Retry (≤1)", "✅ Verified")] == 1 and links[("Retry (≤1)", "⚠️ Warning")] == 1
    assert links[("Grader #1: PASS", "✅ Verified")] == 1
    assert links[("Route: general", "Không cần kiểm chứng")] == 1


def test_scatter_explorer_and_kpis():
    assert [p["id"] for p in ed.scatter_points(RECORDS)] == ["b", "c"]
    rows = ed.explorer_rows(RECORDS, {"Grader ↔ RAGAS lệch pha"})
    assert [r["id"] for r in rows] == ["c"]
    assert ed.explorer_rows(RECORDS, set()) == []
    k = ed.kpis({"results": RECORDS, "summary": {"tier4_self_correction": {"heal_rate": 0.5}}})
    assert k == {"cases": 4, "pass_rate": 0.75, "recovery_rate": 0.5, "p95_latency_s": 1.0}


def test_load_runs_skips_broken_files_and_pairs_grader_by_time(tmp_path):
    (tmp_path / "e2e_1.json").write_text(json.dumps({"run_at": "2026-11-10", "summary": {}, "results": []}))
    (tmp_path / "e2e_2.json").write_text("{not json")
    for name, at, f1 in (("grader_1", "2026-11-09", 0.7), ("grader_2", "2026-11-11", 0.9)):
        (tmp_path / f"{name}.json").write_text(json.dumps({"run_at": at, "summary": {"f1": f1}, "results": []}))
    runs = ed.load_runs(tmp_path)
    assert [r["name"] for r in runs["e2e"]] == ["e2e_1"]
    assert ed.grader_for(runs["e2e"][0], runs["grader"])["summary"]["f1"] == 0.7


def test_demo_runs_feed_every_chart():
    runs = ed.demo_runs()
    current = runs["e2e"][-1]
    assert all(v is not None for v in ed.radar_values(current, ed.grader_for(current, runs["grader"])).values())
    assert ed.sankey(current["results"])[0] and ed.scatter_points(current["results"])


def _page(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(ed, "RESULTS_DIR", tmp_path)
    return AppTest.from_file(str(UI_DIR / "dashboard_page.py"), default_timeout=30)


def test_dashboard_requires_the_admin_key(monkeypatch, tmp_path):
    at = _page(monkeypatch, tmp_path).run()
    assert not at.exception
    assert at.text_input[0].label == "Admin API key" and not at.get("plotly_chart")
    at.text_input[0].input("wrong").run()
    assert any("Sai khóa" in e.value for e in at.error)
    at.text_input[0].input("test-admin-key").run()
    assert not at.exception and at.session_state["admin_ok"]


def test_dashboard_renders_demo_data_when_no_run_exists(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTH_DISABLED", "true")
    at = _page(monkeypatch, tmp_path).run()
    assert not at.exception
    assert any("dữ liệu mẫu" in i.value for i in at.info)
    assert len(at.get("plotly_chart")) == 3 and len(at.metric) == 4
