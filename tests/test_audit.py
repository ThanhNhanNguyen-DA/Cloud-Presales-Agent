"""End-to-end audit: scoring maths, judge cache/budget, and layer formulas (no network)."""

import json
from unittest.mock import MagicMock

import numpy as np

from evals import audit


def _judge(tmp_path, replies, budget=10):
    llm = MagicMock()
    llm.invoke.side_effect = [MagicMock(content=r) for r in replies]
    return audit.Judge(llm, budget, cache_path=tmp_path / "cache.json"), llm


def test_judge_caches_and_respects_budget(tmp_path):
    judge, llm = _judge(tmp_path, ['noise {"overall": 8} noise', '{"overall": 3}'], budget=1)
    assert judge.ask("p1") == {"overall": 8}
    assert judge.ask("p1") == {"overall": 8} and llm.invoke.call_count == 1      # cached
    assert judge.ask("p2") is None and judge.skipped == 1                        # over budget
    again = audit.Judge(MagicMock(), 0, cache_path=tmp_path / "cache.json")
    assert again.ask("p1") == {"overall": 8}                                     # survives restarts


def test_judge_failure_is_not_cached(tmp_path):
    judge, _ = _judge(tmp_path, ["not json"])
    assert judge.ask("p") is None and not (tmp_path / "cache.json").exists()


def test_store_layer_scores_junk_duplicates_and_index_recall():
    def vec(i):
        v = np.zeros(512)
        v[i] = 1.0
        return json.dumps(v.tolist())
    rows = [{"id": f"d{i}", "category": "Feature", "content": f"Real documentation paragraph number {i} " * 5,
             "embedding": vec(i)} for i in range(20)]
    rows.append({"id": "junk", "category": "Feature", "content": "X › Metadata › Reference\n\nKhông áp dụng",
                 "embedding": vec(30)})
    rows.append({**rows[0], "id": "dup"})
    layer = audit.audit_store(rows, rpc_top10=lambda v: [f"d{i}" for i in range(5)],
                              sample_queries=["q"], embed=lambda q: np.ones(512) / np.sqrt(512))
    assert layer["metrics"]["junk"] == 1 and layer["metrics"]["duplicates"] == 1
    assert layer["parts"]["topic_coverage"] == 0.25
    assert any("SLA" in i for i in layer["issues"])
    assert 0 < layer["score"] < 100


def test_generation_and_evaluation_layers():
    records = [{"id": "a", "verdicts": ["PASS"], "n_facts": 1, "missing_facts": []},
               {"id": "b", "verdicts": ["FAIL", "FAIL"], "n_facts": 1, "missing_facts": ["99.9"]}]
    judged = [{"overall": 9, "faithfulness": 0.95, "correctness": 1.0, "issues": []},
              {"overall": 4, "faithfulness": 0.4, "correctness": 0.2, "issues": ["wrong SLA"]}]
    gen = audit.audit_generation(records, judged)
    assert gen["score"] == 65.0 and gen["issues"] == ["b: wrong SLA"]
    ev = audit.audit_evaluation(records, judged, n_cases=20, human_labels=False)
    assert ev["parts"]["judge_vs_grader"] == 1.0 and ev["parts"]["judge_vs_fact_checks"] == 1.0
    assert ev["parts"]["human_labels"] == 0.0 and any("human" in i for i in ev["issues"])


def test_overall_reweights_missing_layers():
    layers = {k: {"score": None} for k in audit.WEIGHTS}
    layers["L2_retrieval"]["score"], layers["L3_generation"]["score"] = 80.0, 60.0
    assert audit.overall(layers) == round((0.25 * 80 + 0.30 * 60) / 0.55, 1)
    assert audit.overall({k: {"score": None} for k in audit.WEIGHTS}) is None


def test_dashboard_reads_audits(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
    import eval_data
    (tmp_path / "audit_1.json").write_text(json.dumps({"overall": 70, "layers": {"L1_store": {"score": 70}}}))
    (tmp_path / "audit_2.json").write_text("{broken")
    audits = eval_data.load_audits(tmp_path)
    assert [a["name"] for a in audits] == ["audit_1"] and audits[0]["overall"] == 70
