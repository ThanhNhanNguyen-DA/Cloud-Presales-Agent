"""Ensure the evaluator measures runtime acceptance without leaking gold categories."""

from unittest.mock import patch

import pytest

from app.tools.rag_tools import RetrievalResult
from scripts import eval_retrieval_production as evaluation


def passage(identifier, score):
    return {"id": identifier, "content": f"source {identifier}", "service_name": "ELB",
            "rerank_score": score, "metadata": {"source": "elb.docx"}}


def test_reports_runtime_hit_rate_and_keeps_unjudged_labels_unknown():
    returned = RetrievalResult([passage("wrong", 0.9), passage("gold", 0.8), passage("low", 0.01)],
                               "hybrid", "scored", 0.0)
    with patch.object(evaluation, "retrieve_documents", return_value=returned) as retrieve, \
         patch.object(evaluation, "RERANK_MIN_SCORE", 0.1):
        report = evaluation.evaluate_queries([{"id": "q1", "question": "ELB TLS?",
                                               "gold_id": "gold", "category": "Feature"}])
    assert report["metrics"]["hit_at_1"] == 0.0
    assert report["metrics"]["hit_at_k"] == 1.0 and report["metrics"]["mrr_at_k"] == 0.5
    args = retrieve.call_args.kwargs
    assert args["category"] is None and args["exclude"] == ["Competitor"]
    assert args["min_score"] == 0.0
    candidates = report["queries"][0]["candidates"]
    assert [c["relevant"] for c in candidates] == [None, True, None]


def test_competitor_evaluation_uses_provider_scope_and_reports_degradation():
    returned = RetrievalResult([], "vector", "scored", 0.0, ["Keyword unavailable"])
    with patch.object(evaluation, "retrieve_documents", return_value=returned) as retrieve:
        report = evaluation.evaluate_queries([{"question": "Bizfly pricing?", "scope": "competitor",
                                               "provider": "bizfly", "answerable": False}])
    assert retrieve.call_args.kwargs["category"] == "Competitor"
    assert retrieve.call_args.kwargs["provider"] == "bizfly"
    assert report["metrics"]["degraded_queries"] == 1
    assert report["metrics"]["hit_at_k"] is None


def test_unscored_results_cannot_be_used_to_calibrate():
    with patch.object(evaluation, "retrieve_documents", return_value=RetrievalResult([], "vector", "disabled", 0.0)):
        with pytest.raises(RuntimeError, match="available, enabled reranker"):
            evaluation.evaluate_queries([{"question": "ELB?"}])
