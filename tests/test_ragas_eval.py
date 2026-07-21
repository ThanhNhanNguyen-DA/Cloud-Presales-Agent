"""
Tests for the RAGAS-style metric library (app/evaluation/metrics.py) used by the offline
batch evaluator. The judge LLM and the embedding model are mocked so nothing hits a live
service.
"""

from unittest.mock import MagicMock, patch

from app.evaluation import metrics


def test_llm_judged_metrics_parse_the_score():
    """faithfulness/context_recall/answer_correctness parse the judge's 0–1 number."""
    fake_judge = MagicMock()
    fake_judge.invoke.return_value = MagicMock(content="0.75")
    with patch.object(metrics, "_get_judge", return_value=fake_judge):
        assert metrics.faithfulness("answer", "context") == 0.75
        assert metrics.context_recall("ground truth", "context") == 0.75
        assert metrics.answer_correctness("answer", "ground truth") == 0.75
    print("✓ LLM-judged metrics parse the score")


def test_embedding_metrics_use_cosine():
    """answer_relevancy/answer_similarity are the cosine of the two texts' embeddings."""
    with patch.object(metrics, "embed_text", side_effect=lambda t: [1.0, 0.0]):
        # identical direction -> cosine 1.0
        assert metrics.answer_relevancy("q", "a") == 1.0
        assert metrics.answer_similarity("a", "gt") == 1.0
    print("✓ embedding metrics use cosine similarity")


def test_metrics_are_best_effort_on_failure():
    """A judge/embedding failure yields None for that metric, never an exception."""
    with patch.object(metrics, "_get_judge", side_effect=RuntimeError("no judge")):
        assert metrics.faithfulness("answer", "context") is None
    with patch.object(metrics, "embed_text", side_effect=RuntimeError("no model")):
        assert metrics.answer_relevancy("q", "a") is None
    print("✓ metrics degrade to None on failure")


def test_metrics_return_none_on_empty_input():
    """Empty answer/context/ground-truth means there is nothing to score."""
    assert metrics.faithfulness("", "context") is None
    assert metrics.context_recall("gt", "") is None
    assert metrics.answer_similarity("answer", "") is None
    print("✓ metrics return None for empty inputs")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
