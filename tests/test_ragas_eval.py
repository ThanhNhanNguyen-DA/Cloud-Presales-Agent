"""
Tests for the inline RAGAS evaluator (the three reference-free metrics scored per answer).

The metric primitives are mocked so nothing hits a live Groq/embedding service; async is
driven with asyncio.run (no pytest plugin needed).
"""

import asyncio
from unittest.mock import patch


def test_evaluate_answer_returns_three_reference_free_metrics():
    """The sync evaluator returns exactly the reference-free trio, via the primitives."""
    from app.evaluation import ragas_live

    with patch.object(ragas_live, "context_precision", return_value=0.5), \
         patch.object(ragas_live, "faithfulness", return_value=0.9), \
         patch.object(ragas_live, "answer_relevancy", return_value=0.7):
        result = ragas_live.evaluate_answer("q", "a", "ctx")

    assert result == {"context_precision": 0.5, "faithfulness": 0.9, "answer_relevancy": 0.7}
    print("✓ evaluate_answer returns the reference-free trio")


def test_evaluate_answer_async_matches_sync_and_calls_primitives():
    """The async evaluator returns the same trio, delegating to the blocking primitives."""
    from app.evaluation import ragas_live

    with patch.object(ragas_live, "context_precision", return_value=0.4) as m_cp, \
         patch.object(ragas_live, "faithfulness", return_value=0.8) as m_fth, \
         patch.object(ragas_live, "answer_relevancy", return_value=0.6) as m_rel:
        result = asyncio.run(ragas_live.evaluate_answer_async("q", "a", "ctx"))

    assert result == {"context_precision": 0.4, "faithfulness": 0.8, "answer_relevancy": 0.6}
    m_cp.assert_called_once_with("q", "ctx")
    m_fth.assert_called_once_with("a", "ctx")
    m_rel.assert_called_once_with("q", "a")
    print("✓ evaluate_answer_async returns the reference-free trio")


def test_metrics_are_best_effort_on_failure():
    """A judge/embedding failure yields None for that metric, never an exception."""
    from app.evaluation import ragas_live

    with patch.object(ragas_live, "_get_judge", side_effect=RuntimeError("no judge")):
        assert ragas_live.faithfulness("answer", "context") is None
    with patch.object(ragas_live, "embed_text", side_effect=RuntimeError("no model")):
        assert ragas_live.answer_relevancy("q", "a") is None
    print("✓ metrics degrade to None on failure")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
