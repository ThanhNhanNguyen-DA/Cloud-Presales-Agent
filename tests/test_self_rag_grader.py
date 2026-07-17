"""
Tests for the Self-RAG Grounding Grader (Phase 8).

The grader makes a decoupled LLM call to compare the final answer against the raw
tool output. To keep these tests deterministic and free of API-key/quota needs (the
same pattern used by every other test in this repo), `grader_llm` is mocked.
"""

from unittest.mock import patch, MagicMock
from app.graders.self_rag_grader import grounding_grader_node


def _mock_grader_returning(verdict):
    """Build a mock grader_llm whose .invoke(...).content is the given verdict string."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=verdict)
    return mock_llm


def test_grader_pass_when_numbers_match():
    """Numbers in the answer match the raw output -> PASS."""
    state = {
        "raw_tool_output": "{'calculated_monthly_cost_usd': 50.0, 'estimated_hours': 2.5}",
        "final_response": "The monthly storage cost is 50 USD, and the backup takes about 2.5 hours.",
        "retry_count": 0,
    }
    with patch("app.graders.self_rag_grader.grader_llm", _mock_grader_returning("PASS")):
        result = grounding_grader_node(state)

    assert result["grading_passed"] is True


def test_grader_fail_when_numbers_mismatch():
    """Mismatched numbers after the retry budget is used -> FAIL + warning appended."""
    state = {
        "raw_tool_output": "{'calculated_monthly_cost_usd': 50.0, 'estimated_hours': 2.5}",
        "final_response": "The monthly storage cost is 5 USD, and the backup takes about 25 hours.",
        "retry_count": 1,  # Simulate having already retried once.
    }
    with patch("app.graders.self_rag_grader.grader_llm", _mock_grader_returning("FAIL")):
        result = grounding_grader_node(state)

    assert result["grading_passed"] is False
    assert "⚠️" in result["final_response"]


def test_grader_auto_pass_when_no_raw_output():
    """No raw tool data (e.g. the general route) -> auto PASS without an LLM call."""
    state = {
        "raw_tool_output": "",
        "final_response": "Hello! How can I help you with your cloud questions today?",
        "retry_count": 0,
    }
    mock_llm = _mock_grader_returning("FAIL")  # Should never be consulted.
    with patch("app.graders.self_rag_grader.grader_llm", mock_llm):
        result = grounding_grader_node(state)

    assert result["grading_passed"] is True
    mock_llm.invoke.assert_not_called()


def test_grader_first_failure_requests_retry_without_warning():
    """First failure bumps retry_count for a retry but does NOT warn the user yet."""
    state = {
        "raw_tool_output": "{'calculated_monthly_cost_usd': 50.0}",
        "final_response": "The monthly storage cost is 5 USD.",
        "retry_count": 0,
    }
    with patch("app.graders.self_rag_grader.grader_llm", _mock_grader_returning("FAIL")):
        result = grounding_grader_node(state)

    assert result["grading_passed"] is False
    assert result["retry_count"] == 1
    assert "⚠️" not in result["final_response"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
