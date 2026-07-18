"""
Tests for automated background evaluation + interaction logging.

Async functions are driven with asyncio.run (no pytest plugin needed). Supabase and the
RAGAS metrics are mocked, so nothing hits a live service.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


def test_evaluate_answer_async_runs_metrics_concurrently():
    """The async evaluator returns both metrics, delegating to the (blocking) primitives."""
    from app.evaluation import ragas_live

    with patch.object(ragas_live, "faithfulness", return_value=0.9) as m_fth, \
         patch.object(ragas_live, "answer_relevancy", return_value=0.7) as m_rel:
        result = asyncio.run(ragas_live.evaluate_answer_async("q", "a", "ctx"))

    assert result == {"faithfulness": 0.9, "answer_relevancy": 0.7}
    m_fth.assert_called_once_with("a", "ctx")
    m_rel.assert_called_once_with("q", "a")
    print("✓ evaluate_answer_async returns both metrics")


def test_evaluate_and_log_inserts_row_with_metrics():
    """evaluate_and_log scores the answer then writes one row into cloud_interaction_logs."""
    from app.evaluation import interaction_log

    with patch.object(interaction_log, "evaluate_answer_async",
                      new=AsyncMock(return_value={"faithfulness": 0.8, "answer_relevancy": 0.6})), \
         patch.object(interaction_log, "supabase") as mock_sb:
        asyncio.run(interaction_log.evaluate_and_log(
            session_id="sess-1",
            query="How big is the VM?",
            response="8 vCPU / 16GB.",
            context="raw tool output",
            route="calculator",
            grading_passed=True,
            retry_count=0,
            ttft_seconds=0.42,
            total_seconds=1.1,
        ))

        insert = mock_sb.table.return_value.insert
        insert.assert_called_once()
        row = insert.call_args[0][0]

    assert row["route"] == "calculator"
    assert row["faithfulness"] == 0.8
    assert row["answer_relevancy"] == 0.6
    assert row["grading_passed"] is True
    assert row["ttft_seconds"] == 0.42
    print("✓ evaluate_and_log writes a row carrying the metrics")


def test_evaluate_and_log_swallows_errors():
    """A DB/scoring failure must never propagate (observability can't break the answer)."""
    from app.evaluation import interaction_log

    with patch.object(interaction_log, "evaluate_answer_async",
                      new=AsyncMock(side_effect=RuntimeError("boom"))):
        # Should not raise.
        asyncio.run(interaction_log.evaluate_and_log(
            session_id=None, query="q", response="a", context="c",
            route="general", grading_passed=None, retry_count=None,
        ))
    print("✓ evaluate_and_log is best-effort (errors swallowed)")


def test_load_recent_logs_returns_rows_and_degrades():
    """log_store reads recent rows, and returns [] on error instead of raising."""
    from app.evaluation import log_store

    with patch.object(log_store, "supabase") as mock_sb:
        chain = mock_sb.table.return_value.select.return_value.order.return_value.limit.return_value
        chain.execute.return_value = MagicMock(data=[{"route": "document"}])
        assert log_store.load_recent_logs(10) == [{"route": "document"}]

    with patch.object(log_store, "supabase") as mock_sb:
        mock_sb.table.side_effect = RuntimeError("unreachable")
        assert log_store.load_recent_logs(10) == []
    print("✓ load_recent_logs returns rows and degrades gracefully")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
