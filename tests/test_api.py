"""
Tests for the FastAPI backend (Phase 9) using FastAPI's TestClient.

The graph and the feedback tool are mocked so the suite is deterministic and does
not require a live Groq/Supabase connection (the same pattern as the other tests).
"""

from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app, headers={"X-API-Key": "test-api-key"})


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_endpoint():
    """The endpoint is async, runs the graph via ainvoke, and returns the live RAGAS scorecard."""
    mock_result = {"route": "general", "final_response": "Hello! How can I help you today?",
                   "raw_tool_output": None}
    ragas = {"faithfulness": None, "answer_relevancy": 0.7}
    with patch("app.main.graph") as mock_graph, \
         patch("app.main.evaluate_answer_async", new=AsyncMock(return_value=ragas)) as mock_eval:
        mock_graph.ainvoke = AsyncMock(return_value=mock_result)
        response = client.post("/chat", json={"query": "Hello"})

    assert response.status_code == 200
    data = response.json()
    assert len(data["response"]) > 0
    assert data["route"] == "general"
    mock_eval.assert_awaited()          # scored live, after the answer
    assert data["ragas"] == ragas       # returned with the answer


def test_chat_stream_endpoint():
    # With stream_mode=["updates","messages"], the graph yields (mode, payload) tuples.
    async def fake_stream(_state, **_kwargs):  # **_kwargs absorbs stream_mode=[...]
        yield ("updates", {"supervisor": {"route": "general", "final_response": None}})
        yield ("updates", {"general": {"route": "general", "final_response": "Hi there!"}})

    ragas = {"faithfulness": None, "answer_relevancy": 0.7}
    with patch("app.main.graph") as mock_graph, \
         patch("app.main.evaluate_answer_async", new=AsyncMock(return_value=ragas)):
        mock_graph.astream = fake_stream
        response = client.post("/chat/stream", json={"query": "Hello"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert '"type": "progress"' in body
    assert '"node": "supervisor"' in body
    assert '"type": "ragas"' in body    # live scorecard emitted after the answer
    assert '"type": "final"' in body
    assert "Hi there!" in body


def _post_feedback(payload):
    """
    Call /feedback through the REAL `save_user_feedback` tool (so its input schema is
    genuinely validated), with only the DB and the embedding model mocked. These tests
    used to hit the live Supabase project and insert fake corrections into the feedback
    memory on every run.
    """
    with patch("app.tools.memory_tools.embed_text", return_value=[0.0] * 512), \
         patch("app.tools.memory_tools.supabase") as mock_sb:
        response = client.post("/feedback", json=payload)
    return response, mock_sb


def test_feedback_endpoint_upvote_no_correction():
    """Upvote case: `save_user_feedback` must accept correction=None."""
    response, mock_sb = _post_feedback({
        "query": "test query",
        "response": "test response",
        "is_positive": True,
        "correction": None
    })

    assert response.status_code == 200
    assert "recorded successfully" in response.json()["message"]
    mock_sb.table.assert_called_once_with("cloud_feedback_memory")
    row = mock_sb.table.return_value.insert.call_args.args[0]
    assert row["is_positive"] is True and row["user_correction"] is None


def test_feedback_endpoint_downvote_with_correction():
    """Downvote case with a correction string."""
    response, mock_sb = _post_feedback({
        "query": "test query",
        "response": "wrong response",
        "is_positive": False,
        "correction": "The correct answer is X."
    })

    assert response.status_code == 200
    assert "recorded successfully" in response.json()["message"]
    row = mock_sb.table.return_value.insert.call_args.args[0]
    assert row["is_positive"] is False
    assert row["user_correction"] == "The correct answer is X."


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
