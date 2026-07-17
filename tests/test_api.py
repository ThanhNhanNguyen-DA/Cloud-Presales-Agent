"""
Tests for the FastAPI backend (Phase 9) using FastAPI's TestClient.

The graph and the feedback tool are mocked so the suite is deterministic and does
not require a live Groq/Supabase connection (the same pattern as the other tests).
"""

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_endpoint():
    mock_result = {"route": "general", "final_response": "Hello! How can I help you today?"}
    with patch("app.main.graph") as mock_graph:
        mock_graph.invoke.return_value = mock_result
        response = client.post("/chat", json={"query": "Hello"})

    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert len(data["response"]) > 0
    assert data["route"] == "general"


def test_chat_stream_endpoint():
    def fake_stream(_state):
        yield {"supervisor": {"route": "general", "final_response": None}}
        yield {"general": {"route": "general", "final_response": "Hi there!"}}

    with patch("app.main.graph") as mock_graph:
        mock_graph.stream.side_effect = fake_stream
        response = client.post("/chat/stream", json={"query": "Hello"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    # Progress events for each node, then a final event carrying the answer.
    assert '"type": "progress"' in body
    assert '"node": "supervisor"' in body
    assert '"type": "final"' in body
    assert "Hi there!" in body


def test_feedback_endpoint_upvote_no_correction():
    """
    Upvote case: correction is None. Exercises the REAL tool (not mocked) so the
    input-schema validation is genuinely tested — `save_user_feedback` must accept
    correction=None. The tool catches any DB error internally and returns a string,
    so this stays 200 even without a live Supabase connection.
    """
    response = client.post("/feedback", json={
        "query": "test query",
        "response": "test response",
        "is_positive": True,
        "correction": None
    })

    assert response.status_code == 200
    assert "message" in response.json()


def test_feedback_endpoint_downvote_with_correction():
    """Downvote case with a correction string, against the real tool."""
    response = client.post("/feedback", json={
        "query": "test query",
        "response": "wrong response",
        "is_positive": False,
        "correction": "The correct answer is X."
    })

    assert response.status_code == 200
    assert "message" in response.json()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
