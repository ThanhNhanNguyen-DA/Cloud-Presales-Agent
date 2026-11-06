"""Activity-log details attached to /chat/stream progress events."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.graders.self_rag_grader import GRADING_WARNING, UNVERIFIED_WARNING
from app.main import app
from app.progress import describe_step, found_data, tools_used
from tests.conftest import TEST_API_KEY

RAW_OK = "calculate_backup_and_cost({'data_volume_gb': 10}) -> {'estimated_minutes': 16.7}"
RAW_EMPTY = "search_cloud_documents({'query': 'x'}) -> No relevant documents found in the database."


def test_tools_used_and_found_data():
    raw = RAW_OK + "\n---\n" + RAW_EMPTY + "\n---\n" + RAW_OK
    assert tools_used(raw) == ["calculate_backup_and_cost", "search_cloud_documents"]
    assert found_data(raw) is True
    assert found_data(RAW_EMPTY) is False
    assert tools_used(None) == [] and found_data(None) is False


def test_supervisor_and_agent_details():
    assert describe_step("supervisor", {"route": "document", "retrieved_experience": "x"}, None) == \
        {"route": "document", "used_corrections": True}
    assert describe_step("calculator", {"raw_tool_output": RAW_OK}, None) == \
        {"tools": ["calculate_backup_and_cost"], "found_data": True}
    assert describe_step("document", {"raw_tool_output": None}, None) == {"tools": [], "found_data": False}
    assert describe_step("general", {"final_response": "hi"}, None) == {"tools": []}


def test_grader_verdicts():
    assert describe_step("grader", {"grading_passed": True}, None)["verdict"] == "skipped"
    assert describe_step("grader", {"grading_passed": True}, RAW_OK, attempt=2) == {"verdict": "pass", "attempt": 2}
    assert describe_step("grader", {"grading_passed": False, "grader_feedback": "- x"}, RAW_OK)["verdict"] == "retry"
    assert describe_step("grader", {"grading_passed": False, "final_response": "a" + GRADING_WARNING},
                         RAW_OK, attempt=2)["verdict"] == "warn"
    assert describe_step("grader", {"grading_passed": False, "final_response": "a" + UNVERIFIED_WARNING},
                         RAW_OK)["verdict"] == "unverified"


def test_stream_progress_events_carry_details():
    async def fake_astream(_state, **_kwargs):
        yield ("updates", {"supervisor": {"route": "calculator", "retrieved_experience": ""}})
        yield ("updates", {"calculator": {"raw_tool_output": RAW_OK, "final_response": "16.7 min"}})
        yield ("updates", {"grader": {"grading_passed": False, "grader_feedback": "- x", "retry_count": 1}})
        yield ("updates", {"calculator": {"raw_tool_output": RAW_OK, "final_response": "16.7 min"}})
        yield ("updates", {"grader": {"grading_passed": True, "grader_feedback": None}})

    graph = MagicMock()
    graph.astream = fake_astream
    client = TestClient(app, headers={"X-API-Key": TEST_API_KEY})
    with patch("app.main.graph", graph), \
         patch("app.main.evaluate_answer_async", new=AsyncMock(return_value={})), \
         patch("app.main.load_conversation_history", return_value=[]):
        text = client.post("/chat/stream", json={"query": "q"}).text
    progress = [json.loads(line[5:]) for line in text.splitlines()
                if line.startswith("data:") and '"progress"' in line]
    assert [p["node"] for p in progress] == ["supervisor", "calculator", "grader", "calculator", "grader"]
    assert progress[0]["detail"]["route"] == "calculator"
    assert progress[2]["detail"] == {"verdict": "retry", "attempt": 1}
    assert progress[4]["detail"] == {"verdict": "pass", "attempt": 2}


def test_stream_announces_tool_calls_early():
    from langchain_core.messages import AIMessageChunk

    async def fake_astream(_state, **_kwargs):
        chunk = AIMessageChunk(content="", tool_call_chunks=[
            {"name": "calculate_backup_and_cost", "args": "", "id": "c1", "index": 0}])
        yield ("messages", (chunk, {"langgraph_node": "calculator"}))
        yield ("messages", (AIMessageChunk(content="", tool_call_chunks=[
            {"name": None, "args": "{}", "id": None, "index": 0}]), {"langgraph_node": "calculator"}))
        yield ("messages", (AIMessageChunk(content="", tool_call_chunks=[
            {"name": "x", "args": "", "id": "g", "index": 0}]), {"langgraph_node": "supervisor"}))

    graph = MagicMock()
    graph.astream = fake_astream
    client = TestClient(app, headers={"X-API-Key": TEST_API_KEY})
    with patch("app.main.graph", graph), \
         patch("app.main.evaluate_answer_async", new=AsyncMock(return_value={})), \
         patch("app.main.load_conversation_history", return_value=[]):
        text = client.post("/chat/stream", json={"query": "q"}).text
    calls = [json.loads(line[5:]) for line in text.splitlines() if '"tool_call"' in line]
    assert [c["detail"] for c in calls] == [{"tool": "calculate_backup_and_cost", "agent": "calculator"}]
