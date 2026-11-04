"""
Day-4 tests: request ids, per-request LLM metrics, logging instead of print, and nodes that
return partial state updates.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app import observability
from app.agents import tool_agent
from app.graph.builder import general_node
from app.graph.supervisor import supervisor_node
from app.llm import FallbackChatLLM
from app.main import app
from tests.conftest import TEST_API_KEY

client = TestClient(app, headers={"X-API-Key": TEST_API_KEY})


def _chat(headers=None):
    with patch("app.main.graph") as g, \
         patch("app.main.evaluate_answer_async", new=AsyncMock(return_value={})), \
         patch("app.main.load_conversation_history", return_value=[]):
        g.ainvoke = AsyncMock(return_value={"route": "general", "final_response": "Hi"})
        return client.post("/chat", json={"query": "hello"}, headers=headers or {})


# --------------------------------------------------------------------------- request id
def test_response_carries_a_generated_request_id():
    rid = _chat().headers["X-Request-ID"]
    assert len(rid) == 32 and rid.isalnum()


def test_valid_incoming_request_id_is_kept():
    assert _chat({"X-Request-ID": "abc-123"}).headers["X-Request-ID"] == "abc-123"


def test_unsafe_incoming_request_id_is_replaced():
    rid = _chat({"X-Request-ID": "evil value; injected=1"}).headers["X-Request-ID"]
    assert "evil" not in rid and len(rid) == 32


def test_chat_logs_one_summary_line(caplog):
    with caplog.at_level(logging.INFO, logger="app.main"):
        _chat()
    summary = [r for r in caplog.records if "chat done" in r.getMessage()]
    assert len(summary) == 1
    assert "route=general" in summary[0].getMessage()
    assert "llm_calls=0" in summary[0].getMessage()


# --------------------------------------------------------------------------- LLM metrics
def test_llm_wrapper_records_latency_and_tokens():
    with patch("app.llm.ChatGroq"):
        llm = FallbackChatLLM(["m1"], api_key="k")
    reply = AIMessage(content="ok", usage_metadata={"input_tokens": 120, "output_tokens": 30,
                                                     "total_tokens": 150})
    fake = MagicMock()
    fake.invoke.return_value = reply
    llm._llms = [fake]

    metrics = observability.start_request("req-1")
    llm.invoke("hi")
    llm.invoke("again")
    assert metrics.llm_calls == 2
    assert (metrics.input_tokens, metrics.output_tokens) == (240, 60)
    assert metrics.summary()["tokens_in"] == 240


def test_metrics_are_safe_without_a_request():
    observability._metrics.set(None)
    with patch("app.llm.ChatGroq"):
        llm = FallbackChatLLM(["m1"], api_key="k")
    fake = MagicMock()
    fake.invoke.return_value = "plain"  # no usage_metadata at all
    llm._llms = [fake]
    assert llm.invoke("hi") == "plain"


def test_log_records_carry_the_request_id():
    observability.start_request("req-xyz")
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "msg", None, None)
    observability._RequestIdFilter().filter(record)
    assert record.request_id == "req-xyz"


def test_no_print_left_in_app_code():
    from pathlib import Path
    offenders = [str(p) for p in Path("app").rglob("*.py")
                 if p.name != "run_migration.py" and "print(" in p.read_text()]
    assert offenders == []


# --------------------------------------------------------------------------- partial updates
STATE = {"messages": [], "user_query": "hello", "route": None, "retrieved_experience": None,
         "raw_tool_output": "old", "agent_response": None, "final_response": None,
         "grading_passed": None, "retry_count": 0, "grader_feedback": None}


def test_supervisor_returns_only_its_keys():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="general")
    with patch("app.graph.supervisor.llm", llm), \
         patch("app.graph.supervisor.retrieve_past_experiences", return_value=""):
        update = supervisor_node(dict(STATE))
    assert update == {"route": "general", "retrieved_experience": ""}


def test_general_node_returns_only_its_keys():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="Hi!")
    with patch("app.graph.builder.llm", llm):
        assert general_node(dict(STATE)) == {"agent_response": "Hi!", "final_response": "Hi!"}


def test_tool_agent_returns_only_its_keys():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="answer")
    state = dict(STATE)
    update = tool_agent.run_tool_agent(state, llm_with_tools=llm, tools=[], system_prompt="s")
    assert set(update) == {"raw_tool_output", "agent_response", "final_response"}
    assert state["raw_tool_output"] == "old"  # input state is not mutated
