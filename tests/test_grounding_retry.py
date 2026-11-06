"""
Day-3 tests: the shared tool loop, the grader verdict parsing, the bounded retry that re-runs
the SAME agent with the grader's feedback, and the streaming `reset` / `error` events.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk, SystemMessage

from app.agents import tool_agent
from app.evaluation.metrics import parse_score
from app.graders import self_rag_grader as grader
from app.graph.builder import build_graph
from app.graph.supervisor import parse_route
from app.main import app
from app.tools.presales_tools import calculate_backup_and_cost
from tests.conftest import TEST_API_KEY


def _tool_call(name="calculate_backup_and_cost", args=None, call_id="c1"):
    args = {"data_volume_gb": 10, "bandwidth_mbps": 100} if args is None else args
    return AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": args}])


def _scripted_llm(*replies):
    llm = MagicMock()
    llm.invoke.side_effect = list(replies)
    return llm


def _state(**overrides):
    state = {"messages": [], "user_query": "Backup 10GB over 100Mbps?", "route": "calculator",
             "retrieved_experience": None, "raw_tool_output": None, "agent_response": None,
             "final_response": None, "grading_passed": None, "retry_count": 0,
             "grader_feedback": None}
    state.update(overrides)
    return state


# --------------------------------------------------------------------------- tool loop
def test_tool_exception_becomes_error_result_not_crash():
    broken = MagicMock()
    broken.name = "calculate_backup_and_cost"
    broken.invoke.side_effect = RuntimeError("DB down")
    llm = _scripted_llm(_tool_call(), AIMessage(content="The tool failed, please retry later."))
    state = tool_agent.run_tool_agent(_state(), llm_with_tools=llm, tools=[broken], system_prompt="sys")
    assert "DB down" in state["raw_tool_output"] and "error" in state["raw_tool_output"]
    assert state["final_response"] == "The tool failed, please retry later."


def test_unknown_tool_name_is_reported_to_the_llm():
    llm = _scripted_llm(_tool_call(name="nope"), AIMessage(content="ok"))
    state = tool_agent.run_tool_agent(_state(), llm_with_tools=llm, tools=[calculate_backup_and_cost],
                                      system_prompt="sys")
    assert "not found" in state["raw_tool_output"]


def test_exhausted_tool_rounds_force_a_final_answer():
    # 1 initial + 2 rounds of tool calls, then the forced (tool_choice="none") answer.
    llm = _scripted_llm(*[_tool_call(call_id=f"c{i}") for i in range(3)],
                        AIMessage(content="Based on the results: 16.7 minutes."))
    state = tool_agent.run_tool_agent(_state(), llm_with_tools=llm, tools=[calculate_backup_and_cost],
                                      system_prompt="sys", max_rounds=2)
    assert state["final_response"] == "Based on the results: 16.7 minutes."
    forced = llm.invoke.call_args_list[-1]
    assert forced.kwargs == {"tool_choice": "none"}
    assert forced.args[0][-1].content == tool_agent.FINAL_ANSWER_NOTE
    assert state["raw_tool_output"].count("->") == 3     # the pending calls were executed too


@pytest.mark.parametrize("forced_reply", [
    AIMessage(content=""), _tool_call(call_id="again"), RuntimeError("model down"),
])
def test_exhausted_tool_rounds_never_return_empty_answer(forced_reply):
    llm = _scripted_llm(*[_tool_call(call_id=f"c{i}") for i in range(3)], forced_reply)
    state = tool_agent.run_tool_agent(_state(), llm_with_tools=llm, tools=[calculate_backup_and_cost],
                                      system_prompt="sys", max_rounds=2)
    assert state["final_response"] == tool_agent.INCOMPLETE_ANSWER


def test_empty_final_text_is_replaced():
    llm = _scripted_llm(AIMessage(content="   "))
    state = tool_agent.run_tool_agent(_state(), llm_with_tools=llm, tools=[], system_prompt="sys")
    assert state["final_response"] == tool_agent.INCOMPLETE_ANSWER


def test_raw_output_is_reset_on_every_run():
    llm = _scripted_llm(AIMessage(content="Please tell me the bandwidth."))
    state = tool_agent.run_tool_agent(_state(raw_tool_output="stale data from attempt 1"),
                                      llm_with_tools=llm, tools=[], system_prompt="sys")
    assert state["raw_tool_output"] is None


def test_grader_feedback_reaches_the_retry_prompt():
    llm = _scripted_llm(AIMessage(content="fixed"))
    tool_agent.run_tool_agent(_state(grader_feedback="- cost should be 0.23"), llm_with_tools=llm,
                              tools=[], system_prompt="sys")
    system = llm.invoke.call_args.args[0][0]
    assert isinstance(system, SystemMessage)
    assert "cost should be 0.23" in system.content and "failed an automatic fact check" in system.content


def test_list_content_blocks_are_joined():
    assert tool_agent.message_text(AIMessage(content=[{"type": "text", "text": "a"}, "b"])) == "ab"


# --------------------------------------------------------------------------- parsing
@pytest.mark.parametrize("reply, expected", [
    ('{"verdict": "PASS", "issues": []}', (True, [])),
    ('```json\n{"verdict": "FAIL", "issues": ["cost 0.5 vs 0.23"]}\n```', (False, ["cost 0.5 vs 0.23"])),
    ("PASS", (True, [])),
    ("fail", (False, [])),
    ("FAIL — this does not PASS", (False, [])),   # used to be read as PASS
    ("I am not sure", (None, [])),
    ('{"verdict": "MAYBE"}', (None, [])),
])
def test_parse_verdict(reply, expected):
    assert grader.parse_verdict(reply) == expected


@pytest.mark.parametrize("reply, route", [
    ("calculator", "calculator"), ("Calculator.", "calculator"), ("**document**", "document"),
    ("Route: competitor", "competitor"), ("no idea", "general"), ("", "general"),
])
def test_parse_route(reply, route):
    assert parse_route(reply) == route


@pytest.mark.parametrize("reply, score", [("0.85", 0.85), ("85%", 0.85), ("85", 0.85), ("250", None)])
def test_parse_score(reply, score):
    assert parse_score(reply) == score


# --------------------------------------------------------------------------- grader node
def _grade(reply=None, error=None, **state):
    llm = MagicMock()
    if error:
        llm.invoke.side_effect = error
    else:
        llm.invoke.return_value = MagicMock(content=reply)
    before = _state(raw_tool_output="{'cost': 0.23}", final_response="Cost 0.5 USD", **state)
    with patch.object(grader, "grader_llm", llm):
        return {**before, **grader.grounding_grader_node(before)}  # node returns a partial update


def test_first_fail_stores_feedback_for_retry():
    result = _grade('{"verdict": "FAIL", "issues": ["cost 0.5 != 0.23"]}')
    assert result["retry_count"] == 1
    assert result["grader_feedback"] == "- cost 0.5 != 0.23"
    assert "⚠️" not in result["final_response"]


def test_second_fail_warns_and_clears_feedback():
    result = _grade('{"verdict": "FAIL", "issues": ["x"]}', retry_count=1, grader_feedback="- x")
    assert result["retry_count"] == 2 and result["grader_feedback"] is None
    assert result["final_response"].endswith(grader.GRADING_WARNING)


@pytest.mark.parametrize("kwargs", [{"error": RuntimeError("rate limited")}, {"reply": "hmm"}])
def test_grader_failure_does_not_crash_or_loop(kwargs):
    result = _grade(**kwargs)
    assert result["grading_passed"] is False
    assert result["retry_count"] == grader.MAX_GRADING_ATTEMPTS  # no retry
    assert result["final_response"].endswith(grader.UNVERIFIED_WARNING)


# --------------------------------------------------------------------------- graph retry
def test_fail_then_pass_reruns_same_agent_with_feedback_once():
    supervisor_llm = MagicMock()
    supervisor_llm.invoke.return_value = MagicMock(content="calculator")
    calc_llm = _scripted_llm(
        _tool_call(call_id="a1"), AIMessage(content="Backup takes 30 minutes, 0.5 USD/month."),
        _tool_call(call_id="a2"), AIMessage(content="Backup takes 16.7 minutes, 0.23 USD/month."),
    )
    grader_llm = _scripted_llm(
        MagicMock(content='{"verdict": "FAIL", "issues": ["16.7 min, not 30"]}'),
        MagicMock(content='{"verdict": "PASS", "issues": []}'),
    )
    with patch("app.graph.supervisor.llm", supervisor_llm), \
         patch("app.graph.supervisor.retrieve_past_experiences", return_value="") as recall, \
         patch("app.agents.calculator_agent.llm_with_tools", calc_llm), \
         patch.object(grader, "grader_llm", grader_llm):
        result = build_graph().invoke(_state(route=None))

    assert result["final_response"] == "Backup takes 16.7 minutes, 0.23 USD/month."
    assert result["grading_passed"] is True and result["grader_feedback"] is None
    assert result["raw_tool_output"].count("estimated_minutes") == 1   # only the retry's tool call
    supervisor_llm.invoke.assert_called_once()                         # routing not repeated
    recall.assert_called_once()
    retry_system_prompt = calc_llm.invoke.call_args_list[2].args[0][0].content
    assert "16.7 min, not 30" in retry_system_prompt


def test_fail_twice_ends_with_warning():
    supervisor_llm = MagicMock()
    supervisor_llm.invoke.return_value = MagicMock(content="calculator")
    calc_llm = MagicMock()
    calc_llm.invoke.side_effect = [_tool_call(call_id="a1"), AIMessage(content="wrong"),
                                   _tool_call(call_id="a2"), AIMessage(content="still wrong")]
    grader_llm = MagicMock()
    grader_llm.invoke.return_value = MagicMock(content='{"verdict": "FAIL", "issues": ["x"]}')
    with patch("app.graph.supervisor.llm", supervisor_llm), \
         patch("app.graph.supervisor.retrieve_past_experiences", return_value=""), \
         patch("app.agents.calculator_agent.llm_with_tools", calc_llm), \
         patch.object(grader, "grader_llm", grader_llm):
        result = build_graph().invoke(_state(route=None))
    assert result["final_response"] == "still wrong" + grader.GRADING_WARNING
    assert grader_llm.invoke.call_count == 2


# --------------------------------------------------------------------------- streaming
client = TestClient(app, headers={"X-API-Key": TEST_API_KEY})


def _events(stream_items):
    """Run /chat/stream against a fake graph whose `astream` yields `stream_items`."""
    async def fake_astream(_state, **_kwargs):
        for item in stream_items:  # a list, or a (sync) generator that may raise
            yield item

    fake_graph = MagicMock()
    fake_graph.astream = fake_astream
    with patch("app.main.graph", fake_graph), \
         patch("app.main.evaluate_answer_async", new=AsyncMock(return_value={})), \
         patch("app.main.load_conversation_history", return_value=[]):
        response = client.post("/chat/stream", json={"query": "q"})
    return [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")]


def _token(text, msg_id, node="calculator"):
    return ("messages", (AIMessageChunk(content=text, id=msg_id), {"langgraph_node": node}))


def test_stream_withholds_specialist_drafts_until_grading_finishes():
    events = _events([
        _token("30 min", "m1"),
        ("updates", {"grader": {"final_response": "30 min", "raw_tool_output": "old"}}),
        _token("16.7 ", "m2"), _token("min", "m2"),
        ("updates", {"calculator": {"final_response": "16.7 min", "raw_tool_output": None}}),
        ("updates", {"grader": {"final_response": "16.7 min", "route": "calculator"}}),
    ])
    types = [e["type"] for e in events]
    assert types.count("reset") == 0
    assert [e["text"] for e in events if e["type"] == "token"] == ["16.7 min"]
    grader_steps = [i for i, event in enumerate(events) if event.get("node") == "grader"]
    assert types.index("token") > max(grader_steps)
    final = events[-1]
    assert final["type"] == "final" and final["response"] == "16.7 min"


def test_stream_ignores_supervisor_and_grader_tokens():
    events = _events([_token("calculator", "s1", node="supervisor"), _token("answer", "m1"),
                      _token('{"verdict":"PASS"}', "g1", node="grader"),
                      ("updates", {"grader": {"route": "calculator", "final_response": "checked answer"}})])
    assert [e["text"] for e in events if e["type"] == "token"] == ["checked answer"]
    assert "reset" not in [e["type"] for e in events]


def test_stream_reports_graph_failure():
    def boom():
        yield _token("partial", "m1")
        raise RuntimeError("all models down")
    events = _events(boom())
    assert events[-1]["type"] == "error"
    assert "all models down" not in json.dumps(events)
    assert not any(e["type"] == "token" for e in events)


def test_general_chat_still_streams_without_waiting_for_a_grader():
    events = _events([_token("Hello", "general-1", node="general")])
    assert [e["text"] for e in events if e["type"] == "token"] == ["Hello"]


def test_stream_delivers_the_grading_warning_with_the_answer():
    answer = "Partially supported answer" + grader.GRADING_WARNING
    events = _events([_token("Unverified draft", "m1"),
                      ("updates", {"grader": {"route": "document", "final_response": answer}})])
    assert [e["text"] for e in events if e["type"] == "token"] == [answer]


def test_chat_returns_502_when_graph_fails():
    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(side_effect=RuntimeError("all models down"))
    with patch("app.main.graph", fake_graph), \
         patch("app.main.load_conversation_history", return_value=[]):
        response = client.post("/chat", json={"query": "q"})
    assert response.status_code == 502 and "all models down" not in response.text


# --------------------------------------------------------------------------- found in live runs
def test_raw_output_records_tool_arguments():
    llm = _scripted_llm(_tool_call(), AIMessage(content="ok"))
    state = tool_agent.run_tool_agent(_state(), llm_with_tools=llm, tools=[calculate_backup_and_cost],
                                      system_prompt="sys")
    assert state["raw_tool_output"].startswith("calculate_backup_and_cost({'data_volume_gb': 10")


def test_grader_sees_the_user_question():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content='{"verdict": "PASS", "issues": []}')
    with patch.object(grader, "grader_llm", llm):
        grader.grounding_grader_node(_state(raw_tool_output="x", final_response="y"))
    assert "Backup 10GB over 100Mbps?" in llm.invoke.call_args.args[0]


@pytest.mark.parametrize("tool_path, param", [
    ("app.tools.rag_tools.search_cloud_documents", "category"),
    ("app.tools.competitor_tools.find_competitor_mapping", "competitor_name"),
    ("app.tools.competitor_tools.find_competitor_mapping", "competitor_service_name"),
])
def test_optional_tool_params_accept_null(tool_path, param):
    # Groq validates tool calls server-side: a non-nullable schema + a null argument = HTTP 400.
    import importlib

    from langchain_core.utils.function_calling import convert_to_openai_tool
    module, name = tool_path.rsplit(".", 1)
    schema = convert_to_openai_tool(getattr(importlib.import_module(module), name))
    types = {t.get("type") for t in schema["function"]["parameters"]["properties"][param]["anyOf"]}
    assert "null" in types
