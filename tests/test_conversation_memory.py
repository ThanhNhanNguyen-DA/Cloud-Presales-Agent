"""
Tests for multi-turn conversation memory.

Covers the storage helpers (load/save) with Supabase mocked, and an end-to-end check
that preloaded history actually reaches a specialist agent's prompt through the graph.
Same mocking style as the rest of the suite — no live Groq/Supabase needed.
"""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from app.tools.conversation_tools import load_conversation_history, save_conversation_turn


def test_load_history_maps_and_orders_chronologically():
    """Rows come back newest-first from the DB; load must flip them to oldest-first and map roles."""
    rows_newest_first = [
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q1"},
    ]
    with patch("app.tools.conversation_tools.supabase") as mock_sb:
        chain = (mock_sb.table.return_value.select.return_value
                 .eq.return_value.order.return_value.limit.return_value)
        chain.execute.return_value = MagicMock(data=rows_newest_first)
        messages = load_conversation_history("sess-1", max_turns=5)

    assert [m.content for m in messages] == ["Q1", "A1", "Q2", "A2"]
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    # Window respected: asked the DB for at most 2 * max_turns rows.
    mock_sb.table.return_value.select.return_value.eq.return_value.order.return_value.limit.assert_called_with(10)
    print("✓ load_conversation_history maps roles and orders chronologically")


def test_load_history_none_session_skips_db():
    """A stateless call (no session_id) must return [] without touching Supabase."""
    with patch("app.tools.conversation_tools.supabase") as mock_sb:
        assert load_conversation_history(None) == []
        mock_sb.table.assert_not_called()
    print("✓ load_conversation_history is a no-op without a session_id")


def test_save_turn_inserts_expected_payload():
    with patch("app.tools.conversation_tools.supabase") as mock_sb:
        save_conversation_turn("sess-1", "assistant", "hi there", route="general")
        insert = mock_sb.table.return_value.insert
        insert.assert_called_once()
        payload = insert.call_args[0][0]
        assert payload == {
            "session_id": "sess-1",
            "role": "assistant",
            "content": "hi there",
            "route": "general",
        }
    print("✓ save_conversation_turn inserts the expected row")


def test_save_turn_noop_on_empty_inputs():
    """No session_id, or empty content, must not write to the DB."""
    with patch("app.tools.conversation_tools.supabase") as mock_sb:
        save_conversation_turn(None, "user", "hello")
        save_conversation_turn("sess-1", "assistant", "")
        mock_sb.table.assert_not_called()
    print("✓ save_conversation_turn skips empty/stateless writes")


def test_format_history_for_routing():
    """The supervisor's history block is rendered for follow-up routing (empty when no history)."""
    from app.graph.supervisor import _format_history_for_routing

    out = _format_history_for_routing([
        HumanMessage(content="What maps to AWS S3?"),
        AIMessage(content="Object Storage."),
    ])
    assert "Recent conversation" in out
    assert "User: What maps to AWS S3?" in out
    assert "Assistant: Object Storage." in out
    assert _format_history_for_routing([]) == ""
    print("✓ _format_history_for_routing renders recent turns")


def test_history_reaches_the_agent_prompt():
    """Preloaded history must be spliced into the specialist agent's message list, in order."""
    from app.graph.builder import build_graph

    captured = {}

    def mock_calc_invoke(messages):
        # Copy: the agent mutates this list (appends the AI reply) after we return.
        captured["messages"] = list(messages)
        result = MagicMock()
        result.content = "Recommended: 8 vCPU / 16GB."
        result.tool_calls = None  # no tool call -> grader auto-passes (no raw output)
        return result

    with patch("app.graph.supervisor.llm") as mock_sup_llm, \
         patch("app.graph.builder.llm") as mock_gen_llm, \
         patch("app.agents.calculator_agent.llm_with_tools") as mock_calc_llm, \
         patch("app.graph.supervisor.retrieve_past_experiences", return_value=""):

        mock_sup_llm.invoke = lambda prompt: MagicMock(content="calculator")
        mock_gen_llm.invoke = MagicMock(return_value=MagicMock(content="unused"))
        mock_calc_llm.invoke = mock_calc_invoke

        graph = build_graph()
        prior = [
            HumanMessage(content="My app runs on Java."),
            AIMessage(content="Understood, Java runtime."),
        ]
        result = graph.invoke({
            "messages": prior,
            "session_id": "sess-1",
            "user_query": "Now size a VM for 5000 CCU.",
            "route": None,
            "retrieved_experience": None,
            "raw_tool_output": None,
            "agent_response": None,
            "final_response": None,
            "grading_passed": None,
            "retry_count": 0,
        })

    assert result["route"] == "calculator"
    msgs = captured["messages"]
    contents = [getattr(m, "content", "") for m in msgs]
    assert "My app runs on Java." in contents, "prior user turn should be in the prompt"
    assert "Understood, Java runtime." in contents, "prior assistant turn should be in the prompt"
    # System prompt first, current question last.
    assert msgs[-1].content == "Now size a VM for 5000 CCU."
    print("✓ conversation history is injected into the agent prompt in order")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
