"""Model-chain configuration: provider specs and per-role chains."""

from unittest.mock import patch

from app import config
from app.llm import FallbackChatLLM, parse_model_spec


def test_parse_model_spec():
    assert parse_model_spec("openai/gpt-oss-120b") == ("groq", "openai/gpt-oss-120b")
    assert parse_model_spec("gemini:gemini-3.5-flash-lite") == ("gemini", "gemini-3.5-flash-lite")
    assert parse_model_spec("hf:Qwen/Qwen3-32B") == ("hf", "Qwen/Qwen3-32B")
    assert parse_model_spec("weird:thing") == ("groq", "weird:thing")


def test_role_defaults_and_env_override(monkeypatch):
    monkeypatch.delenv("SUPERVISOR_MODELS", raising=False)
    assert config.models_for("supervisor")[0] == "qwen/qwen3.8-27b"
    monkeypatch.setenv("SUPERVISOR_MODELS", "gemini:gemini-3.5-flash-lite, openai/gpt-oss-120b")
    assert config.models_for("supervisor") == ["gemini:gemini-3.5-flash-lite", "openai/gpt-oss-120b"]
    assert config.models_for("grader") == config.GROQ_MODELS
    assert config.models_for(None) == config.GROQ_MODELS


def test_models_without_a_key_are_skipped(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch("app.llm.ChatGroq"):
        llm = FallbackChatLLM(["gemini:gemini-3.5-flash-lite", "openai/gpt-oss-120b"], api_key="k")
    assert llm._models == ["openai/gpt-oss-120b"]


def test_gemini_models_use_the_openai_compatible_endpoint(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    with patch("langchain_openai.ChatOpenAI") as chat_openai:
        FallbackChatLLM(["gemini:gemini-3.5-flash-lite"], api_key="k")
    kwargs = chat_openai.call_args.kwargs
    assert kwargs["base_url"].startswith("https://generativelanguage.googleapis.com")
    assert kwargs["model"] == "gemini-3.5-flash-lite" and kwargs["reasoning_effort"] == "low"


def test_flatten_tool_history_for_gemini():
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    from app.llm import flatten_tool_history
    msgs = [SystemMessage("s"), HumanMessage("q"),
            AIMessage(content="", tool_calls=[{"name": "calc", "args": {"gb": 10}, "id": "fc_1"}]),
            ToolMessage(content="16.7 min", tool_call_id="fc_1")]
    flat = flatten_tool_history(msgs)
    assert flat[:2] == msgs[:2]
    assert isinstance(flat[2], AIMessage) and not flat[2].tool_calls
    assert flat[2].content == '[Called tools: calc({"gb": 10})]'
    assert isinstance(flat[3], HumanMessage) and "16.7 min" in flat[3].content
    assert flatten_tool_history("plain prompt") == "plain prompt"


def test_only_gemini_calls_are_flattened():
    from langchain_core.messages import AIMessage, ToolMessage
    with patch("app.llm.ChatGroq"):
        llm = FallbackChatLLM(["openai/gpt-oss-120b"], api_key="k")
    seen = []
    fake = type("F", (), {"invoke": lambda self, inp, **kw: seen.append(inp) or AIMessage(content="ok")})()
    msgs = [AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
            ToolMessage(content="r", tool_call_id="1")]
    llm._timed_invoke(fake, "openai/gpt-oss-120b", msgs, None, {})
    llm._timed_invoke(fake, "gemini:gemini-3.5-flash-lite", msgs, None, {})
    assert seen[0][0].tool_calls and not seen[1][0].tool_calls
