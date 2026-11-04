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
