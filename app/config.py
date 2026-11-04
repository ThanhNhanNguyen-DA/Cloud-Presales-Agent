import os
from dotenv import load_dotenv

load_dotenv()

# Groq is the LLM provider (GPT-OSS 120B, Qwen3.8 fallback). Embeddings run locally on Qwen, so no
# provider embedding key is needed — GROQ_API_KEY only pays for the chat calls.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# Local Hugging Face embedding model (Qwen3-Embedding-0.6B), truncated via MRL to EMBEDDING_DIM.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))

# Ordered model chains. FallbackChatLLM (app/llm.py) tries them in order and moves on when a
# model is rate-limited, removed, or returns an invalid tool call. Each entry is
# "provider:model" (groq | gemini | hf) or a bare Groq model name.
#   LLM_MODELS          default chain (specialist agents, general answers); GROQ_MODELS is
#                       accepted as the old name.
#   <ROLE>_MODELS       optional per-role chain: SUPERVISOR_MODELS, GRADER_MODELS, JUDGE_MODELS.
# Defaults: GPT-OSS 120B, then Qwen3.8 27B (the Llama 3.x models used originally were
# removed from Groq in 2026). Benchmarks: scripts/eval_orchestrator.py.
DEFAULT_MODELS = "openai/gpt-oss-120b,qwen/qwen3.8-27b"


def _model_list(value: str) -> list[str]:
    return [m.strip() for m in value.split(",") if m.strip()]


GROQ_MODELS = _model_list(os.getenv("LLM_MODELS") or os.getenv("GROQ_MODELS") or DEFAULT_MODELS)


# Per-role defaults, chosen with scripts/eval_orchestrator.py (54 routing questions, vi/en):
# qwen3.8-27b routed 100% correctly at 0.35 s p50 (gpt-oss-120b: 98%, 0.75 s) — and moving
# routing off gpt-oss-120b leaves more of its free-tier quota for the agents and the grader.
ROLE_DEFAULTS = {"supervisor": "qwen/qwen3.8-27b,openai/gpt-oss-120b"}


def models_for(role: str | None = None) -> list[str]:
    """Model chain for a role: <ROLE>_MODELS env, else the role default, else the default chain."""
    if role:
        override = _model_list(os.getenv(f"{role.upper()}_MODELS", "") or ROLE_DEFAULTS.get(role, ""))
        if override:
            return override
    return GROQ_MODELS


def validate_config() -> None:
    """
    Fail fast when a required setting is missing.

    Called when the API starts (and before the first LLM / Supabase client is built),
    not at import time — so tests and tooling can import the app without a real .env.
    """
    missing = [
        name
        for name, value in (
            ("GROQ_API_KEY", GROQ_API_KEY),
            ("SUPABASE_URL", SUPABASE_URL),
            ("SUPABASE_KEY", SUPABASE_KEY),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required settings in .env: {', '.join(missing)}")
