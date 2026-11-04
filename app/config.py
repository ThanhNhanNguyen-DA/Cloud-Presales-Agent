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

# Ordered Groq fallback chain. The app tries the first model and FallbackChatLLM
# (app/llm.py) switches to the next when a model's rate limit is exhausted.
# Primary: GPT-OSS 120B; fallback: Qwen3.8 27B (both support tool calling on Groq).
# (The Llama 3.x models used originally were removed from Groq in 2026.)
# Override via env, e.g. GROQ_MODELS="openai/gpt-oss-120b,qwen/qwen3.8-27b".
GROQ_MODELS = [
    m.strip()
    for m in os.getenv(
        "GROQ_MODELS",
        "openai/gpt-oss-120b,qwen/qwen3.8-27b",
    ).split(",")
    if m.strip()
]


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
