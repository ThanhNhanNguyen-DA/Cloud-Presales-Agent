import os
from dotenv import load_dotenv

load_dotenv()

# Groq is the LLM provider (Llama 3.3 70B). Embeddings run locally on Qwen, so no
# provider embedding key is needed — GROQ_API_KEY only pays for the chat calls.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# Local Hugging Face embedding model (Qwen3-Embedding-0.6B), truncated via MRL to EMBEDDING_DIM.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))

# Ordered Groq fallback chain. The app tries the first model and FallbackChatLLM
# (app/llm.py) switches to the next when a model's rate limit is exhausted.
# Primary: Llama 3.3 70B; fallback: a smaller, higher-quota model.
# Override via env, e.g. GROQ_MODELS="llama-3.3-70b-versatile,llama-3.1-8b-instant".
GROQ_MODELS = [
    m.strip()
    for m in os.getenv(
        "GROQ_MODELS",
        "llama-3.3-70b-versatile,llama-3.1-8b-instant",
    ).split(",")
    if m.strip()
]

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in .env")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL or SUPABASE_KEY is not set in .env")
