"""
Shared embedding helper using a local Hugging Face model (Qwen3-Embedding-0.6B).

The model is natively 1024-dim but trained with Matryoshka (MRL), so it is
truncated to EMBEDDING_DIM (512) cleanly. It runs locally — no API key needed for
embeddings; the weights (~1.2 GB) download and cache on first use.

The model (and torch) are loaded lazily on the first `embed_text` call, not at import,
so importing the app / running the test suite doesn't pay the ~15 s load. The API server
warms it up at startup instead (app/main.py lifespan), so the first request isn't slow.

Used by the Document Agent (RAG), the Feedback Memory, and the ingestion scripts.
"""
import threading

from app.config import EMBEDDING_DIM, EMBEDDING_MODEL

_model = None
_model_lock = threading.Lock()


def get_embedding_model():
    """Load the SentenceTransformer once (thread-safe) and reuse it."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # another thread may have loaded it while we waited
                from sentence_transformers import SentenceTransformer  # heavy (torch) — deferred

                _model = SentenceTransformer(EMBEDDING_MODEL, truncate_dim=EMBEDDING_DIM)
    return _model


def embed_text(text: str) -> list:
    """
    Generate a normalized embedding vector for `text`.

    Args:
        text: The text to embed.

    Returns:
        A list of EMBEDDING_DIM (512) floats. Vectors are L2-normalized so cosine
        similarity (pgvector's `<=>` operator) behaves correctly.
    """
    vector = get_embedding_model().encode(text, normalize_embeddings=True)
    return vector.tolist()
