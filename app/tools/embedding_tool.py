"""
Shared embedding helper using a local Hugging Face model (Qwen3-Embedding-0.6B).

The model is natively 1024-dim but trained with Matryoshka (MRL), so it is
truncated to EMBEDDING_DIM (512) cleanly. It runs locally — no API key needed for
embeddings; the weights (~1.2 GB) download and cache on first use.

Used by the Document Agent (RAG), the Feedback Memory, and the ingestion scripts.
"""
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL, EMBEDDING_DIM


# Loaded once at import. First instantiation downloads + caches the model weights.
embedding_model = SentenceTransformer(EMBEDDING_MODEL, truncate_dim=EMBEDDING_DIM)


def embed_text(text: str) -> list:
    """
    Generate a normalized embedding vector for `text`.

    Args:
        text: The text to embed.

    Returns:
        A list of EMBEDDING_DIM (512) floats. Vectors are L2-normalized so cosine
        similarity (pgvector's `<=>` operator) behaves correctly.
    """
    vector = embedding_model.encode(text, normalize_embeddings=True)
    return vector.tolist()
