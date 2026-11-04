"""
Cross-encoder reranking for RAG results.

Vector search gives a candidate list fast; a cross-encoder then reads (question, chunk)
pairs together and re-scores them. Measured on data/eval/retrieval_eval.json (60 questions,
candidates = top 15 from vector search, RTX 3060, fp16):

    first stage only            Hit@1 0.72  MRR 0.83
    + BAAI/bge-reranker-v2-m3   Hit@1 0.87  MRR 0.91   ~180 ms/query   (default)
    + mmarco-mMiniLMv2-L12      Hit@1 0.83  MRR 0.90   ~45 ms/query    (lighter option)

Scores are sigmoid probabilities. Relevant chunks scored ≥ 0.87, questions the corpus can't
answer ≤ 0.01, so RERANK_MIN_SCORE (default 0.1) cleanly drops irrelevant "best matches".

Settings: RERANK_ENABLED (default true), RERANK_MODEL, RERANK_MIN_SCORE.
The model is loaded lazily (and warmed up at API start-up, see app/main.py).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.1"))

_model = None
_lock = threading.Lock()


def rerank_enabled() -> bool:
    return os.getenv("RERANK_ENABLED", "true").strip().lower() not in {"0", "false", "no"}


def get_reranker():
    """Load the cross-encoder once (fp16 on GPU); thread-safe."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                import torch
                from sentence_transformers import CrossEncoder

                kwargs = {"torch_dtype": torch.float16} if torch.cuda.is_available() else {}
                _model = CrossEncoder(RERANK_MODEL, max_length=512, model_kwargs=kwargs)
                logger.info("Loaded reranker %s", RERANK_MODEL)
    return _model


def rerank(query: str, docs: list[dict[str, Any]], top_k: int,
           min_score: float = RERANK_MIN_SCORE) -> list[dict[str, Any]]:
    """
    Re-order `docs` (each with a "content" key) by cross-encoder relevance.

    Returns at most `top_k` docs scoring ≥ `min_score`, each with a "rerank_score" key.
    If reranking is disabled or fails, falls back to the input order (vector similarity).
    """
    if not docs:
        return []
    if not rerank_enabled():
        return docs[:top_k]
    try:
        scores = get_reranker().predict([(query, d["content"]) for d in docs], batch_size=16)
    except Exception as exc:  # noqa: BLE001 - never lose the answer because of the reranker
        logger.warning("Reranking failed, using vector order: %s", exc)
        return docs[:top_k]
    ranked = sorted(
        ({**d, "rerank_score": float(s)} for d, s in zip(docs, scores, strict=True)),
        key=lambda d: d["rerank_score"], reverse=True,
    )
    return [d for d in ranked if d["rerank_score"] >= min_score][:top_k]
