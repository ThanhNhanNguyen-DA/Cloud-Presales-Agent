"""
Cross-encoder reranking for RAG results.

Vector search gives a candidate list fast; a cross-encoder then reads (question, chunk)
pairs together and re-scores them. Measured on data/eval/retrieval_eval.json (60 questions,
candidates = top 15 from vector search, RTX 3060, fp16):

    first stage only            Hit@1 0.72  MRR 0.83
    + BAAI/bge-reranker-v2-m3   Hit@1 0.87  MRR 0.91   ~180 ms/query   (default)
    + mmarco-mMiniLMv2-L12      Hit@1 0.83  MRR 0.90   ~45 ms/query    (lighter option)

Scores use a sigmoid scale, NOT calibrated probabilities of answer correctness.
RERANK_MIN_SCORE (default 0.1) is a provisional relevance cutoff. Calibrate it on
reviewed positive AND negative queries with scripts.calibrate_retrieval.

Settings: RERANK_ENABLED (default true), RERANK_MODEL, RERANK_MIN_SCORE.
The model is loaded lazily (and warmed up at API start-up, see app/main.py).
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.1"))
if not math.isfinite(RERANK_MIN_SCORE) or not 0 <= RERANK_MIN_SCORE <= 1:
    raise ValueError("RERANK_MIN_SCORE must be a finite value in [0, 1].")

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
                _model = CrossEncoder(RERANK_MODEL, max_length=512, model_kwargs=kwargs,
                                      activation_fn=torch.nn.Sigmoid())
                logger.info("Loaded reranker %s", RERANK_MODEL)
    return _model


@dataclass
class RerankResult:
    documents: list[dict[str, Any]]
    status: str  # scored | disabled | unavailable
    threshold: float


def rank_documents(query: str, docs: list[dict[str, Any]], top_k: int,
                   min_score: float = RERANK_MIN_SCORE) -> RerankResult:
    """Filter scored evidence; a model failure never silently bypasses the cutoff."""
    if not docs:
        return RerankResult([], "scored", min_score)
    if not rerank_enabled():
        return RerankResult(docs[:top_k], "disabled", min_score)
    try:
        scores = get_reranker().predict([(query, d["content"]) for d in docs], batch_size=16)
        scores = [float(score) for score in scores]
        if len(scores) != len(docs) or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
            raise ValueError("Invalid reranker scores")
    except Exception as exc:  # noqa: BLE001 - never lose the answer because of the reranker
        logger.warning("Reranking unavailable; returning no unverified passages: %s", exc)
        return RerankResult([], "unavailable", min_score)
    ranked = sorted(
        ({**d, "rerank_score": float(s)} for d, s in zip(docs, scores, strict=True)),
        key=lambda d: d["rerank_score"], reverse=True,
    )
    # Repeated ingestion must not fill the context with copies of the same passage.
    unique, seen = [], {}
    for doc in ranked:
        key = (re.sub(r"\s+", " ", doc["content"]).strip().casefold(),
               doc.get("service_name"), (doc.get("metadata") or {}).get("provider"))
        if doc["rerank_score"] < min_score:
            continue
        if key in seen:
            if doc.get("id"):
                seen[key].setdefault("duplicate_ids", []).append(doc["id"])
            continue
        unique.append(doc)
        seen[key] = doc
    return RerankResult(unique[:top_k], "scored", min_score)


def rerank(query: str, docs: list[dict[str, Any]], top_k: int,
           min_score: float = RERANK_MIN_SCORE) -> list[dict[str, Any]]:
    """Compatibility helper for callers that only need the accepted documents."""
    return rank_documents(query, docs, top_k, min_score).documents
