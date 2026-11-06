"""Fuse independent rankings without treating their scores as probabilities."""

from __future__ import annotations

import hashlib
import re

# Keep product names, abbreviations and numbers; omit common question scaffolding.
_STOP_WORDS = set("""
    a an and are as at be by can do does for from how i in is it of on or our the to
    what when where which with you your
    anh bao bạn bên cho có của cần cách các gì giúp hãy không là làm mình nào những
    một này nhiêu như sao thì thông tin tôi trong từ và về với được đó
""".split())


def lexical_query(query: str) -> str:
    """Literal OR terms for websearch_to_tsquery, never raw tsquery operators."""
    terms = dict.fromkeys(re.findall(r"[^\W_]+", query.lower()))
    return " OR ".join(f'"{term}"' for term in terms if term not in _STOP_WORDS)[:2000]


def document_key(doc: dict) -> str:
    if doc.get("id"):
        return str(doc["id"])
    value = f"{doc.get('service_name')}|{doc.get('metadata')}|{doc.get('content')}"
    return hashlib.sha256(value.encode()).hexdigest()


def reciprocal_rank_fusion(*rankings: list[dict], limit: int = 30, k: int = 60) -> list[dict]:
    """One vote per document per ranking; stable ties, original metadata preserved."""
    documents: dict[str, dict] = {}
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen = set()
        for rank, doc in enumerate(ranking, 1):
            key = document_key(doc)
            if key in seen:
                continue
            seen.add(key)
            documents[key] = {**documents.get(key, {}), **doc}
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    keys = sorted(documents, key=lambda key: scores[key], reverse=True)[:limit]
    return [{**documents[key], "fusion_score": scores[key]} for key in keys]


def eligible_documents(rows: list[dict], *, category: str | None = None,
                       exclude: list[str] | None = None, provider: str | None = None) -> list[dict]:
    """Re-check scope and provenance even when an old RPC ignores the filters."""
    result = []
    for row in rows:
        meta = row.get("metadata") or {}
        if category and row.get("category") != category:
            continue
        if row.get("category") in (exclude or []):
            continue
        if "Competitor" in (exclude or []) and meta.get("role") == "competitor":
            continue
        if provider and meta.get("provider") != provider:
            continue
        source = str(meta.get("source") or "").strip()
        if not source or source.lower() == "sample_data" or not str(row.get("content") or "").strip():
            continue
        if meta.get("review_status") in {"pending", "rejected"}:
            continue
        result.append(row)
    return result
