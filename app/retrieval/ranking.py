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


# Vietnamese two-syllable phrases too common in questions to carry meaning.
_STOP_PHRASES = {
    "hỗ trợ", "dịch vụ", "bao nhiêu", "như thế", "thế nào", "là gì", "có thể", "sử dụng",
    "cung cấp", "cho biết", "hiện tại", "hiện nay", "nhất của", "tương đương",
}


def _is_latin(token: str) -> bool:
    """Product names, abbreviations and numbers: 'elb', 's3', 'bizfly', '116', 'v2'."""
    return token.isascii()


def lexical_query(query: str, mode: str = "or") -> str:
    """
    websearch_to_tsquery text: quoted terms joined by OR (never raw tsquery operators).

    The keyword channel is for EXACT IDENTIFIERS — prices ("116.820"), product names and
    abbreviations ("elb", "bizfly", "s3") — which embeddings handle poorly. Meaning is the
    vector channel's job. PostgreSQL's ts_rank_cd has no IDF, so OR-ing a rare identifier
    with a common phrase lets the common one win (measured: '"116.820" OR "đ tháng"' returned
    only FPT price tables; '"116.820"' alone returned the two chunks with that price).

      * If the question contains identifiers (Latin tokens, numbers), use only those.
      * Otherwise use adjacent two-syllable Vietnamese phrases ("cân bằng", "bằng tải") —
        never single syllables, which match almost every document — minus very common
        phrases; a lone syllable is used only when nothing else exists.
    Stop words split the question; phrases never cross a stop word.

    mode="and" requires every term (precise: '"770.000" "bizfly"'); the caller tries it
    first and falls back to mode="or" when nothing matches all terms.
    """
    # "116.820" stays one token: PostgreSQL's parser keeps dotted numbers as one lexeme.
    tokens = re.findall(r"\d+(?:\.\d+)+|[^\W_]+", query.lower())
    identifiers: list[str] = []
    phrases: list[str] = []
    lone_syllables: list[str] = []
    run: list[str] = []

    def flush_run() -> None:
        if len(run) == 1:
            lone_syllables.append(run[0])
        for a, b in zip(run, run[1:], strict=False):
            if f"{a} {b}" not in _STOP_PHRASES:
                phrases.append(f"{a} {b}")
        run.clear()

    for token in tokens:
        if token in _STOP_WORDS:
            flush_run()
        elif _is_latin(token):
            flush_run()
            if len(token) >= 2 or token.isdigit():
                identifiers.append(token)
        else:
            run.append(token)
    flush_run()
    terms = list(dict.fromkeys(identifiers or phrases or lone_syllables))
    joiner = " " if mode == "and" else " OR "   # websearch syntax: space = AND
    return joiner.join(f'"{term}"' for term in terms)[:2000]


# Abbreviations users type that the documents spell out (checked against the corpus: "KMS"
# appears in no chunk, the docs say "Key Management Service"). Explicit and small on purpose:
# a wrong expansion would steer retrieval, so add entries only for confirmed product names.
ALIASES = {
    "kms": "Key Management Service",
    "eip": "Elastic IP",
    "elb": "Elastic Load Balancer",
    "az": "Availability Zone",
    "vpc": "Virtual Private Cloud",
    "rds": "Relational Database Service",
    "vdi": "Virtual Desktop Infrastructure",
    "daas": "Desktop as a Service",
    "k8s": "Kubernetes",
    "lb": "Load Balancer",
}


def expand_aliases(query: str) -> str:
    """'KMS là gì?' -> 'KMS (Key Management Service) là gì?' — unless already spelled out."""
    def replace(match: re.Match) -> str:
        word = match.group(0)
        full = ALIASES[word.lower()]
        return word if full.lower() in query.lower() else f"{word} ({full})"
    pattern = r"\b(" + "|".join(map(re.escape, ALIASES)) + r")\b"
    return re.sub(pattern, replace, query, flags=re.IGNORECASE)


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
                       exclude: list[str] | None = None) -> list[dict]:
    """Re-check scope and provenance even when an old RPC ignores the filters."""
    result = []
    for row in rows:
        meta = row.get("metadata") or {}
        if category and row.get("category") != category:
            continue
        if row.get("category") in (exclude or []):
            continue
        source = str(meta.get("source") or "").strip()
        if not source or source.lower() == "sample_data" or not str(row.get("content") or "").strip():
            continue
        if meta.get("review_status") in {"pending", "rejected"}:
            continue
        result.append(row)
    return result
