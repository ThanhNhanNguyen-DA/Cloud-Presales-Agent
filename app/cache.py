"""
Answer cache for repeated questions (Redis), so an identical question costs no LLM tokens.

Key: the question normalised (Unicode NFC, case-folded, whitespace collapsed, trailing
punctuation dropped). Only used for a question WITHOUT conversation history — a follow-up
("còn tier Archive thì sao?") means something different in every conversation — and only
answers the grader let through are stored (no warnings, no errors).

Settings: REDIS_URL (unset = cache off), ANSWER_CACHE_TTL_SECONDS (default 86400).
Fail-open: if Redis is down, requests run uncached. Approving a correction clears the
cache, since it can change answers (see /admin/feedback/{id}/review).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import unicodedata

logger = logging.getLogger(__name__)

PREFIX = "qa:v1:"
TTL_SECONDS = int(os.getenv("ANSWER_CACHE_TTL_SECONDS", "86400"))

_client = None


def get_client():
    """Lazy Redis client; None when REDIS_URL is unset."""
    global _client
    url = os.getenv("REDIS_URL")
    if not url:
        return None
    if _client is None:
        import redis

        _client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=1, socket_connect_timeout=1)
    return _client


def normalize(question: str) -> str:
    text = unicodedata.normalize("NFC", question).casefold()
    return re.sub(r"[\s?!.。…]+$", "", re.sub(r"\s+", " ", text).strip())


def _key(question: str) -> str:
    return PREFIX + hashlib.sha256(normalize(question).encode()).hexdigest()


def get_answer(question: str) -> dict | None:
    client = get_client()
    if client is None:
        return None
    try:
        raw = client.get(_key(question))
    except Exception as exc:  # noqa: BLE001 - fail-open: a cache outage must not break chat
        logger.warning("Answer cache unavailable: %s", exc)
        return None
    return json.loads(raw) if raw else None


def cacheable(result: dict) -> bool:
    """Only clean answers: produced, and passed by the grader (general chat skips grading)."""
    return bool(result.get("final_response")) and (
        result.get("route") == "general" or result.get("grading_passed") is True)


def put_answer(question: str, result: dict, ragas: dict | None) -> None:
    client = get_client()
    if client is None or not cacheable(result):
        return
    value = {"route": result.get("route"), "response": result["final_response"],
             "raw_tool_output": result.get("raw_tool_output"), "ragas": ragas}
    try:
        client.setex(_key(question), TTL_SECONDS, json.dumps(value, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Answer cache write failed: %s", exc)


def clear() -> int:
    """Drop every cached answer (e.g. after a correction is approved). Returns the count."""
    client = get_client()
    if client is None:
        return 0
    try:
        keys = list(client.scan_iter(f"{PREFIX}*", count=500))
        return client.delete(*keys) if keys else 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("Answer cache clear failed: %s", exc)
        return 0
