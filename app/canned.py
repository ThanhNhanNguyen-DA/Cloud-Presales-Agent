"""
Reviewed canned answers (table cloud_canned_answers, migration 006).

A frequently asked question with a human-approved answer is answered without any LLM call.
It is invisible to users — they get an ordinary answer; the server log records `canned=<id>`.

A stored question matches when BOTH hold:
  * semantic: cosine similarity of the question embeddings >= CANNED_MIN_SIMILARITY;
  * identifiers: the numbers are exactly the same, and every product name / abbreviation in
    the asked question also appears in the stored one ("ELB" never matches a "VPN" answer,
    "10GB" never matches "100GB"). Generic words ("cloud", "cmc") are ignored.
Calibration (draft FAQ, 8 paraphrases + 11 near-misses): near-misses scored <= 0.82
("Key Management Service giá bao nhiêu?" vs "... dùng để làm gì?"), close paraphrases 0.92-0.99;
loose rewordings (0.62-0.85) fall through to the normal pipeline, which is the safe side.

Only status='approved', unexpired rows are returned by the RPC. Used only for questions
without conversation history. Fail-open: any error means "no canned answer".
Settings: CANNED_ANSWERS_ENABLED (default true), CANNED_MIN_SIMILARITY (default 0.90).
"""

from __future__ import annotations

import logging
import os
import re

from app.db.supabase_client import supabase
from app.retrieval.ranking import ALIASES, expand_aliases
from app.tools.embedding_tool import embed_query

logger = logging.getLogger(__name__)

MIN_SIMILARITY = float(os.getenv("CANNED_MIN_SIMILARITY", "0.90"))
_GENERIC = {"cmc", "cloud", "service", "services"}
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[^\W_]+")


def enabled() -> bool:
    return os.getenv("CANNED_ANSWERS_ENABLED", "true").strip().lower() not in {"0", "false", "no"}


def identifiers(text: str) -> tuple[set[str], set[str]]:
    """
    (numbers, product words) of a question, with known abbreviations spelled out.
    A Latin word counts as a product word only if written with a capital (ELB, AWS, Auto
    Scaling): unaccented Vietnamese syllables ("sao", "ra", "to") are plain ASCII too.
    """
    tokens = _TOKEN.findall(expand_aliases(text))
    numbers = {t.replace(",", ".") for t in tokens if t[0].isdigit()}
    words = {t.lower() for t in tokens
             if t.isascii() and not t[0].isdigit() and len(t) >= 2 and any(c.isupper() for c in t)}
    return numbers, words - _GENERIC - set(ALIASES)  # an abbreviation counts via its spelled-out words


def identifiers_match(asked: str, stored: str) -> bool:
    asked_numbers, asked_words = identifiers(asked)
    stored_numbers, stored_words = identifiers(stored)
    return asked_numbers == stored_numbers and asked_words <= stored_words


def find_answer(question: str) -> dict | None:
    """The best approved canned answer for `question`, or None."""
    if not enabled():
        return None
    try:
        rows = supabase.rpc("match_canned_answer", {
            "query_embedding": embed_query(question), "match_threshold": MIN_SIMILARITY,
            "match_count": 3}).execute().data or []
    except Exception as exc:  # noqa: BLE001 - fail-open: the normal pipeline answers instead
        logger.warning("Canned answers unavailable: %s", exc)
        return None
    for row in rows:  # ordered by similarity
        if identifiers_match(question, row["question"]):
            try:
                supabase.rpc("record_canned_hit", {"answer_id": row["id"]}).execute()
            except Exception as exc:  # noqa: BLE001 - usage counting is best-effort
                logger.warning("Could not record canned answer hit: %s", exc)
            return row
    return None
