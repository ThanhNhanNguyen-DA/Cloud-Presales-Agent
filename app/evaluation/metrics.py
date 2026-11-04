"""
RAGAS-style evaluation metrics, organised into the standard three groups.

Computed with the project's own building blocks — the Groq judge LLM and the local Qwen
embeddings — so no external `ragas` package is needed. Used by the offline batch evaluator
(scripts/run_ragas_eval.py).

Metric taxonomy
---------------
1. Retrieval metrics — evaluate the vector DB / retriever:
     * context_precision — signal-to-noise of the retrieved context (relevant vs noise).
     * context_recall    — did retrieval fetch everything the correct answer needs?
2. Generation metrics — evaluate the LLM response:
     * faithfulness      — is the answer strictly grounded in the retrieved context
                           (hallucination check)?
     * answer_relevancy  — how directly does the answer address the question?
3. End-to-end metrics — compare against a golden ground-truth answer:
     * answer_correctness — factual accuracy of the answer vs the ground truth.
     * answer_similarity  — semantic resemblance (embedding cosine) to the ground truth.

Design notes
------------
* Best-effort: any failure returns None for that metric and NEVER breaks the caller.
* LLM-judged metrics return a fraction in [0, 1]; embedding metrics return cosine in [0, 1].
"""

from __future__ import annotations

import asyncio
import re

import numpy as np

from app.llm import get_chat_llm
from app.tools.embedding_tool import embed_text

# The judge is built once, lazily (so importing this module is cheap and side-effect free).
_judge = None


def _get_judge():
    global _judge
    if _judge is None:
        _judge = get_chat_llm(temperature=0)
    return _judge


_NUMBER = re.compile(r"(\d+(?:[.,]\d+)?|[.,]\d+)\s*(%?)")


def parse_score(text: str) -> float | None:
    """
    First number in the judge's reply as a score in [0, 1].

    "0.85" -> 0.85, ".5" -> 0.5, "0,7" -> 0.7, "85%" -> 0.85, "85" -> 0.85 (a 0-100 scale),
    "Score: 1" -> 1.0. Anything above 100 is rejected (None) rather than clamped to 1.0.
    """
    match = _NUMBER.search(text or "")
    if not match:
        return None
    value = float(match.group(1).replace(",", "."))
    if match.group(2) == "%" or 1.0 < value <= 100.0:
        value /= 100.0
    if not 0.0 <= value <= 1.0:
        return None
    return round(value, 3)


def _score_from_llm(prompt: str) -> float | None:
    """Ask the judge for a single 0.0–1.0 number and parse it; None on any failure."""
    try:
        raw = _get_judge().invoke(prompt).content
        return parse_score(raw if isinstance(raw, str) else str(raw))
    except Exception:
        return None


def _cosine(a, b) -> float:
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
    return float(np.dot(va, vb) / denom)


def _cosine_texts(a: str, b: str) -> float | None:
    """Cosine similarity between two texts' embeddings, clamped to [0, 1]; None if empty."""
    if not a.strip() or not b.strip():
        return None
    try:
        return round(max(0.0, min(1.0, _cosine(embed_text(a), embed_text(b)))), 3)
    except Exception:
        return None


# =====================================================================================
# 1. Retrieval metrics
# =====================================================================================

_CONTEXT_PRECISION_PROMPT = """You are a strict retrieval-quality judge.
Given a QUESTION and the CONTEXT that a retrieval system returned for it, estimate the
signal-to-noise ratio: what fraction of the CONTEXT is actually relevant and useful for
answering the QUESTION (as opposed to noise / unrelated material)?

QUESTION:
{question}

CONTEXT:
{context}

Reply with ONLY a single number between 0.0 and 1.0
(1.0 = all of the context is relevant, 0.0 = none is). No words, no units."""


_CONTEXT_RECALL_PROMPT = """You are a strict retrieval-recall judge.
Given the GROUND-TRUTH answer and the retrieved CONTEXT, estimate what fraction of the
factual statements in the GROUND-TRUTH answer are supported by (can be found in) the
CONTEXT. This measures whether retrieval fetched everything needed to answer fully.

GROUND-TRUTH ANSWER:
{ground_truth}

CONTEXT:
{context}

Reply with ONLY a single number between 0.0 and 1.0
(1.0 = every ground-truth fact is present in the context, 0.0 = none is). No words."""


def context_precision(question: str, context: str) -> float | None:
    """Retrieval signal-to-noise: fraction of the retrieved context relevant to the question."""
    if not question.strip() or not context.strip():
        return None
    return _score_from_llm(_CONTEXT_PRECISION_PROMPT.format(
        question=question[:2000], context=context[:6000]))


def context_recall(ground_truth: str, context: str) -> float | None:
    """Retrieval completeness: fraction of ground-truth facts present in the retrieved context."""
    if not ground_truth.strip() or not context.strip():
        return None
    return _score_from_llm(_CONTEXT_RECALL_PROMPT.format(
        ground_truth=ground_truth[:4000], context=context[:6000]))


# =====================================================================================
# 2. Generation metrics
# =====================================================================================

_FAITHFULNESS_PROMPT = """You are a strict RAG faithfulness judge.
Given the CONTEXT (data retrieved by a tool or document search) and the ANSWER,
estimate what fraction of the factual claims in the ANSWER are directly supported by
the CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Reply with ONLY a single number between 0.0 and 1.0
(1.0 = every claim is supported by the context, 0.0 = none is). No words, no units."""


def faithfulness(answer: str, context: str) -> float | None:
    """Generation grounding: fraction of answer claims supported by the retrieved context."""
    if not answer.strip() or not context.strip():
        return None  # nothing to ground against (e.g. general route)
    return _score_from_llm(_FAITHFULNESS_PROMPT.format(
        context=context[:6000], answer=answer[:4000]))


def answer_relevancy(question: str, answer: str) -> float | None:
    """Generation relevance: how directly the answer addresses the question (embedding cosine)."""
    return _cosine_texts(question, answer)


# =====================================================================================
# 3. End-to-end metrics (need a golden ground-truth answer)
# =====================================================================================

_ANSWER_CORRECTNESS_PROMPT = """You are a strict answer-grading judge.
Compare the GENERATED ANSWER against the GROUND-TRUTH ("golden") answer and estimate its
overall factual correctness: do the facts, numbers, and conclusions in the generated
answer agree with the ground truth? Penalise wrong or contradictory facts.

GROUND-TRUTH ANSWER:
{ground_truth}

GENERATED ANSWER:
{answer}

Reply with ONLY a single number between 0.0 and 1.0
(1.0 = fully correct vs the ground truth, 0.0 = entirely wrong). No words, no units."""


def answer_correctness(answer: str, ground_truth: str) -> float | None:
    """End-to-end accuracy: factual agreement of the answer with the golden ground truth."""
    if not answer.strip() or not ground_truth.strip():
        return None
    return _score_from_llm(_ANSWER_CORRECTNESS_PROMPT.format(
        ground_truth=ground_truth[:4000], answer=answer[:4000]))


def answer_similarity(answer: str, ground_truth: str) -> float | None:
    """End-to-end semantic resemblance between the answer and the ground truth (embedding cosine)."""
    return _cosine_texts(answer, ground_truth)


# =====================================================================================
# Live scoring — the reference-free subset, scored per answer (no ground truth needed)
# =====================================================================================

async def evaluate_answer_async(question: str, answer: str, context: str = "") -> dict[str, float | None]:
    """
    Score one answer live, concurrently, with the two lightest reference-free metrics.

    Kept deliberately cheap — one Groq judge call (faithfulness) plus one local-embedding
    metric (answer_relevancy) — so live scoring adds minimal per-answer LLM load and rarely
    forces a rate-limit fallback. Each is offloaded to a worker thread via asyncio.to_thread
    and they run together with asyncio.gather, keeping the FastAPI event loop free. Called
    right after the answer is produced (see app/main.py). context_precision, being a second
    judge call, is left to the offline batch evaluator.
    """
    fth, rel = await asyncio.gather(
        asyncio.to_thread(faithfulness, answer, context),
        asyncio.to_thread(answer_relevancy, question, answer),
    )
    return {"faithfulness": fth, "answer_relevancy": rel}
