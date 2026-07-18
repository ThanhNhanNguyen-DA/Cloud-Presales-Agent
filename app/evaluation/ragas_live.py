"""
Live, per-answer RAGAS-style scoring.

For every answer the system generates, this computes the two RAGAS metrics that need
NO ground truth (so they can run live, on each response):

  * faithfulness      — the fraction of the answer's factual claims that are supported
                        by the retrieved context, judged by the LLM. One extra Groq
                        call; returns None when there is no context (e.g. the 'general'
                        route or a tool that produced no output).
  * answer_relevancy  — cosine similarity between the question and the answer embeddings
                        (local Qwen; no extra LLM call). A lightweight proxy for RAGAS's
                        question-regeneration relevancy.

Design notes
------------
* Best-effort: any failure returns None for that metric and NEVER breaks the answer.
* Runs AFTER the answer is produced, so it does not delay time-to-first-token.
* The other two RAGAS metrics (context precision / recall) require a labelled
  ground-truth set and therefore cannot be computed live — use the full `ragas`
  library offline for those (see the Evaluation page).
"""

from __future__ import annotations

import asyncio
import re
from typing import Dict, Optional

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


def _cosine(a, b) -> float:
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
    return float(np.dot(va, vb) / denom)


def answer_relevancy(question: str, answer: str) -> Optional[float]:
    """Cosine similarity between question and answer embeddings, clamped to [0, 1]."""
    if not question.strip() or not answer.strip():
        return None
    try:
        score = _cosine(embed_text(question), embed_text(answer))
        return round(max(0.0, min(1.0, score)), 3)
    except Exception:
        return None


def faithfulness(answer: str, context: str) -> Optional[float]:
    """LLM-judged fraction of answer claims supported by the retrieved context."""
    if not answer.strip() or not context.strip():
        return None  # nothing to ground against (e.g. general route)
    try:
        prompt = _FAITHFULNESS_PROMPT.format(context=context[:6000], answer=answer[:4000])
        raw = _get_judge().invoke(prompt).content
        text = raw if isinstance(raw, str) else str(raw)
        match = re.search(r"\d?\.\d+|\d", text)
        if not match:
            return None
        return round(max(0.0, min(1.0, float(match.group()))), 3)
    except Exception:
        return None


def evaluate_answer(question: str, answer: str, context: str = "") -> Dict[str, Optional[float]]:
    """Compute the live RAGAS metrics for one (question, answer, context) triple (blocking)."""
    return {
        "faithfulness": faithfulness(answer, context),
        "answer_relevancy": answer_relevancy(question, answer),
    }


# --- Async variants -----------------------------------------------------------------
# The judge LLM call and the embedding forward-pass are both blocking/CPU-bound, so we
# offload them to worker threads with asyncio.to_thread. This keeps the FastAPI event
# loop free while evaluation runs in the background (see app/evaluation/interaction_log.py).
# The two metrics are independent, so they run concurrently with asyncio.gather.


async def evaluate_answer_async(question: str, answer: str, context: str = "") -> Dict[str, Optional[float]]:
    """Async, non-blocking version of `evaluate_answer` — the two metrics run concurrently."""
    fth, rel = await asyncio.gather(
        asyncio.to_thread(faithfulness, answer, context),
        asyncio.to_thread(answer_relevancy, question, answer),
    )
    return {"faithfulness": fth, "answer_relevancy": rel}
