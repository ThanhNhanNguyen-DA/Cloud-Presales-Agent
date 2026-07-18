"""
Asynchronous, automated interaction logging.

Every answered question is scored and recorded into `cloud_interaction_logs` AFTER the
response has already been sent to the user, so evaluation never adds latency to the
answer. This is scheduled from the FastAPI endpoints as a background task.

The stored rows are what the Streamlit monitoring dashboard reads: route mix,
hallucination/retry rate (from the grounding grader), and the live RAGAS metrics
(faithfulness, answer relevancy) over time.

Everything here is best-effort: a failure to evaluate or persist is logged and
swallowed — observability must never break the product.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from app.db.supabase_client import supabase
from app.evaluation.ragas_live import evaluate_answer_async


async def evaluate_and_log(
    *,
    session_id: Optional[str],
    query: str,
    response: Optional[str],
    context: str,
    route: Optional[str],
    grading_passed: Optional[bool],
    retry_count: Optional[int],
    ttft_seconds: Optional[float] = None,
    total_seconds: Optional[float] = None,
) -> None:
    """
    Score one answer (live RAGAS) and append a row to `cloud_interaction_logs`.

    Runs in the background (non-blocking): the RAGAS metrics are computed with the async
    evaluator, and the DB insert is offloaded to a worker thread (the Supabase client is
    synchronous). Any error is caught so a logging hiccup never surfaces to the user.
    """
    try:
        ragas = await evaluate_answer_async(query, response or "", context or "")
        row = {
            "session_id": session_id,
            "user_query": query,
            "route": route,
            "final_response": response,
            "grading_passed": grading_passed,
            "retry_count": retry_count,
            "faithfulness": ragas.get("faithfulness"),
            "answer_relevancy": ragas.get("answer_relevancy"),
            "ttft_seconds": ttft_seconds,
            "total_seconds": total_seconds,
        }
        await asyncio.to_thread(
            lambda: supabase.table("cloud_interaction_logs").insert(row).execute()
        )
    except Exception as e:  # noqa: BLE001 - observability must never break the answer
        print(f"Error logging interaction: {str(e)}")
