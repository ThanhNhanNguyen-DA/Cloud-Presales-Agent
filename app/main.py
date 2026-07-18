"""
Async FastAPI backend wrapping the LangGraph Cloud Presales agent.

Exposes:
- POST /chat         -> full (non-streaming) response
- POST /chat/stream  -> Server-Sent Events: step-by-step progress then the final answer
- POST /feedback     -> records an Upvote/Downvote (+ optional correction) for the feedback memory
- GET  /health       -> liveness check

Async design (pragmatic):
- Endpoints are `async def`; the graph runs via `ainvoke`/a threaded `stream` so the
  event loop is never blocked by the (synchronous) LLM, Supabase, and embedding calls.
- Blocking helpers (conversation load/save, feedback insert) are offloaded with
  `asyncio.to_thread`.
- Evaluation + metrics logging happen in a BACKGROUND task AFTER the answer is sent, so
  RAGAS scoring never adds latency. The scores land in `cloud_interaction_logs`, which
  the Streamlit dashboard reads — evaluation is fully automated, not inline.
"""

import asyncio
import json
import time
from typing import Optional

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from app.graph.builder import build_graph
from app.tools.memory_tools import save_user_feedback
from app.tools.conversation_tools import load_conversation_history, save_conversation_turn
from app.evaluation.interaction_log import evaluate_and_log

app = FastAPI(title="Cloud Presales Agentic RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for easy local Streamlit testing; restrict before deploying.
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None  # groups turns of one conversation; None = stateless


class ChatResponse(BaseModel):
    route: str
    response: str
    # RAGAS is no longer returned inline: it is computed in the background and written to
    # cloud_interaction_logs, then viewed on the Streamlit monitoring dashboard.


class FeedbackRequest(BaseModel):
    query: str
    response: str
    is_positive: bool
    correction: Optional[str] = None


def _initial_state(query: str, session_id: Optional[str] = None) -> dict:
    """
    Build a fresh GraphState dict for a single request.

    Loads the recent turns of `session_id` (if given) into `messages` so the agents
    and supervisor can answer follow-up questions in context. With no session_id the
    call is stateless (empty history). Synchronous (hits Supabase) — callers offload it
    with asyncio.to_thread.
    """
    return {
        "messages": load_conversation_history(session_id),
        "session_id": session_id,
        "user_query": query,
        "route": None,
        "retrieved_experience": None,
        "raw_tool_output": None,
        "agent_response": None,
        "final_response": None,
        "grading_passed": None,
        "retry_count": 0,
    }


def _persist_turn(session_id: Optional[str], query: str, response: Optional[str], route: Optional[str]) -> None:
    """Append the user question and the assistant answer to the conversation memory (no-op if stateless)."""
    if not session_id:
        return
    save_conversation_turn(session_id, "user", query)
    save_conversation_turn(session_id, "assistant", response or "", route=route)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Run the full graph and return the final answer. Evaluation/logging runs in the background."""
    state = await asyncio.to_thread(_initial_state, request.query, request.session_id)
    result = await graph.ainvoke(state)

    # Save the conversation turn in the foreground so a fast follow-up sees it.
    await asyncio.to_thread(
        _persist_turn, request.session_id, request.query,
        result.get("final_response"), result.get("route"),
    )

    # Score + log AFTER responding, so RAGAS never adds latency.
    background_tasks.add_task(
        evaluate_and_log,
        session_id=request.session_id,
        query=request.query,
        response=result.get("final_response"),
        context=result.get("raw_tool_output") or "",
        route=result.get("route"),
        grading_passed=result.get("grading_passed"),
        retry_count=result.get("retry_count"),
    )
    return ChatResponse(route=result["route"], response=result["final_response"])


# Nodes that produce the user-facing answer. We stream tokens ONLY from these — the
# supervisor (intent classification) and grader also call the LLM, but their tokens are
# internal and must not leak into the answer.
ANSWER_NODES = {"calculator", "document", "competitor", "general"}

_STREAM_END = object()  # sentinel marking the sync graph stream is exhausted


def _chunk_text(chunk) -> str:
    """Extract plain text from a LangChain message chunk (content may be str or blocks)."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # some providers return a list of content blocks
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return ""


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream the answer token-by-token via SSE, plus progress and timing events.

    Event types (one JSON object per `data:` line):
      - {"type":"progress","node":...}   after each graph node completes
      - {"type":"ttft","seconds":...}    at the first answer token (time-to-first-token)
      - {"type":"token","text":...}      each answer token as it is generated
      - {"type":"final","route","response","ttft","total"}   once the graph finishes

    The sync graph generator is pumped from a worker thread (one `next()` per step) so it
    never blocks the event loop. RAGAS scoring + logging happen in a BackgroundTask after
    the stream closes, so they add nothing to TTFT.
    """
    # `outcome` is filled by the generator during streaming and read by the background task.
    outcome: dict = {}

    async def event_generator():
        t0 = time.perf_counter()
        ttft = None
        route = None
        final_response = None
        raw_output = None
        grading_passed = None
        retry_count = None

        state = await asyncio.to_thread(_initial_state, request.query, request.session_id)
        stream = graph.stream(state, stream_mode=["updates", "messages"])

        # Pump the synchronous generator one step at a time in a thread (non-blocking).
        while True:
            item = await asyncio.to_thread(next, stream, _STREAM_END)
            if item is _STREAM_END:
                break
            mode, payload = item
            if mode == "updates":
                for node_name, node_state in payload.items():
                    yield f"data: {json.dumps({'type': 'progress', 'node': node_name}, ensure_ascii=False)}\n\n"
                    if isinstance(node_state, dict):
                        route = node_state.get("route") or route
                        final_response = node_state.get("final_response") or final_response
                        raw_output = node_state.get("raw_tool_output") or raw_output
                        if node_state.get("grading_passed") is not None:
                            grading_passed = node_state["grading_passed"]
                        if node_state.get("retry_count") is not None:
                            retry_count = node_state["retry_count"]
            else:  # mode == "messages"
                chunk, meta = payload
                if meta.get("langgraph_node") not in ANSWER_NODES:
                    continue
                text = _chunk_text(chunk)
                if not text:
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - t0
                    yield f"data: {json.dumps({'type': 'ttft', 'seconds': round(ttft, 3)}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'text': text}, ensure_ascii=False)}\n\n"

        total = time.perf_counter() - t0

        # Persist the conversation turn in the foreground (memory = correctness).
        await asyncio.to_thread(_persist_turn, request.session_id, request.query, final_response, route)

        # Hand the metrics to the background task attached to the response.
        outcome.update({
            "session_id": request.session_id,
            "query": request.query,
            "response": final_response,
            "context": raw_output or "",
            "route": route,
            "grading_passed": grading_passed,
            "retry_count": retry_count,
            "ttft_seconds": round(ttft, 3) if ttft is not None else None,
            "total_seconds": round(total, 3),
        })

        yield f"data: {json.dumps({'type': 'final', 'route': route, 'response': final_response, 'ttft': round(ttft, 3) if ttft is not None else None, 'total': round(total, 3)}, ensure_ascii=False)}\n\n"

    async def _log_after_stream():
        if not outcome:
            return
        await evaluate_and_log(
            session_id=outcome["session_id"],
            query=outcome["query"],
            response=outcome["response"],
            context=outcome["context"],
            route=outcome["route"],
            grading_passed=outcome["grading_passed"],
            retry_count=outcome["retry_count"],
            ttft_seconds=outcome["ttft_seconds"],
            total_seconds=outcome["total_seconds"],
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        background=BackgroundTask(_log_after_stream),
    )


@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    """Persist user feedback (Upvote/Downvote + optional correction) to the feedback memory."""
    result = await asyncio.to_thread(
        save_user_feedback.invoke,
        {
            "query": request.query,
            "response": request.response,
            "is_positive": request.is_positive,
            "correction": request.correction,
        },
    )
    return {"message": result}


@app.get("/health")
async def health():
    return {"status": "ok"}
