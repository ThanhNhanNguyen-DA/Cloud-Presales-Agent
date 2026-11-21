"""
Async FastAPI backend wrapping the LangGraph Cloud Presales agent.

Exposes:
- POST /chat         -> full (non-streaming) response
- POST /chat/stream  -> Server-Sent Events: step-by-step progress then the final answer
- POST /feedback     -> records an Upvote/Downvote (+ optional correction) for the feedback memory
- GET  /health       -> liveness check

Async design (pragmatic):
- Endpoints are `async def`; the graph runs via `ainvoke` / `astream`. Its nodes are sync
  (LLM, Supabase and embedding calls); LangGraph runs them in worker threads, so the event
  loop is never blocked.
- Blocking helpers (conversation load/save, feedback) are offloaded with `asyncio.to_thread`.
- Each answer is scored LIVE with two reference-free RAGAS metrics (faithfulness, answer
  relevancy), computed AFTER the answer is produced (so it never delays time-to-first-token)
  and returned with it. The other metrics stay in the offline batch job
  (evals/eval_ragas.py).

Observability: every request gets an id (X-Request-ID), every log line carries it, and each
answered question logs one summary line (route, latency, LLM calls, tokens) — see
app/observability.py.
"""

import asyncio
import json
import logging
import os
import string
import time
from contextlib import asynccontextmanager

from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import cache, canned, security
from app.agents.tool_agent import message_text
from app.config import validate_config
from app.db.supabase_client import supabase
from app.graph.builder import build_graph
from app.observability import current_metrics, setup_logging, start_request
from app.progress import describe_step
from app.tools.memory_tools import list_feedback, review_feedback, store_feedback
from app.tools.embedding_tool import get_embedding_model
from app.tools.reranker import get_reranker, rerank_enabled
from app.tools.conversation_tools import load_conversation_history, save_conversation_turn
from app.evaluation.metrics import evaluate_answer_async

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Settings are checked when the server starts, not at import (keeps tests env-free).
    setup_logging()
    validate_config()
    security.settings.validate()
    if os.getenv("WARMUP_EMBEDDINGS", "true").strip().lower() not in {"0", "false", "no"}:
        # Load the embedding model now (~15 s) instead of on the first user request.
        await asyncio.to_thread(get_embedding_model)
        if rerank_enabled():
            await asyncio.to_thread(get_reranker)
    yield


app = FastAPI(title="Cloud Presales Agentic RAG Chatbot API", lifespan=lifespan)

_REQUEST_ID_OK = set(string.ascii_letters + string.digits + "-_.")


@app.middleware("http")
async def request_context(request: Request, call_next):
    """
    Give every request an id (the caller's X-Request-ID if it looks sane, else a new one),
    bind it + fresh metrics to the context for logging, and echo it in the response.
    """
    incoming = request.headers.get("x-request-id", "")
    valid = 0 < len(incoming) <= 64 and set(incoming) <= _REQUEST_ID_OK
    metrics = start_request(incoming if valid else None)
    response = await call_next(request)
    response.headers["X-Request-ID"] = metrics.request_id
    if request.url.path != "/health":
        # For /chat/stream this is time-to-headers; the full summary is logged at stream end.
        logger.info("%s %s -> %d in %.3fs", request.method, request.url.path,
                    response.status_code, metrics.elapsed())
    return response


def _log_summary(endpoint: str, route: str | None, **extra) -> None:
    """One line per answered question: route, latency, LLM calls and tokens."""
    metrics = current_metrics()
    stats = metrics.summary() if metrics else {}
    fields = " ".join(f"{k}={v}" for k, v in {**stats, **extra}.items())
    logger.info("%s done route=%s %s", endpoint, route, fields)


app.add_middleware(
    CORSMiddleware,
    allow_origins=security.settings.cors_origins,  # CORS_ORIGINS in .env
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

graph = build_graph()


# Input limits: a question is short text; anything bigger is abuse or a mistake and only
# burns LLM tokens. session_id is an opaque client-chosen id (the UI sends a UUID).
MAX_QUERY_CHARS = 2000
MAX_RESPONSE_CHARS = 20000
MAX_CORRECTION_CHARS = 2000
SESSION_ID_PATTERN = r"^[A-Za-z0-9_.:-]{1,100}$"


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    # Groups turns of one conversation; None = stateless.
    session_id: str | None = Field(default=None, pattern=SESSION_ID_PATTERN)


class ChatResponse(BaseModel):
    route: str
    response: str
    # Live (reference-free) RAGAS scorecard, scored after the answer and returned with it:
    # {"context_precision": float|None, "faithfulness": float|None, "answer_relevancy": float|None}.
    ragas: dict | None = None


class FeedbackRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    response: str = Field(max_length=MAX_RESPONSE_CHARS)
    is_positive: bool
    correction: str | None = Field(default=None, max_length=MAX_CORRECTION_CHARS)


class ReviewRequest(BaseModel):
    decision: Literal["approve", "reject"]


def _scoped_session(caller: str, session_id: str | None) -> str | None:
    """Namespace the conversation by API key, so one client can't read another's history."""
    return f"{caller}:{session_id}" if session_id else None


def _initial_state(query: str, session_id: str | None = None) -> dict:
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


def _persist_turn(session_id: str | None, query: str, response: str | None, route: str | None) -> None:
    """Append the user question and the assistant answer to the conversation memory (no-op if stateless)."""
    if not session_id:
        return
    save_conversation_turn(session_id, "user", query)
    save_conversation_turn(session_id, "assistant", response or "", route=route)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, caller: str = Depends(security.chat_caller)):
    """Run the full graph, score the answer live (RAGAS), and return both together."""
    session_id = _scoped_session(caller, request.session_id)
    state = await asyncio.to_thread(_initial_state, request.query, session_id)
    fresh = not state["messages"]  # no history: the answer doesn't depend on context
    hit = await asyncio.to_thread(cache.get_answer, request.query) if fresh else None
    if hit:
        await asyncio.to_thread(_persist_turn, session_id, request.query, hit["response"], hit["route"])
        _log_summary("chat", hit["route"], cache="hit")
        return ChatResponse(route=hit["route"] or "general", response=hit["response"], ragas=hit["ragas"])
    faq = await asyncio.to_thread(canned.find_answer, request.query) if fresh else None
    if faq:
        await asyncio.to_thread(_persist_turn, session_id, request.query, faq["answer"], faq["route"])
        _log_summary("chat", faq["route"], canned=faq["id"])
        return ChatResponse(route=faq["route"], response=faq["answer"], ragas=None)
    try:
        result = await graph.ainvoke(state)
    except Exception as exc:  # noqa: BLE001 - e.g. every Groq model unavailable
        logger.exception("Graph failed: %s", exc)
        raise HTTPException(
            status_code=502, detail="The assistant failed to answer. Please try again."
        ) from None

    # Save the conversation turn so a fast follow-up sees it.
    await asyncio.to_thread(
        _persist_turn, session_id, request.query,
        result.get("final_response"), result.get("route"),
    )

    # Live RAGAS: scored after the answer is ready, returned with it.
    ragas = await evaluate_answer_async(
        request.query,
        result.get("final_response") or "",
        result.get("raw_tool_output") or "",
    )
    if fresh:
        await asyncio.to_thread(cache.put_answer, request.query, result, ragas)
    _log_summary("chat", result.get("route"), graded=result.get("grading_passed"),
                 retries=result.get("retry_count") or 0)
    return ChatResponse(
        route=result.get("route") or "general",
        response=result.get("final_response") or "",
        ragas=ragas,
    )


@app.get("/history")
async def history(
    session_id: str = Query(pattern=SESSION_ID_PATTERN),
    limit: int = Query(default=50, ge=1, le=200),
    caller: str = Depends(security.require_api_key),
):
    """Messages of one conversation (oldest first), so a reloaded UI can show them again."""
    messages = await asyncio.to_thread(
        load_conversation_history, _scoped_session(caller, session_id), (limit + 1) // 2)
    return {"messages": [{"role": "user" if m.type == "human" else "assistant", "content": m.content}
                         for m in messages][-limit:]}


# Nodes that produce the user-facing answer. We stream tokens ONLY from these — the
# supervisor (intent classification) and grader also call the LLM, but their tokens are
# internal and must not leak into the answer.
ANSWER_NODES = {"calculator", "document", "general"}



def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, caller: str = Depends(security.chat_caller)):
    """
    Stream the answer token by token via SSE. A specialist's draft streams live; if the
    grader asks for a retry, a `reset` event discards it, and `final` carries the
    authoritative text (with any verification warning). A repeated question without
    conversation history is answered from the Redis cache (progress node "cache").

    Event types (one JSON object per `data:` line):
      - {"type":"progress","node":...,"detail":{...}}   after each graph node completes;
                                         `detail` = plain facts for the UI log (app/progress.py);
                                         node "tool_call" = a tool is starting ({"tool", "agent"})
      - {"type":"ttft","seconds":...}    at the first answer token (time-to-first-token)
      - {"type":"token","text":...}      answer tokens as they are generated
      - {"type":"reset"}                 discard the tokens shown so far: a NEW answer message
                                         starts (the grader asked for a retry, or the agent
                                         wrote text before calling a tool)
      - {"type":"error","message":...}   the graph failed; no `final` follows
      - {"type":"ragas", ...metrics}     the live scorecard, once the answer is complete
      - {"type":"final","route","response","ttft","total","ragas"}   once the graph finishes

    The graph is consumed with `graph.astream`; LangGraph runs the sync nodes in worker
    threads, so the event loop is never blocked. RAGAS is scored AFTER the tokens are sent, so it never
    delays TTFT — the scorecard simply appears right after the answer.
    """

    session_id = _scoped_session(caller, request.session_id)

    async def event_generator():
        t0 = time.perf_counter()
        ttft = None
        route = None
        final_response = None
        raw_output = None
        current_message = None  # identifies the LLM message whose tokens are on screen
        grader_runs = 0
        streamed = False
        grading_passed = None
        fresh = False

        try:
            state = await asyncio.to_thread(_initial_state, request.query, session_id)
            fresh = not state["messages"]
            hit = await asyncio.to_thread(cache.get_answer, request.query) if fresh else None
            if hit:
                yield _sse({"type": "progress", "node": "cache", "detail": {"hit": True}})
                ttft = time.perf_counter() - t0
                yield _sse({"type": "ttft", "seconds": round(ttft, 3)})
                yield _sse({"type": "token", "text": hit["response"]})
                await asyncio.to_thread(_persist_turn, session_id, request.query, hit["response"], hit["route"])
                _log_summary("chat/stream", hit["route"], cache="hit")
                yield _sse({"type": "ragas", **(hit["ragas"] or {})})
                yield _sse({"type": "final", "route": hit["route"], "response": hit["response"],
                            "ttft": round(ttft, 3), "total": round(time.perf_counter() - t0, 3),
                            "ragas": hit["ragas"], "cached": True})
                return
            faq = await asyncio.to_thread(canned.find_answer, request.query) if fresh else None
            if faq:  # reviewed canned answer: delivered like any answer, no LLM call
                yield _sse({"type": "progress", "node": "supervisor", "detail": {"route": faq["route"]}})
                ttft = time.perf_counter() - t0
                yield _sse({"type": "ttft", "seconds": round(ttft, 3)})
                yield _sse({"type": "token", "text": faq["answer"]})
                await asyncio.to_thread(_persist_turn, session_id, request.query, faq["answer"], faq["route"])
                _log_summary("chat/stream", faq["route"], canned=faq["id"])
                yield _sse({"type": "final", "route": faq["route"], "response": faq["answer"],
                            "ttft": round(ttft, 3), "total": round(time.perf_counter() - t0, 3), "ragas": None})
                return
            # LangGraph runs the (sync) nodes in worker threads and streams their updates
            # and LLM tokens here, without blocking the event loop.
            async for item in graph.astream(state, stream_mode=["updates", "messages"]):
                mode, payload = item
                if mode == "updates":
                    for node_name, node_state in payload.items():
                        grader_runs += node_name == "grader"
                        detail = describe_step(node_name, node_state, raw_output, attempt=grader_runs or 1)
                        yield _sse({"type": "progress", "node": node_name, "detail": detail})
                        if isinstance(node_state, dict):
                            # Key presence, not truthiness: a retry legitimately resets
                            # raw_tool_output, and the grader may rewrite final_response.
                            route = node_state.get("route") or route
                            if "final_response" in node_state:
                                final_response = node_state["final_response"]
                            if "raw_tool_output" in node_state:
                                raw_output = node_state["raw_tool_output"]
                            if "grading_passed" in node_state:
                                grading_passed = node_state["grading_passed"]
                else:  # mode == "messages"
                    chunk, meta = payload
                    node = meta.get("langgraph_node")
                    if node not in ANSWER_NODES:
                        continue
                    # Announce a tool as soon as the model starts calling it (the agent's own
                    # progress event only arrives when the whole node has finished).
                    for call in getattr(chunk, "tool_call_chunks", None) or []:
                        if call.get("name"):
                            yield _sse({"type": "progress", "node": "tool_call",
                                        "detail": {"tool": call["name"], "agent": node}})
                    text = message_text(chunk)
                    if not text:
                        continue
                    message_key = getattr(chunk, "id", None) or meta.get("langgraph_step")
                    if current_message is not None and message_key != current_message:
                        yield _sse({"type": "reset"})
                    current_message = message_key
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                        yield _sse({"type": "ttft", "seconds": round(ttft, 3)})
                    streamed = True
                    yield _sse({"type": "token", "text": text})
        except Exception as exc:  # noqa: BLE001 - report instead of cutting the stream silently
            logger.exception("Graph failed while streaming: %s", exc)
            yield _sse({"type": "error", "message": "The assistant failed to answer. Please try again."})
            return

        if not streamed and final_response:
            # Text that never came from a streamed LLM message (e.g. the "could not complete" fallback).
            ttft = time.perf_counter() - t0
            yield _sse({"type": "ttft", "seconds": round(ttft, 3)})
            yield _sse({"type": "token", "text": final_response})

        # Persist the conversation turn so a fast follow-up sees it.
        await asyncio.to_thread(_persist_turn, session_id, request.query, final_response, route)

        # Live RAGAS: scored after the tokens (never delays TTFT), emitted then echoed in `final`.
        ragas = await evaluate_answer_async(request.query, final_response or "", raw_output or "")
        yield _sse({"type": "ragas", **ragas})
        if fresh:
            await asyncio.to_thread(cache.put_answer, request.query,
                                    {"route": route, "final_response": final_response,
                                     "raw_tool_output": raw_output, "grading_passed": grading_passed}, ragas)

        _log_summary("chat/stream", route, ttft_s=round(ttft, 3) if ttft is not None else None)
        yield _sse({
            "type": "final",
            "route": route,
            "response": final_response,
            "ttft": round(ttft, 3) if ttft is not None else None,
            "total": round(time.perf_counter() - t0, 3),
            "ragas": ragas,
        })

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/feedback")
async def feedback(request: FeedbackRequest, caller: str = Depends(security.feedback_caller)):
    """
    Persist user feedback (Upvote/Downvote + optional correction) to the feedback memory.

    A correction is stored as *pending* and only influences answers after an admin approves
    it via /admin/feedback/{id}/review. The caller's key label is recorded as the submitter.
    """
    try:
        message = await asyncio.to_thread(
            store_feedback,
            request.query,
            request.response,
            request.is_positive,
            request.correction,
            caller,
        )
    except Exception as exc:  # noqa: BLE001 - DB/embedding failure; don't leak internals
        logger.exception("Could not save feedback: %s", exc)
        raise HTTPException(status_code=502, detail="Could not save feedback. Please try again.") from None
    return {"message": message}


_DB_ERROR = "Database error (has sql/migrations/002_security.sql been run?)."


@app.get("/admin/feedback")
async def admin_list_feedback(
    status: Literal["pending", "approved", "rejected"] = "pending",
    limit: int = 50,
    _: str = Depends(security.require_admin_key),
):
    """List feedback awaiting review (or already reviewed), newest first."""
    try:
        rows = await asyncio.to_thread(list_feedback, status, max(1, min(limit, 200)))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not list feedback: %s", exc)
        raise HTTPException(status_code=502, detail=_DB_ERROR) from None
    return {"items": rows}


@app.post("/admin/feedback/{feedback_id}/review")
async def admin_review_feedback(
    body: ReviewRequest,
    feedback_id: str = Path(pattern=r"^[0-9a-fA-F-]{36}$"),
    reviewer: str = Depends(security.require_admin_key),
):
    """Approve (the correction starts being used) or reject one feedback row."""
    try:
        row = await asyncio.to_thread(review_feedback, feedback_id, body.decision == "approve", reviewer)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not review feedback %s: %s", feedback_id, exc)
        raise HTTPException(status_code=502, detail=_DB_ERROR) from None
    if row is None:
        raise HTTPException(status_code=404, detail="Feedback not found.")
    if body.decision == "approve":
        await asyncio.to_thread(cache.clear)  # an approved correction can change cached answers
    return {"id": row.get("id"), "review_status": row.get("review_status")}


HEALTH_DB_TIMEOUT_S = 3.0


def _ping_database() -> None:
    supabase.table("cloud_documents").select("id").limit(1).execute()


@app.get("/health")
async def health():
    """
    Liveness plus a database check. Always 200 while the API process is up (a liveness probe
    must not restart the API because Supabase is down); `database` says whether lookups can
    work, so the UI can warn instead of answers silently turning into "no information".
    """
    try:
        await asyncio.wait_for(asyncio.to_thread(_ping_database), timeout=HEALTH_DB_TIMEOUT_S)
        database = "ok"
    except Exception as exc:  # noqa: BLE001 - any failure means lookups will fail
        logger.warning("Health check: database unreachable: %s", exc)
        database = "unreachable"
    return {"status": "ok" if database == "ok" else "degraded", "database": database}
