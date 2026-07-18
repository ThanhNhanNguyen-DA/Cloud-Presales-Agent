"""
FastAPI backend (Phase 9) wrapping the LangGraph Cloud Presales agent.

Exposes:
- POST /chat         -> full (non-streaming) response
- POST /chat/stream  -> Server-Sent Events: step-by-step progress then the final answer
- POST /feedback     -> records an Upvote/Downvote (+ optional correction) for the feedback memory
- GET  /health       -> liveness check
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import time

from app.graph.builder import build_graph
from app.tools.memory_tools import save_user_feedback
from app.tools.conversation_tools import load_conversation_history, save_conversation_turn
from app.evaluation.ragas_live import evaluate_answer

app = FastAPI(title="Cloud Presales Agentic RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in this phase for easy local Streamlit testing; restrict before deploying.
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
    ragas: Optional[dict] = None  # live RAGAS metrics: faithfulness, answer_relevancy


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
    call is stateless (empty history) — unchanged from before multi-turn memory.
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
def chat(request: ChatRequest):
    """Run the full graph and return the final answer in one shot."""
    result = graph.invoke(_initial_state(request.query, request.session_id))
    ragas = evaluate_answer(
        request.query,
        result.get("final_response") or "",
        result.get("raw_tool_output") or "",
    )
    _persist_turn(request.session_id, request.query, result.get("final_response"), result.get("route"))
    return ChatResponse(route=result["route"], response=result["final_response"], ragas=ragas)


# Nodes that produce the user-facing answer. We stream tokens ONLY from these — the
# supervisor (intent classification) and grader also call the LLM, but their tokens are
# internal and must not leak into the answer.
ANSWER_NODES = {"calculator", "document", "competitor", "general"}


def _chunk_text(chunk) -> str:
    """Extract plain text from a LangChain message chunk (content may be str or blocks)."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # some providers return a list of content blocks
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return ""


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """
    Stream the answer token-by-token via SSE, plus progress and timing events.

    Event types (one JSON object per `data:` line):
      - {"type":"progress","node":...}            after each graph node completes
      - {"type":"ttft","seconds":...}             at the first answer token (time-to-first-token)
      - {"type":"token","text":...}               each answer token as it is generated
      - {"type":"final","route","response",       once the graph finishes, with the
                 "ttft","total"}                   authoritative answer and timings (seconds)
    """
    def event_generator():
        t0 = time.perf_counter()
        ttft = None
        route = None
        final_response = None
        raw_output = None

        # stream_mode=["updates","messages"] interleaves node-completion updates with
        # per-token LLM message chunks. Each item is (mode, payload).
        for mode, payload in graph.stream(_initial_state(request.query, request.session_id),
                                          stream_mode=["updates", "messages"]):
            if mode == "updates":
                for node_name, node_state in payload.items():
                    yield f"data: {json.dumps({'type': 'progress', 'node': node_name}, ensure_ascii=False)}\n\n"
                    if isinstance(node_state, dict):
                        route = node_state.get("route") or route
                        final_response = node_state.get("final_response") or final_response
                        raw_output = node_state.get("raw_tool_output") or raw_output
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

        # Persist the completed turn to the conversation memory (no-op if stateless).
        _persist_turn(request.session_id, request.query, final_response, route)

        # Live RAGAS: computed after the answer is complete, so it never delays TTFT.
        ragas = evaluate_answer(request.query, final_response or "", raw_output or "")
        yield f"data: {json.dumps({'type': 'ragas', **ragas}, ensure_ascii=False)}\n\n"

        final_msg = {
            "type": "final",
            "route": route,
            "response": final_response,
            "ttft": round(ttft, 3) if ttft is not None else None,
            "total": round(time.perf_counter() - t0, 3),
            "ragas": ragas,
        }
        yield f"data: {json.dumps(final_msg, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    """Persist user feedback (Upvote/Downvote + optional correction) to the feedback memory."""
    result = save_user_feedback.invoke({
        "query": request.query,
        "response": request.response,
        "is_positive": request.is_positive,
        "correction": request.correction,
    })
    return {"message": result}


@app.get("/health")
def health():
    return {"status": "ok"}
