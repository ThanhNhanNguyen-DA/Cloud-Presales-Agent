"""
Streamlit chat UI for the Cloud Presales Agentic RAG Chatbot.

Talks to the FastAPI backend. Each answer carries a live RAGAS scorecard (shown right below
it). Shows Upvote/Downvote buttons after each answer; a Downvote opens a correction box that
feeds the feedback memory, closing the self-learning loop.
"""

import json
import os
import uuid

import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
# Key sent as X-API-Key. Defaults to the first key in API_KEYS ("label:key" or bare key).
_first_key = (os.getenv("API_KEYS", "").split(",")[0].strip()).rpartition(":")[2]
API_KEY = os.getenv("PRESALES_API_KEY") or _first_key
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}


def _error_text(resp: requests.Response) -> str:
    """Human-readable message for a failed API call."""
    if resp.status_code == 401:
        return "The backend rejected the API key (set PRESALES_API_KEY or API_KEYS in .env)."
    if resp.status_code == 429:
        return f"Too many requests — please wait {resp.headers.get('Retry-After', 'a few')} s and retry."
    if resp.status_code == 422:
        return "The request was rejected (the question may be empty or too long — max 2000 characters)."
    try:
        return f"Backend error {resp.status_code}: {resp.json().get('detail', resp.text)}"
    except ValueError:
        return f"Backend error {resp.status_code}."


def send_feedback(payload: dict) -> tuple[bool, str]:
    """POST /feedback; returns (ok, message to show)."""
    try:
        resp = requests.post(f"{API_BASE_URL}/feedback", json=payload, headers=HEADERS, timeout=30)
    except requests.exceptions.RequestException as e:
        return False, f"Error connecting to backend: {e}"
    if not resp.ok:
        return False, _error_text(resp)
    return True, resp.json().get("message", "Feedback recorded.")

# Live (reference-free) RAGAS metrics shown in the per-answer scorecard.
SCORECARD_METRICS = [
    ("faithfulness", "Faithfulness", "Answer grounded in the context (hallucination check)."),
    ("answer_relevancy", "Answer Relevancy", "How directly the answer addresses the question."),
]

st.set_page_config(page_title="Cloud Presales Agentic RAG Chatbot", page_icon="☁️", layout="centered")
st.title("☁️ Cloud Presales Agentic RAG Chatbot")
st.caption("Presales support system: cost/sizing calculation, SLA/security document lookup, competitor service comparison.")


def render_scorecard(ragas: dict) -> None:
    """Show the live RAGAS scorecard (3 reference-free metrics) below an answer."""
    if not ragas:
        return
    st.caption("🎯 RAGAS scorecard (live · reference-free)")
    cols = st.columns(len(SCORECARD_METRICS))
    for col, (key, label, help_) in zip(cols, SCORECARD_METRICS, strict=True):
        val = ragas.get(key)
        col.metric(label, f"{val:.2f}" if isinstance(val, (int, float)) else "—", help=help_)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dict: {role, content, route, ragas, feedback_given}

# Conversation id sent with every request so the backend can load/save multi-turn memory.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Render chat history
for idx, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("ttft") is not None:
            st.caption(f"⚡ TTFT {msg['ttft']}s · ⏱ Total {msg.get('total', '?')}s · route: {msg.get('route', '?')}")
        if msg["role"] == "assistant":
            render_scorecard(msg.get("ragas"))
        if msg["role"] == "assistant" and msg.get("is_error"):
            pass  # no feedback buttons on an error message
        elif msg["role"] == "assistant" and not msg.get("feedback_given"):
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("👍 Correct", key=f"upvote_{idx}"):
                    ok, text = send_feedback({
                        "query": msg["query"],
                        "response": msg["content"],
                        "is_positive": True,
                        "correction": None
                    })
                    if ok:
                        st.session_state.chat_history[idx]["feedback_given"] = True
                        st.rerun()
                    st.error(text)
            with col2:
                if st.button("👎 Wrong / Needs fixing", key=f"downvote_{idx}"):
                    st.session_state[f"show_correction_{idx}"] = True

            if st.session_state.get(f"show_correction_{idx}"):
                correction_text = st.text_area(
                    "Enter the correct information for the system to remember:",
                    key=f"correction_input_{idx}"
                )
                if st.button("Submit correction", key=f"submit_correction_{idx}"):
                    ok, text = send_feedback({
                        "query": msg["query"],
                        "response": msg["content"],
                        "is_positive": False,
                        "correction": correction_text
                    })
                    if ok:
                        st.session_state.chat_history[idx]["feedback_given"] = True
                        st.session_state.chat_history[idx]["feedback_message"] = text
                        st.rerun()
                    st.error(text)
        elif msg["role"] == "assistant" and msg.get("feedback_given"):
            st.caption("✅ " + msg.get("feedback_message", "Feedback recorded for this answer."))

# New question input
user_input = st.chat_input("Enter your question (e.g. 'How long does a 10GB backup over 100Mbps take?')")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    meta = {}  # filled from the "ttft" / "ragas" / "final" / "error" events

    def stream_answer(placeholder) -> str:
        """
        Render answer tokens from the SSE endpoint into `placeholder`; return the text shown.

        A "reset" event means a new answer message started (e.g. the fact-checker asked
        for a retry), so the text shown so far is discarded — the user sees one answer.
        """
        shown = ""
        with requests.post(f"{API_BASE_URL}/chat/stream",
                           json={"query": user_input, "session_id": st.session_state.session_id},
                           headers=HEADERS, stream=True, timeout=180) as resp:
            if not resp.ok:
                meta["error"] = _error_text(resp)
                return ""
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                event = json.loads(line[len("data:"):].strip())
                etype = event.get("type")
                if etype == "token":
                    shown += event["text"]
                    placeholder.markdown(shown + "▌")
                elif etype == "reset":
                    shown = ""
                    placeholder.markdown("_Re-checking the answer…_")
                elif etype == "ttft":
                    meta["ttft"] = event["seconds"]
                elif etype == "ragas":
                    meta["ragas"] = {k: event.get(k) for k, _, _ in SCORECARD_METRICS}
                elif etype == "error":
                    meta["error"] = event.get("message", "The assistant failed to answer.")
                elif etype == "final":
                    meta["route"] = event.get("route")
                    meta["total"] = event.get("total")
                    meta["ttft"] = event.get("ttft", meta.get("ttft"))
                    meta["final"] = event.get("response")
                    meta["ragas"] = event.get("ragas", meta.get("ragas"))
        return shown

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            streamed_text = stream_answer(placeholder)
        except requests.exceptions.RequestException as e:
            meta = {"route": "error", "error": f"Error connecting to backend: {str(e)}"}
            streamed_text = ""
        # The authoritative text (may carry the fact-check warning added after streaming).
        placeholder.markdown(meta.get("error") or meta.get("final") or streamed_text)
        render_scorecard(meta.get("ragas"))

    # Prefer the authoritative final_response (the grader may adjust it post-stream).
    final_text = meta.get("error") or meta.get("final") or streamed_text

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": final_text,
        "query": user_input,
        "route": meta.get("route"),
        "ttft": meta.get("ttft"),
        "total": meta.get("total"),
        "ragas": meta.get("ragas"),
        "feedback_given": False,
        "is_error": bool(meta.get("error")),
    })
    st.rerun()

with st.sidebar:
    st.subheader("System info")
    st.write("Backend API:", API_BASE_URL)
    st.write("Session:", st.session_state.session_id[:8])
    st.write("Conversation turns:", len([m for m in st.session_state.chat_history if m["role"] == "user"]))
    if st.button("Clear chat history"):
        # Start a brand-new conversation so the backend memory is not carried over.
        st.session_state.chat_history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
