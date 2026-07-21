"""
Streamlit chat UI for the Cloud Presales Agentic RAG Chatbot.

Talks to the FastAPI backend. Shows Upvote/Downvote buttons after each answer; a Downvote
opens a correction box that feeds the feedback memory, closing the self-learning loop.
"""

import json
import uuid

import streamlit as st
import requests

API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Cloud Presales Agentic RAG Chatbot", page_icon="☁️", layout="centered")
st.title("☁️ Cloud Presales Agentic RAG Chatbot")
st.caption("Presales support system: cost/sizing calculation, SLA/security document lookup, competitor service comparison.")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dict: {role, content, route, feedback_given}

# Conversation id sent with every request so the backend can load/save multi-turn memory.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Render chat history
for idx, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("ttft") is not None:
            st.caption(f"⚡ TTFT {msg['ttft']}s · ⏱ Total {msg.get('total', '?')}s · route: {msg.get('route', '?')}")
        if msg["role"] == "assistant" and not msg.get("feedback_given"):
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("👍 Correct", key=f"upvote_{idx}"):
                    requests.post(f"{API_BASE_URL}/feedback", json={
                        "query": msg["query"],
                        "response": msg["content"],
                        "is_positive": True,
                        "correction": None
                    })
                    st.session_state.chat_history[idx]["feedback_given"] = True
                    st.rerun()
            with col2:
                if st.button("👎 Wrong / Needs fixing", key=f"downvote_{idx}"):
                    st.session_state[f"show_correction_{idx}"] = True

            if st.session_state.get(f"show_correction_{idx}"):
                correction_text = st.text_area(
                    "Enter the correct information for the system to remember:",
                    key=f"correction_input_{idx}"
                )
                if st.button("Submit correction", key=f"submit_correction_{idx}"):
                    requests.post(f"{API_BASE_URL}/feedback", json={
                        "query": msg["query"],
                        "response": msg["content"],
                        "is_positive": False,
                        "correction": correction_text
                    })
                    st.session_state.chat_history[idx]["feedback_given"] = True
                    st.success("Recorded! The system will apply this correction to similar future questions.")
                    st.rerun()
        elif msg["role"] == "assistant" and msg.get("feedback_given"):
            st.caption("✅ Feedback recorded for this answer.")

# New question input
user_input = st.chat_input("Enter your question (e.g. 'How long does a 10GB backup over 100Mbps take?')")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    meta = {}  # filled by the generator from "ttft"/"final" events

    def stream_answer():
        """Yield answer tokens from the SSE endpoint; capture timing/route in `meta`."""
        with requests.post(f"{API_BASE_URL}/chat/stream",
                           json={"query": user_input, "session_id": st.session_state.session_id},
                           stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                event = json.loads(line[len("data:"):].strip())
                etype = event.get("type")
                if etype == "token":
                    yield event["text"]
                elif etype == "ttft":
                    meta["ttft"] = event["seconds"]
                elif etype == "final":
                    meta["route"] = event.get("route")
                    meta["total"] = event.get("total")
                    meta["ttft"] = event.get("ttft", meta.get("ttft"))
                    meta["final"] = event.get("response")

    with st.chat_message("assistant"):
        try:
            streamed_text = st.write_stream(stream_answer())
        except requests.exceptions.RequestException as e:
            streamed_text = f"Error connecting to backend: {str(e)}"
            meta = {"route": "error"}

    # Prefer the authoritative final_response (the grader may adjust it post-stream).
    final_text = meta.get("final") or streamed_text

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": final_text,
        "query": user_input,
        "route": meta.get("route"),
        "ttft": meta.get("ttft"),
        "total": meta.get("total"),
        "feedback_given": False
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
