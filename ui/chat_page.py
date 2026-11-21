"""
Chat page of the Streamlit UI (entry point: ui/streamlit_app.py).

Layout: chat on the left, a live "Nhật ký hệ thống" (system activity log) on the right that
explains, in plain Vietnamese, what the assistant is doing for each question (understanding
the request, which tool it used, the automatic number check, the quality score, the time).

Talks to the FastAPI backend (/chat/stream SSE, /feedback, /health). A downvote opens a
correction box; corrections are applied after a reviewer approves them.
"""

import json
import os
import time
import uuid
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv

from activity_log import entries_for_event, format_entry, grading_status, route_badge
from answer_rendering import format_answer_markdown

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
# Key sent as X-API-Key. Defaults to the first key in API_KEYS ("label:key" or bare key).
_first_key = (os.getenv("API_KEYS", "").split(",")[0].strip()).rpartition(":")[2]
API_KEY = os.getenv("PRESALES_API_KEY") or _first_key
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

SCORECARD = [
    ("faithfulness", "Bám sát dữ liệu", "Câu trả lời dựa đúng vào dữ liệu tra cứu được (không bịa)."),
    ("answer_relevancy", "Đúng trọng tâm", "Câu trả lời đi thẳng vào câu hỏi."),
]

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; }
      .log-line { padding: 4px 8px; margin: 2px 0; border-left: 3px solid rgba(128,128,128,.35);
                  border-radius: 4px; font-size: 0.9rem; line-height: 1.35; }
      .log-ok    { border-left-color: #2e9e5b; }
      .log-warn  { border-left-color: #d99a00; }
      .log-error { border-left-color: #d64545; }
      .log-time  { font-family: ui-monospace, monospace; font-size: 0.75rem; opacity: .6; }
      .log-q     { font-weight: 600; margin: 4px 0 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- backend helpers
def _error_text(resp: requests.Response) -> str:
    """Plain-language message for a failed API call."""
    if resp.status_code == 401:
        return "Hệ thống từ chối khóa truy cập (kiểm tra PRESALES_API_KEY hoặc API_KEYS trong .env)."
    if resp.status_code == 429:
        return f"Bạn gửi hơi nhanh — vui lòng đợi {resp.headers.get('Retry-After', 'vài')} giây rồi thử lại."
    if resp.status_code == 422:
        return "Câu hỏi không hợp lệ (có thể đang trống hoặc dài quá 2000 ký tự)."
    return f"Hệ thống đang gặp lỗi (mã {resp.status_code}). Vui lòng thử lại sau."


@st.cache_data(ttl=15, show_spinner=False)
def backend_health() -> dict | None:
    """The backend's /health body ({"status", "database"}), or None if the API is unreachable."""
    try:
        # Above the backend's own 3 s database-check timeout, so a slow DB is not read as "API down".
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
    except requests.exceptions.RequestException:
        return None
    if not resp.ok:
        return None
    try:
        return resp.json()
    except ValueError:
        return {}


def send_feedback(payload: dict) -> tuple[bool, str]:
    """POST /feedback; returns (ok, message to show)."""
    try:
        resp = requests.post(f"{API_BASE_URL}/feedback", json=payload, headers=HEADERS, timeout=30)
    except requests.exceptions.RequestException:
        return False, "Không kết nối được tới hệ thống."
    if not resp.ok:
        return False, _error_text(resp)
    return True, "Đã ghi nhận góp ý. Cảm ơn bạn!" if payload["is_positive"] else (
        "Đã ghi nhận chỉnh sửa — sẽ được áp dụng sau khi người duyệt xác nhận.")


# --------------------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("chat_history", [])  # {role, content, query, route, total, ragas, feedback_given, is_error}
ss.setdefault("activity", [])      # {question, time, entries}
ss.setdefault("session_id", str(uuid.uuid4()))


def new_conversation() -> None:
    ss.chat_history, ss.activity, ss.session_id = [], [], str(uuid.uuid4())


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("☁️ Trợ lý Presales")
    health = backend_health()
    if health is None:
        st.error("Không kết nối được tới hệ thống", icon="🔴")
        st.caption(f"Kiểm tra backend tại {API_BASE_URL}")
    elif health.get("database") == "unreachable":
        st.warning("Kho dữ liệu (Supabase) không truy cập được: các câu hỏi cần tra cứu tài liệu "
                   "sẽ báo lỗi. Tính toán chi phí / cấu hình vẫn dùng được.", icon="🟠")
    else:
        st.success("Hệ thống sẵn sàng", icon="🟢")

    st.divider()
    st.button("🆕 Cuộc trò chuyện mới", on_click=new_conversation, width="stretch")
    asked = sum(m["role"] == "user" for m in ss.chat_history)
    st.caption(f"Đã hỏi {asked} câu trong cuộc trò chuyện này.")

    with st.expander("ℹ️ Cách đọc nhật ký hệ thống"):
        st.markdown(
            "- 🧭 **Đã hiểu yêu cầu** — hệ thống xếp câu hỏi vào một nhóm chuyên môn.\n"
            "- 🔧 **Đã dùng** — công cụ tính toán hoặc nguồn dữ liệu được tra cứu.\n"
            "- ✅ **Đã kiểm tra** — số liệu trong câu trả lời được đối chiếu tự động với dữ liệu gốc.\n"
            "- ⚠️ — có điểm cần bạn kiểm tra lại trước khi gửi cho khách hàng.\n"
            "- 📊 **Chấm điểm** — mức độ bám sát dữ liệu và đúng trọng tâm (càng gần 100% càng tốt)."
        )


# --------------------------------------------------------------------------- layout
st.title("☁️ Trợ lý Cloud Presales")
st.caption("Tính chi phí & cấu hình · tra cứu tài liệu SLA / bảo mật / tính năng")

chat_col, log_col = st.columns([0.64, 0.36], gap="large")

with log_col:
    st.markdown("#### 🧾 Nhật ký hệ thống")
    log_area = st.empty()


def render_log(live: dict | None = None) -> None:
    """Draw the activity panel: the live (or latest) question first, earlier ones folded."""
    items = ([live] if live else []) + list(reversed(ss.activity))
    with log_area.container(height=640, border=True):
        if not items:
            st.caption("Nhật ký sẽ hiện ở đây khi bạn đặt câu hỏi: hệ thống hiểu yêu cầu thế nào, "
                       "dùng công cụ gì, kiểm tra số liệu ra sao và mất bao lâu.")
            return
        for i, item in enumerate(items):
            lines = "".join(
                f"<div class='log-line log-{e['level']}'>{format_entry(e)}</div>" for e in item["entries"]
            ) or "<div class='log-line'>⏳ Đang xử lý…</div>"
            header = f"🕑 {item['time']} — {item['question'][:60]}{'…' if len(item['question']) > 60 else ''}"
            if i == 0:
                st.markdown(f"<div class='log-q'>{header}</div>{lines}", unsafe_allow_html=True)
            else:
                with st.expander(header):
                    st.markdown(lines, unsafe_allow_html=True)


def render_scorecard(ragas: dict | None, route: str | None = None) -> None:
    if not ragas or route == "general":  # small talk: nothing to score against
        return
    parts = [f"{label} **{ragas[key]:.0%}**" for key, label, _ in SCORECARD
             if isinstance(ragas.get(key), (int, float))]
    if parts:
        st.caption("📊 " + " · ".join(parts),
                   help=" ".join(f"{label}: {help_}" for _, label, help_ in SCORECARD))


render_log()

with chat_col:
    if not ss.chat_history:
        st.info("👋 Xin chào! Hãy đặt câu hỏi ở ô bên dưới.")

    for idx, msg in enumerate(ss.chat_history):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(format_answer_markdown(msg["content"]), unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])
            if msg["role"] != "assistant":
                continue
            if not msg.get("is_error"):
                meta = [route_badge(msg.get("route"))]
                if isinstance(msg.get("total"), (int, float)):
                    meta.append(f"⏱ {msg['total']:.1f} giây".replace(".", ","))
                status = grading_status(msg.get("grading"), msg.get("found_data"), msg.get("tool_failed", False))
                if status:
                    label, icon, color = status
                    st.badge(label, icon=icon, color=color)
                st.caption(" · ".join(meta))
                render_scorecard(msg.get("ragas"), msg.get("route"))

            if msg.get("is_error"):
                continue
            if msg.get("feedback_given"):
                st.caption("✅ " + msg.get("feedback_message", "Đã ghi nhận góp ý."))
                continue
            up, down, _ = st.columns([1, 1.4, 3])
            if up.button("👍 Chính xác", key=f"up_{idx}"):
                ok, text = send_feedback({"query": msg["query"], "response": msg["content"],
                                          "is_positive": True, "correction": None})
                if ok:
                    msg.update(feedback_given=True, feedback_message=text)
                    st.rerun()
                st.error(text)
            if down.button("👎 Chưa đúng / cần sửa", key=f"down_{idx}"):
                ss[f"show_fix_{idx}"] = True
            if ss.get(f"show_fix_{idx}"):
                fix = st.text_area("Thông tin đúng là gì? Hệ thống sẽ ghi nhớ sau khi được duyệt.",
                                   key=f"fix_{idx}")
                if st.button("Gửi chỉnh sửa", key=f"send_fix_{idx}", disabled=not fix.strip()):
                    ok, text = send_feedback({"query": msg["query"], "response": msg["content"],
                                              "is_positive": False, "correction": fix})
                    if ok:
                        msg.update(feedback_given=True, feedback_message=text)
                        st.rerun()
                    st.error(text)

question = st.chat_input("Nhập câu hỏi…")

if question:
    ss.chat_history.append({"role": "user", "content": question})
    live = {"question": question, "time": datetime.now().strftime("%H:%M:%S"),
            "entries": [{"t": 0.0, "icon": "📨", "text": "Đã nhận câu hỏi", "level": "info"}]}
    render_log(live)
    meta: dict = {}

    with chat_col:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            answer_box = st.empty()
            answer_box.markdown("⏳ _Đang xử lý…_")
            t0 = time.perf_counter()
            shown = ""
            log_ctx: dict = {}  # route + tools already announced, for the activity log
            grading: list[str] = []  # grader verdicts in order: "retry" then "pass" = self-corrected
            try:
                with requests.post(f"{API_BASE_URL}/chat/stream",
                                   json={"query": question, "session_id": ss.session_id},
                                   headers=HEADERS, stream=True, timeout=180) as resp:
                    if not resp.ok:
                        meta["error"] = _error_text(resp)
                    else:
                        for line in resp.iter_lines(decode_unicode=True):
                            if not line or not line.startswith("data:"):
                                continue
                            event = json.loads(line[5:].strip())
                            etype = event.get("type")
                            detail = event.get("detail") or {}
                            if etype == "progress" and event.get("node") == "grader":
                                grading.append(detail.get("verdict"))
                            elif etype == "progress" and "found_data" in detail:
                                meta["found_data"] = detail["found_data"]  # the last specialist run wins
                                meta["tool_failed"] = detail.get("tool_failed", False)
                            new_entries = entries_for_event(event, time.perf_counter() - t0, log_ctx)
                            if new_entries:
                                live["entries"] += new_entries
                                render_log(live)
                            if etype == "token":
                                shown += event["text"]
                                answer_box.markdown(format_answer_markdown(shown + "▌"), unsafe_allow_html=True)
                            elif etype == "reset":
                                shown = ""
                                answer_box.markdown("🔁 _Đang soạn lại câu trả lời cho chính xác hơn…_")
                            elif etype == "error":
                                meta["error"] = "Xin lỗi, hệ thống chưa trả lời được. Vui lòng thử lại."
                            elif etype == "final":
                                meta.update(route=event.get("route"), total=event.get("total"),
                                            final=event.get("response"), ragas=event.get("ragas"))
            except requests.exceptions.RequestException:
                meta["error"] = "Không kết nối được tới hệ thống. Vui lòng kiểm tra backend."

            if meta.get("error") and not any(e["level"] == "error" for e in live["entries"]):
                live["entries"].append({"t": round(time.perf_counter() - t0, 1), "icon": "❌",
                                        "text": meta["error"], "level": "error"})
            # The final text is authoritative: the fact-checker may have added a warning.
            answer_box.markdown(format_answer_markdown(meta.get("error") or meta.get("final") or shown),
                                unsafe_allow_html=True)

    ss.activity.append(live)
    ss.chat_history.append({
        "role": "assistant",
        "content": meta.get("error") or meta.get("final") or shown,
        "query": question,
        "route": meta.get("route"),
        "total": meta.get("total"),
        "ragas": meta.get("ragas"),
        "grading": grading,
        "found_data": meta.get("found_data"),
        "tool_failed": meta.get("tool_failed", False),
        "feedback_given": False,
        "is_error": bool(meta.get("error")),
    })
    st.rerun()
