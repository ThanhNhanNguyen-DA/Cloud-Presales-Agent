"""Streamlit UI: activity-log wording (pure functions) and an AppTest smoke test of the page."""

import json
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

UI_DIR = Path(__file__).resolve().parents[1] / "ui"
sys.path.insert(0, str(UI_DIR))

from activity_log import entries_for_event, format_entry, route_badge  # noqa: E402


# --------------------------------------------------------------------------- wording
def _texts(event):
    return [e["text"] for e in entries_for_event(event, 1.23)]


def test_supervisor_step_is_explained_in_plain_words():
    assert _texts({"type": "progress", "node": "supervisor", "detail": {"route": "calculator"}}) == \
        ["Đã hiểu yêu cầu: **Tính toán chi phí / cấu hình**"]
    out = _texts({"type": "progress", "node": "supervisor",
                  "detail": {"route": "document", "used_corrections": True}})
    assert len(out) == 2 and "góp ý" in out[1]


def test_tools_are_named_for_humans_not_by_function_name():
    out = _texts({"type": "progress", "node": "competitor",
                  "detail": {"tools": ["find_competitor_mapping", "list_all_competitor_services",
                                       "search_competitor_docs"], "found_data": False}})
    assert out[0] == "Đã dùng: bảng đối chiếu dịch vụ AWS/Azure/GCP, dữ liệu website đối thủ"
    assert "Không tìm thấy" in out[1]
    assert "_" not in " ".join(out)


def test_agent_without_tools_means_it_will_ask_back():
    entry = entries_for_event({"type": "progress", "node": "calculator", "detail": {"tools": []}}, 0)[0]
    assert entry["level"] == "warn" and "hỏi lại" in entry["text"]


@pytest.mark.parametrize("verdict, level", [("pass", "ok"), ("retry", "warn"), ("warn", "warn"),
                                            ("unverified", "warn"), ("skipped", "info")])
def test_grader_verdicts(verdict, level):
    entry = entries_for_event({"type": "progress", "node": "grader", "detail": {"verdict": verdict}}, 0)[0]
    assert entry["level"] == level


def test_quality_score_and_finish_line():
    assert _texts({"type": "ragas", "faithfulness": 0.9, "answer_relevancy": 0.756}) == \
        ["Chấm điểm chất lượng: bám sát dữ liệu 90%, đúng trọng tâm 76%"]
    assert _texts({"type": "ragas", "faithfulness": None}) == []
    assert _texts({"type": "final", "total": 9.55}) == ["Hoàn tất sau 9,6 giây"]
    assert _texts({"type": "token", "text": "x"}) == []


def test_tool_is_announced_early_and_not_repeated():
    ctx = {}
    early = entries_for_event({"type": "progress", "node": "tool_call",
                               "detail": {"tool": "calculate_backup_and_cost", "agent": "calculator"}}, 1, ctx)
    assert early[0]["text"] == "Đang dùng: máy tính thời gian sao lưu & chi phí lưu trữ"
    again = entries_for_event({"type": "progress", "node": "tool_call",
                               "detail": {"tool": "calculate_backup_and_cost"}}, 2, ctx)
    done = entries_for_event({"type": "progress", "node": "calculator",
                              "detail": {"tools": ["calculate_backup_and_cost"], "found_data": True}}, 3, ctx)
    assert again == [] and done == []


def test_small_talk_gets_no_quality_score():
    ctx = {}
    entries_for_event({"type": "progress", "node": "supervisor", "detail": {"route": "general"}}, 0, ctx)
    assert entries_for_event({"type": "ragas", "answer_relevancy": 0.37}, 1, ctx) == []


def test_format_and_badge():
    line = format_entry({"t": 1.2, "icon": "🧭", "text": "abc", "level": "info"})
    assert "+1,2s" in line and "abc" in line
    assert route_badge("competitor") == "🆚 So sánh với đối thủ"
    assert route_badge(None).startswith("❔")


# --------------------------------------------------------------------------- page smoke test
def _sse(*events):
    return [f"data: {json.dumps(e, ensure_ascii=False)}" for e in events]


STREAM = _sse(
    {"type": "progress", "node": "supervisor", "detail": {"route": "calculator", "used_corrections": False}},
    {"type": "progress", "node": "calculator",
     "detail": {"tools": ["calculate_backup_and_cost"], "found_data": True}},
    {"type": "ttft", "seconds": 1.1},
    {"type": "token", "text": "Mất khoảng "},
    {"type": "token", "text": "16,7 phút."},
    {"type": "progress", "node": "grader", "detail": {"verdict": "pass", "attempt": 1}},
    {"type": "ragas", "faithfulness": 1.0, "answer_relevancy": 0.8},
    {"type": "final", "route": "calculator", "response": "Mất khoảng 16,7 phút.", "total": 4.2,
     "ragas": {"faithfulness": 1.0, "answer_relevancy": 0.8}},
)


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(UI_DIR / "streamlit_app.py"), default_timeout=30)


def _fake_post(lines):
    resp = MagicMock(ok=True)
    resp.iter_lines.return_value = iter(lines)
    resp.__enter__.return_value = resp
    return MagicMock(return_value=resp)


def test_page_renders_without_backend():
    import requests
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("offline")), \
         warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        at = _app().run()
    assert not at.exception
    assert any("Không kết nối" in e.value for e in at.error)
    assert any("Nhật ký hệ thống" in m.value for m in at.markdown)


def test_asking_a_question_fills_chat_and_activity_log():
    with patch("requests.get", return_value=MagicMock(ok=True)), \
         patch("requests.post", _fake_post(STREAM)):
        at = _app().run()
        at.chat_input[0].set_value("Backup 10GB qua 100Mbps?").run()
    assert not at.exception
    assert at.session_state["chat_history"][-1]["content"] == "Mất khoảng 16,7 phút."
    log = at.session_state["activity"][-1]["entries"]
    texts = " | ".join(e["text"] for e in log)
    for expected in ("Đã nhận câu hỏi", "Tính toán chi phí", "máy tính thời gian sao lưu",
                     "Đang soạn câu trả lời", "số liệu khớp", "bám sát dữ liệu 100%", "Hoàn tất sau 4,2 giây"):
        assert expected in texts
    page = " ".join(m.value for m in at.markdown)
    assert "số liệu khớp" in page and "Mất khoảng 16,7 phút." in page


def test_backend_error_is_shown_in_plain_words():
    with patch("requests.get", return_value=MagicMock(ok=True)), \
         patch("requests.post", _fake_post(_sse({"type": "error", "message": "x"}))):
        at = _app().run()
        at.chat_input[0].set_value("hi").run()
    last = at.session_state["chat_history"][-1]
    assert last["is_error"] and "chưa trả lời được" in last["content"]
    assert at.session_state["activity"][-1]["entries"][-1]["level"] == "error"
