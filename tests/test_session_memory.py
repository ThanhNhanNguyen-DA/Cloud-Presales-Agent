"""Conversation memory survives a page reload: the session id lives in the URL and history is restored."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from app.main import app
from tests.conftest import TEST_API_KEY

UI_DIR = Path(__file__).resolve().parents[1] / "ui"
sys.path.insert(0, str(UI_DIR))
client = TestClient(app, headers={"X-API-Key": TEST_API_KEY})


def test_history_endpoint_is_scoped_to_the_caller():
    msgs = [HumanMessage("Backup 10GB?"), AIMessage("16,7 phút")]
    with patch("app.main.load_conversation_history", return_value=msgs) as load:
        resp = client.get("/history", params={"session_id": "abc-1"})
    assert resp.json() == {"messages": [{"role": "user", "content": "Backup 10GB?"},
                                        {"role": "assistant", "content": "16,7 phút"}]}
    assert load.call_args.args[0] == "tester:abc-1"          # namespaced by API key


def test_history_endpoint_validates_input():
    assert client.get("/history", params={"session_id": "../etc"}).status_code == 422
    assert TestClient(app).get("/history", params={"session_id": "a"}).status_code == 401


def _get(url, *args, **kwargs):
    resp = MagicMock(ok=True)
    if url.endswith("/history"):
        resp.json.return_value = {"messages": [{"role": "user", "content": "Câu hỏi cũ"},
                                               {"role": "assistant", "content": "Trả lời cũ"}]}
    else:
        resp.json.return_value = {"status": "ok", "database": "ok"}
    return resp


def _page():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(UI_DIR / "streamlit_app.py"), default_timeout=30)


def test_reload_with_sid_restores_the_conversation():
    at = _page()
    at.query_params["sid"] = "keep-me-123"
    with patch("requests.get", side_effect=_get):
        at.run()
    assert not at.exception
    assert at.session_state["session_id"] == "keep-me-123"
    history = at.session_state["chat_history"]
    assert [m["content"] for m in history] == ["Câu hỏi cũ", "Trả lời cũ"]
    assert history[1]["restored"] and history[1]["query"] == "Câu hỏi cũ"


def test_first_visit_writes_a_new_sid_to_the_url():
    at = _page()
    with patch("requests.get", side_effect=_get):
        at.run()
    sid = at.session_state["session_id"]
    assert at.query_params["sid"] in (sid, [sid])
