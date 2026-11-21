"""Reviewed canned answers: strict matching, hidden delivery, never for follow-ups, fail-open."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app import canned
from app.main import app
from tests.conftest import TEST_API_KEY

client = TestClient(app, headers={"X-API-Key": TEST_API_KEY})
ROW = {"id": "faq-1", "question": "ELB hỗ trợ những thuật toán cân bằng tải nào?",
       "answer": "Round Robin, Least Connections, Source IP.", "route": "document", "similarity": 0.97}


@pytest.mark.parametrize("asked, stored, ok", [
    ("Làm sao tạo Auto Scaling Group?", "Cách tạo Auto Scaling Group như thế nào?", True),   # unaccented "sao"
    ("KMS dùng để làm gì?", "Key Management Service dùng để làm gì?", True),                 # abbreviation
    ("ELB của AWS hỗ trợ thuật toán nào?", "ELB hỗ trợ những thuật toán cân bằng tải nào?", False),
    ("RDS for PostgreSQL hỗ trợ gì", "RDS for MySQL hỗ trợ những thao tác quản lý nào?", False),
    ("Backup 10GB mất bao lâu", "Backup 100GB mất bao lâu", False),
])
def test_identifiers_must_match(asked, stored, ok):
    assert canned.identifiers_match(asked, stored) is ok


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setenv("CANNED_ANSWERS_ENABLED", "true")
    sb = MagicMock()
    monkeypatch.setattr(canned, "supabase", sb)
    monkeypatch.setattr(canned, "embed_query", lambda q: [0.0] * 512)
    return sb


def test_find_answer_uses_threshold_and_skips_identifier_mismatches(db):
    db.rpc.return_value.execute.return_value = MagicMock(data=[
        {**ROW, "id": "wrong", "question": "VPN site-to-site cấu hình thế nào?"}, ROW])
    assert canned.find_answer("ELB có thuật toán cân bằng tải nào?")["id"] == "faq-1"
    match_call, hit_call = db.rpc.call_args_list
    assert match_call.args[1]["match_threshold"] == canned.MIN_SIMILARITY
    assert hit_call.args == ("record_canned_hit", {"answer_id": "faq-1"})


def test_find_answer_fails_open(db):
    db.rpc.side_effect = RuntimeError("db down")
    assert canned.find_answer("ELB?") is None


def _ask(path, history=None):
    g = MagicMock()
    g.ainvoke = AsyncMock(return_value={"route": "general", "final_response": "hi"})
    g.astream = MagicMock()
    with patch("app.main.graph", g), patch("app.main.canned.find_answer", return_value=ROW) as find, \
         patch("app.main.evaluate_answer_async", new=AsyncMock(return_value={})), \
         patch("app.main.load_conversation_history", return_value=history or []):
        resp = client.post(path, json={"query": "ELB có thuật toán nào?"})
    return resp, g, find


@pytest.mark.parametrize("path", ["/chat", "/chat/stream"])
def test_canned_answer_is_delivered_like_a_normal_answer(path):
    resp, g, _ = _ask(path)
    g.ainvoke.assert_not_called()
    g.astream.assert_not_called()
    if path == "/chat":
        assert resp.json() == {"route": "document", "response": ROW["answer"], "ragas": None}
    else:
        events = [json.loads(l[5:]) for l in resp.text.splitlines() if l.startswith("data:")]
        assert events[-1]["response"] == ROW["answer"] and "cached" not in events[-1]
        assert "canned" not in resp.text and "faq-1" not in resp.text     # hidden from the user


def test_follow_ups_never_use_canned_answers():
    _, g, find = _ask("/chat", history=[MagicMock()])
    find.assert_not_called()
