"""Redis answer cache: repeated questions cost no LLM calls; context-dependent and unchecked answers are never cached."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app import cache
from app.main import app
from tests.conftest import TEST_API_KEY

client = TestClient(app, headers={"X-API-Key": TEST_API_KEY})
GOOD = {"route": "calculator", "final_response": "16,7 phút", "grading_passed": True, "raw_tool_output": "x"}


class FakeRedis:
    def __init__(self):
        self.data = {}

    def get(self, k):
        return self.data.get(k)

    def setex(self, k, ttl, v):
        self.data[k] = v

    def scan_iter(self, pattern, count=None):
        return [k for k in self.data if k.startswith(pattern.rstrip("*"))]

    def delete(self, *keys):
        for k in keys:
            self.data.pop(k, None)
        return len(keys)


@pytest.fixture
def redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(cache, "get_client", lambda: fake)
    return fake


def test_normalize_matches_trivial_variants():
    assert cache.normalize("  Backup 10GB   qua 100Mbps mất bao lâu??  ") == cache.normalize("backup 10gb qua 100mbps mất bao lâu")


def test_only_checked_answers_are_cached(redis):
    cache.put_answer("q1", GOOD, {"faithfulness": 1.0})
    cache.put_answer("q2", {**GOOD, "grading_passed": False}, None)          # warned / failed
    cache.put_answer("q3", {**GOOD, "final_response": ""}, None)             # empty
    cache.put_answer("q4", {"route": "general", "final_response": "Xin chào"}, None)  # no grading needed
    assert cache.get_answer("Q1?")["response"] == "16,7 phút"
    assert cache.get_answer("q2") is None and cache.get_answer("q3") is None
    assert cache.get_answer("q4")["route"] == "general"


def test_cache_off_and_redis_down_fail_open(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setattr(cache, "_client", None)
    assert cache.get_answer("q") is None and cache.clear() == 0
    broken = MagicMock()
    broken.get.side_effect = ConnectionError("down")
    monkeypatch.setattr(cache, "get_client", lambda: broken)
    assert cache.get_answer("q") is None


def _chat(path, history=None, graph_result=GOOD):
    g = MagicMock()
    g.ainvoke = AsyncMock(return_value=graph_result)

    async def astream(_state, **_kw):
        yield ("updates", {"calculator": {"final_response": graph_result["final_response"]}})
        yield ("updates", {"grader": {"route": graph_result["route"], "grading_passed": True}})
    g.astream = astream
    with patch("app.main.graph", g), \
         patch("app.main.evaluate_answer_async", new=AsyncMock(return_value={"faithfulness": 0.9})) as ragas, \
         patch("app.main.load_conversation_history", return_value=history or []):
        resp = client.post(path, json={"query": "Backup 10GB qua 100Mbps?"})
    return resp, g, ragas


@pytest.mark.parametrize("path", ["/chat", "/chat/stream"])
def test_second_identical_question_skips_the_graph(redis, path):
    _, g1, _ = _chat(path)
    resp, g2, ragas = _chat(path)
    assert resp.status_code == 200
    ragas.assert_not_called()                      # no judge call either
    if path == "/chat":
        g2.ainvoke.assert_not_called()
        assert resp.json()["response"] == "16,7 phút"
    else:
        events = [json.loads(l[5:]) for l in resp.text.splitlines() if l.startswith("data:")]
        assert events[0] == {"type": "progress", "node": "cache", "detail": {"hit": True}}
        assert events[-1]["cached"] is True and events[-1]["response"] == "16,7 phút"


def test_follow_ups_with_history_are_never_cached(redis):
    _chat("/chat", history=[MagicMock()])
    assert redis.data == {}


def test_approving_a_correction_clears_the_cache(redis):
    cache.put_answer("q1", GOOD, None)
    admin = TestClient(app, headers={"X-API-Key": "test-admin-key"})
    with patch("app.main.review_feedback", return_value={"id": "x", "review_status": "approved"}):
        admin.post("/admin/feedback/6f1c2a9e-0b7d-4c55-9a51-3c0e2d4b8f10/review", json={"decision": "approve"})
    assert redis.data == {}


def test_activity_log_explains_a_cache_hit():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
    from activity_log import entries_for_event
    entry = entries_for_event({"type": "progress", "node": "cache", "detail": {"hit": True}}, 0.1)[0]
    assert entry["level"] == "ok" and "không tốn lượt gọi AI" in entry["text"]
