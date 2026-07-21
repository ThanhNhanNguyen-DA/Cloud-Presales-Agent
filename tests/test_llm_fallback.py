"""
Tests for FallbackChatLLM's rate-limit handling: model fallback + wait-and-retry.

Underlying ChatGroq construction is patched out, and the per-model `_llms` are replaced
with fakes, so nothing touches the network. `time.sleep` is patched so waits are instant.
"""

from unittest.mock import patch

from app.llm import FallbackChatLLM, _retry_after_seconds


def _make_llm(models, **kwargs):
    with patch("app.llm.ChatGroq"):  # avoid constructing real Groq clients
        return FallbackChatLLM(models, api_key="test-key", **kwargs)


class _Fake:
    """A stand-in chat model: raises the given exceptions in turn, then returns `final`."""
    def __init__(self, errors=(), final="ok"):
        self._errors = list(errors)
        self._final = final
        self.calls = 0

    def invoke(self, *args, **kwargs):
        self.calls += 1
        if self._errors:
            raise self._errors.pop(0)
        return self._final


def test_retry_after_parses_message():
    assert _retry_after_seconds(Exception("Please try again in 2.5s. Upgrade..."), 99) == 2.5
    assert _retry_after_seconds(Exception("no hint here"), 7.0) == 7.0
    print("✓ retry-after parsed from the 429 message")


def test_falls_back_to_second_model_on_rate_limit():
    llm = _make_llm(["m1", "m2"])
    a = _Fake(errors=[Exception("429 rate limit exceeded")])
    b = _Fake(final="from-b")
    llm._llms = [a, b]
    with patch("app.llm.time.sleep"):
        assert llm.invoke("hi") == "from-b"
    print("✓ falls back to the second model when the first is rate-limited")


def test_waits_and_retries_chain_then_succeeds():
    """A single model rate-limited twice: the chain waits and retries, then succeeds."""
    llm = _make_llm(["m1"], rate_limit_max_retries=3)
    fake = _Fake(errors=[Exception("429, please try again in 1s"),
                         Exception("429, please try again in 1s")], final="done")
    llm._llms = [fake]
    with patch("app.llm.time.sleep") as sleep:
        assert llm.invoke("hi") == "done"
    assert fake.calls == 3           # failed twice, succeeded on the third pass
    assert sleep.call_count == 2     # waited between the three passes
    print("✓ waits out a sustained rate limit and retries the chain")


def test_raises_runtime_error_after_all_passes():
    llm = _make_llm(["m1"], rate_limit_max_retries=1)
    llm._llms = [_Fake(errors=[Exception("rate_limit_exceeded")] * 5)]
    with patch("app.llm.time.sleep"):
        try:
            llm.invoke("hi")
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "rate-limited" in str(e).lower()
    print("✓ raises RuntimeError once the retry budget is exhausted")


def test_non_rate_limit_error_is_reraised_immediately():
    llm = _make_llm(["m1", "m2"], rate_limit_max_retries=3)
    llm._llms = [_Fake(errors=[ValueError("a real bug")]), _Fake(final="unused")]
    with patch("app.llm.time.sleep") as sleep:
        try:
            llm.invoke("hi")
            assert False, "expected ValueError"
        except ValueError:
            pass
    sleep.assert_not_called()  # real bugs are not masked as rate limits
    print("✓ non-rate-limit errors propagate immediately")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
