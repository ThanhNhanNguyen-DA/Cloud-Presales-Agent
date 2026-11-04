"""
Logging and per-request metrics.

- `setup_logging()` configures the root logger once (level from LOG_LEVEL, default INFO).
  Every record carries the current request id, so all lines of one request can be grepped
  together: `... [req=3f2a9c1b] app.llm: openai/gpt-oss-120b ok 1.84s tokens=812/96`.
- `start_request()` opens a RequestMetrics for the current request in a ContextVar.
  `asyncio.to_thread` and LangGraph's worker threads copy the context, so the LLM wrapper
  (app/llm.py) can add its call count / latency / token usage from any thread, and the API
  logs one summary line per request (route, latency, LLM calls, tokens).
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field

_request_id: ContextVar[str] = ContextVar("request_id", default="-")
_metrics: ContextVar[RequestMetrics | None] = ContextVar("request_metrics", default=None)


@dataclass
class RequestMetrics:
    request_id: str
    started: float = field(default_factory=time.perf_counter)
    llm_calls: int = 0
    llm_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_llm_call(self, seconds: float, input_tokens: int = 0, output_tokens: int = 0) -> None:
        with self._lock:  # LLM calls may come from several worker threads
            self.llm_calls += 1
            self.llm_seconds += seconds
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def summary(self) -> dict:
        return {
            "latency_s": round(self.elapsed(), 3),
            "llm_calls": self.llm_calls,
            "llm_s": round(self.llm_seconds, 3),
            "tokens_in": self.input_tokens,
            "tokens_out": self.output_tokens,
        }


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


_configured = False


def setup_logging() -> None:
    """Configure root logging once; safe to call repeatedly."""
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler()
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s [req=%(request_id)s] %(name)s: %(message)s"
    ))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    # Third-party HTTP clients are chatty at INFO (one line per Supabase/Groq request).
    for noisy in ("httpx", "httpx2", "httpcore", "hpack", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def start_request(request_id: str | None = None) -> RequestMetrics:
    """Bind a request id + fresh metrics to the current context."""
    rid = (request_id or uuid.uuid4().hex)[:32]
    metrics = RequestMetrics(request_id=rid)
    _request_id.set(rid)
    _metrics.set(metrics)
    return metrics


def current_metrics() -> RequestMetrics | None:
    return _metrics.get()


def current_request_id() -> str:
    return _request_id.get()
