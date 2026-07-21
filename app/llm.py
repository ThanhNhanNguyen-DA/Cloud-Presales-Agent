"""
Groq (Llama 3.3 70B) chat model with automatic fallback on rate-limit exhaustion.

`ChatGroq` raises a 429 rate-limit error when a model's per-minute / per-day quota
runs out. This wrapper holds an *ordered* list of Groq models and, on every call,
tries them in turn: the primary model (Llama 3.3 70B) normally answers, but when it
is rate-limited the call transparently falls through to the next model in the list.

By default it **fails fast**: the Groq client's own retry loop is disabled
(`max_retries=0`) so a 429 surfaces immediately and the switch to the next model is
instant — otherwise the client would retry several times with backoff before the
fallback even runs. It can still optionally retry the *same* model with backoff
(off by default) for transient per-minute limits before moving on.

Exposes the small slice of the LangChain chat-model interface the app uses —
`.invoke()` and `.bind_tools()` — so it is a drop-in replacement in the supervisor,
agents, and grader. Embeddings are NOT handled here: Groq has no embeddings API, so
vectorisation stays on the local Qwen model (see app/tools/embedding_tool.py).
"""

from __future__ import annotations

import re
import time
from typing import Any, List, Optional, Sequence

from langchain_groq import ChatGroq

try:  # Groq's typed rate-limit error, if the installed client exposes it.
    from groq import RateLimitError
    _RATE_LIMIT_TYPES: tuple = (RateLimitError,)
except Exception:  # pragma: no cover - the exception module layout varies by version
    _RATE_LIMIT_TYPES = ()

# langchain_groq sometimes wraps the underlying error, so we also sniff the message.
_RATE_LIMIT_MARKERS = (
    "429", "rate limit", "rate-limit", "ratelimit",
    "too many requests", "quota", "rate_limit_exceeded",
)


def _is_rate_limit_error(exc: Exception) -> bool:
    """True if `exc` looks like a Groq quota / rate-limit (429) error."""
    if _RATE_LIMIT_TYPES and isinstance(exc, _RATE_LIMIT_TYPES):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def _retry_after_seconds(exc: Exception, default: float) -> float:
    """
    How long to wait before retrying a rate-limited call.

    Prefers the server's instruction: the `retry-after` response header, else the
    "Please try again in 2s" hint embedded in Groq's 429 message. Falls back to `default`.
    """
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            header = response.headers.get("retry-after")
            if header:
                return float(header)
        except Exception:  # noqa: BLE001 - header shape varies; fall through to message parsing
            pass
    match = re.search(r"try again in ([\d.]+)\s*s", str(exc), re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return default


class FallbackChatLLM:
    """
    A chat model that fans out across `models` (Groq) when one is rate-limited.

    On each `.invoke()`, models are tried in order. For a given model the call is
    retried `retries_per_model` times with linear backoff (to ride out a transient
    per-minute limit); once those are exhausted the next model is tried. If EVERY model
    is rate-limited in one pass, it waits (honouring the server's "retry-after") and
    retries the whole chain up to `rate_limit_max_retries` times before finally raising a
    RuntimeError — so a sustained per-minute limit (common on the free tier, and during
    offline batch evaluation) rides out instead of crashing. Any non-rate-limit error is
    re-raised immediately (we don't mask real bugs as a quota problem).
    """

    def __init__(
        self,
        models: Sequence[str],
        *,
        api_key: str,
        temperature: float = 0.0,
        retries_per_model: int = 1,
        backoff_seconds: float = 0.0,
        rate_limit_max_retries: int = 4,
        rate_limit_wait_cap: float = 30.0,
        underlying_max_retries: int = 0,
        request_timeout: Optional[float] = None,
        _tools: Optional[Any] = None,
        **model_kwargs: Any,
    ) -> None:
        if not models:
            raise ValueError("FallbackChatLLM needs at least one model name.")
        self._models = list(models)
        self._api_key = api_key
        self._temperature = temperature
        self._retries_per_model = max(1, retries_per_model)
        self._backoff_seconds = backoff_seconds
        # When all models are rate-limited, wait-and-retry the whole chain this many times.
        self._rate_limit_max_retries = max(0, rate_limit_max_retries)
        self._rate_limit_wait_cap = rate_limit_wait_cap

        # Fail fast so the fallback is instant. The Groq client's OWN retry loop is
        # what makes switching feel slow — disable it here so a 429 raises at once
        # and we move to the next model. Optionally cap the per-call wait. Both are
        # overridable via kwargs.
        model_kwargs.setdefault("max_retries", underlying_max_retries)
        if request_timeout is not None:
            model_kwargs.setdefault("request_timeout", request_timeout)
        self._model_kwargs = model_kwargs
        self._tools = _tools

        self._llms: List[Any] = []
        for name in self._models:
            llm: Any = ChatGroq(
                model=name,
                api_key=api_key,
                temperature=temperature,
                **model_kwargs,
            )
            if _tools is not None:
                llm = llm.bind_tools(_tools)
            self._llms.append(llm)

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FallbackChatLLM":
        """Return a new fallback model with `tools` bound to every underlying model."""
        return FallbackChatLLM(
            self._models,
            api_key=self._api_key,
            temperature=self._temperature,
            retries_per_model=self._retries_per_model,
            backoff_seconds=self._backoff_seconds,
            rate_limit_max_retries=self._rate_limit_max_retries,
            rate_limit_wait_cap=self._rate_limit_wait_cap,
            _tools=tools,
            **self._model_kwargs,
        )

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """
        Try each model in order; fall through to the next on a rate-limit error. If every
        model is rate-limited in one pass, wait (honouring retry-after) and retry the whole
        chain, up to `rate_limit_max_retries` times, before raising.
        """
        last_error: Optional[Exception] = None

        for outer in range(self._rate_limit_max_retries + 1):
            for index, llm in enumerate(self._llms):
                model_name = self._models[index]
                for attempt in range(self._retries_per_model):
                    try:
                        return llm.invoke(input, config=config, **kwargs)
                    except Exception as exc:  # noqa: BLE001 - non-rate-limit errors re-raised below
                        if not _is_rate_limit_error(exc):
                            raise
                        last_error = exc
                        is_last_attempt = attempt == self._retries_per_model - 1
                        if not is_last_attempt:
                            time.sleep(self._backoff_seconds * (attempt + 1))
                            print(
                                f"[llm] '{model_name}' rate-limited; "
                                f"retry {attempt + 2}/{self._retries_per_model}…"
                            )
                        elif index < len(self._llms) - 1:
                            print(
                                f"[llm] '{model_name}' rate-limited; "
                                f"switching to '{self._models[index + 1]}'."
                            )

            # Every model was rate-limited this pass. Wait it out and retry the chain.
            if outer < self._rate_limit_max_retries:
                default_wait = min(2.0 * (2 ** outer), self._rate_limit_wait_cap)
                wait = min(_retry_after_seconds(last_error, default_wait) + 0.5,
                           self._rate_limit_wait_cap)
                print(
                    f"[llm] all models rate-limited; waiting {wait:.1f}s then retrying "
                    f"(pass {outer + 2}/{self._rate_limit_max_retries + 1})…"
                )
                time.sleep(wait)

        raise RuntimeError(
            "All Groq models are rate-limited after "
            f"{self._rate_limit_max_retries + 1} passes: " + ", ".join(self._models)
        ) from last_error


def get_chat_llm(
    temperature: float = 0.0,
    models: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> FallbackChatLLM:
    """
    Build the app's standard fallback-enabled Groq chat model.

    Uses the ordered `GROQ_MODELS` chain from config unless `models` is given.
    Rate-limit wait-and-retry is tunable via env: GROQ_RATE_LIMIT_RETRIES (default 4),
    GROQ_RATE_LIMIT_WAIT_CAP seconds (default 30).
    """
    import os

    from app.config import GROQ_API_KEY, GROQ_MODELS

    kwargs.setdefault("rate_limit_max_retries", int(os.getenv("GROQ_RATE_LIMIT_RETRIES", "4")))
    kwargs.setdefault("rate_limit_wait_cap", float(os.getenv("GROQ_RATE_LIMIT_WAIT_CAP", "30")))

    return FallbackChatLLM(
        models or GROQ_MODELS,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        **kwargs,
    )
