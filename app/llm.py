"""
Groq chat model (GPT-OSS 120B primary) with automatic fallback on rate-limit exhaustion.

`ChatGroq` raises a 429 rate-limit error when a model's per-minute / per-day quota
runs out. This wrapper holds an *ordered* list of Groq models and, on every call,
tries them in turn: the primary model (GPT-OSS 120B) normally answers, but when it
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

import json
import logging
import re
import time
from collections.abc import Sequence
from typing import Any

from langchain_groq import ChatGroq

from app.observability import current_metrics

logger = logging.getLogger(__name__)

# Providers reachable through an OpenAI-compatible endpoint (via langchain-openai).
# A model is written "provider:model"; a bare name means Groq (backwards compatible).
OPENAI_COMPATIBLE_PROVIDERS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "hf": ("https://router.huggingface.co/v1", "HUGGINGFACEHUB_API_TOKEN"),
}
PROVIDERS = ("groq", *OPENAI_COMPATIBLE_PROVIDERS)


def parse_model_spec(spec: str) -> tuple[str, str]:
    """'gemini:gemini-3.5-flash-lite' -> ('gemini', ...); 'openai/gpt-oss-120b' -> ('groq', ...)."""
    provider, sep, model = spec.partition(":")
    if sep and provider in PROVIDERS:
        return provider, model
    return "groq", spec


def flatten_tool_history(messages: Any) -> Any:
    """
    Rewrite tool-call turns as plain text, for providers that can't replay them.

    Gemini 3 (OpenAI-compatible endpoint) rejects any earlier function call that lacks its
    `thought_signature` — and langchain-openai doesn't keep that signature, so even Gemini's
    own tool calls fail on the next turn (and Groq's always do when Gemini is a fallback).
    The model still sees what was called and what came back; it can call tools again.
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    if not isinstance(messages, list):
        return messages
    out = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            calls = "; ".join(f"{tc['name']}({json.dumps(tc.get('args') or {}, ensure_ascii=False)})"
                              for tc in m.tool_calls)
            text = m.content if isinstance(m.content, str) else ""
            out.append(AIMessage(content=f"{text}\n[Called tools: {calls}]".strip()))
        elif isinstance(m, ToolMessage):
            out.append(HumanMessage(content=f"[Tool result]\n{m.content}"))
        else:
            out.append(m)
    return out


def _build_chat_model(spec: str, *, groq_api_key: str | None, temperature: float, **kwargs: Any) -> Any:
    """Instantiate the LangChain chat model for one spec. Raises KeyError if its API key is missing."""
    import os

    provider, model = parse_model_spec(spec)
    if provider == "groq":
        return ChatGroq(model=model, api_key=groq_api_key, temperature=temperature, **kwargs)
    from langchain_openai import ChatOpenAI

    base_url, key_env = OPENAI_COMPATIBLE_PROVIDERS[provider]
    api_key = os.getenv(key_env)
    if not api_key:
        raise KeyError(key_env)
    if provider == "gemini":
        # Gemini "thinks" by default; for routing / tool calls / grading that only adds latency
        # (grader p50 13 s -> ~1 s). Override with model_kwargs if a role needs deep reasoning.
        kwargs.setdefault("reasoning_effort", "low")
    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=temperature, **kwargs)

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


_MODEL_UNAVAILABLE_MARKERS = ("model_not_found", "model_decommissioned", "does not exist", "decommissioned")


def _is_model_unavailable_error(exc: Exception) -> bool:
    """True if Groq says the model is gone (404 / decommissioned) — retrying it is pointless."""
    if getattr(exc, "status_code", None) == 404:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _MODEL_UNAVAILABLE_MARKERS)


def _is_bad_generation_error(exc: Exception) -> bool:
    """
    True for Groq's 400 `tool_use_failed`: the model emitted a malformed tool call (wrong tool
    name, arguments that don't match the schema). That is a one-off bad sample, not a bug in
    our request, so it is worth retrying — and then trying another model.
    """
    text = str(exc).lower()
    return "tool_use_failed" in text or "tool call validation failed" in text


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
    A chat model that fans out across `models` when one is rate-limited.

    Each entry is "provider:model" (groq, gemini, hf) or a bare Groq model name. Entries
    whose provider has no API key configured are skipped with a warning, as long as at
    least one model remains.

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
        api_key: str | None = None,
        temperature: float = 0.0,
        retries_per_model: int = 1,
        backoff_seconds: float = 0.0,
        rate_limit_max_retries: int = 4,
        rate_limit_wait_cap: float = 30.0,
        underlying_max_retries: int = 0,
        request_timeout: float | None = None,
        _tools: Any | None = None,
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

        self._llms: list[Any] = []
        usable: list[str] = []
        for name in self._models:
            try:
                llm: Any = _build_chat_model(name, groq_api_key=api_key, temperature=temperature,
                                             **model_kwargs)
            except KeyError as missing:
                logger.warning("Skipping model %r: %s is not set.", name, missing)
                continue
            if _tools is not None:
                llm = llm.bind_tools(_tools)
            self._llms.append(llm)
            usable.append(name)
        if not self._llms:
            raise ValueError(f"No usable model in {self._models}: API keys missing.")
        self._models = usable

    def bind_tools(self, tools: Any, **kwargs: Any) -> FallbackChatLLM:
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

    @staticmethod
    def _timed_invoke(llm: Any, model_name: str, input: Any, config: Any, kwargs: dict) -> Any:
        """One model call; logs latency + token usage and adds them to the request metrics."""
        if parse_model_spec(model_name)[0] == "gemini":
            input = flatten_tool_history(input)
        started = time.perf_counter()
        response = llm.invoke(input, config=config, **kwargs)
        seconds = time.perf_counter() - started
        usage = getattr(response, "usage_metadata", None)
        usage = usage if isinstance(usage, dict) else {}
        tokens_in, tokens_out = int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
        logger.info("%s ok %.2fs tokens=%d/%d", model_name, seconds, tokens_in, tokens_out)
        metrics = current_metrics()
        if metrics is not None:
            metrics.add_llm_call(seconds, tokens_in, tokens_out)
        return response

    @classmethod
    def _invoke_with_regeneration(cls, llm: Any, model_name: str, input: Any, config: Any, kwargs: dict) -> Any:
        """Call the model; on a malformed tool call (400 tool_use_failed) sample once more."""
        try:
            return cls._timed_invoke(llm, model_name, input, config, kwargs)
        except Exception as exc:  # noqa: BLE001
            if not _is_bad_generation_error(exc):
                raise
            logger.warning(f"'{model_name}' returned an invalid tool call; regenerating once…")
            return cls._timed_invoke(llm, model_name, input, config, kwargs)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """
        Try each model in order; fall through to the next on a rate-limit error. If every
        model is rate-limited in one pass, wait (honouring retry-after) and retry the whole
        chain, up to `rate_limit_max_retries` times, before raising.
        """
        last_error: Exception | None = None
        # Models skipped for the rest of this call, with the reason: removed from Groq, or
        # produced an invalid tool call twice in a row.
        unavailable: dict[int, str] = {}

        for outer in range(self._rate_limit_max_retries + 1):
            for index, llm in enumerate(self._llms):
                model_name = self._models[index]
                if index in unavailable:
                    continue
                for attempt in range(self._retries_per_model):
                    try:
                        return self._invoke_with_regeneration(llm, model_name, input, config, kwargs)
                    except Exception as exc:  # noqa: BLE001 - non-rate-limit errors re-raised below
                        if _is_model_unavailable_error(exc):
                            unavailable[index] = "not available on Groq"
                            last_error = exc
                            logger.warning(f"model '{model_name}' is not available on Groq; "
                                  "update GROQ_MODELS. Trying the next model.")
                            break
                        if _is_bad_generation_error(exc):
                            unavailable[index] = "returned invalid tool calls"
                            last_error = exc
                            logger.warning(f"'{model_name}' returned an invalid tool call twice; "
                                  "trying the next model.")
                            break
                        if not _is_rate_limit_error(exc):
                            raise
                        last_error = exc
                        is_last_attempt = attempt == self._retries_per_model - 1
                        if not is_last_attempt:
                            time.sleep(self._backoff_seconds * (attempt + 1))
                            logger.warning(
                                f"'{model_name}' rate-limited; "
                                f"retry {attempt + 2}/{self._retries_per_model}…"
                            )
                        elif index < len(self._llms) - 1:
                            logger.warning(
                                f"'{model_name}' rate-limited; "
                                f"switching to '{self._models[index + 1]}'."
                            )

            if len(unavailable) == len(self._llms):
                details = ", ".join(f"{self._models[i]} ({why})" for i, why in unavailable.items())
                raise RuntimeError(
                    f"No configured Groq model could answer: {details}. If models were removed, "
                    "set GROQ_MODELS in .env (see https://console.groq.com/docs/models)."
                ) from last_error

            # Every model was rate-limited this pass. Wait it out and retry the chain.
            if outer < self._rate_limit_max_retries:
                default_wait = min(2.0 * (2 ** outer), self._rate_limit_wait_cap)
                wait = min(_retry_after_seconds(last_error, default_wait) + 0.5,
                           self._rate_limit_wait_cap)
                logger.warning(
                    f"all models rate-limited; waiting {wait:.1f}s then retrying "
                    f"(pass {outer + 2}/{self._rate_limit_max_retries + 1})…"
                )
                time.sleep(wait)

        raise RuntimeError(
            "All models are rate-limited after "
            f"{self._rate_limit_max_retries + 1} passes: " + ", ".join(self._models)
        ) from last_error


def get_chat_llm(
    temperature: float = 0.0,
    models: Sequence[str] | None = None,
    role: str | None = None,
    **kwargs: Any,
) -> FallbackChatLLM:
    """
    Build the app's standard fallback-enabled chat model.

    The model chain is `models` if given, else the chain configured for `role`
    (e.g. SUPERVISOR_MODELS, GRADER_MODELS — see app/config.models_for), else LLM_MODELS.
    Rate-limit wait-and-retry is tunable via env: GROQ_RATE_LIMIT_RETRIES (default 4),
    GROQ_RATE_LIMIT_WAIT_CAP seconds (default 30).
    """
    import os

    from app.config import GROQ_API_KEY, models_for

    kwargs.setdefault("rate_limit_max_retries", int(os.getenv("GROQ_RATE_LIMIT_RETRIES", "4")))
    kwargs.setdefault("rate_limit_wait_cap", float(os.getenv("GROQ_RATE_LIMIT_WAIT_CAP", "30")))

    return FallbackChatLLM(
        models or models_for(role),
        api_key=GROQ_API_KEY,
        temperature=temperature,
        **kwargs,
    )
