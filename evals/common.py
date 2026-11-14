"""
Shared helpers for the evaluation suite (evals/eval_grader.py, evals/eval_e2e.py).

`configure()` must run BEFORE any `app.*` import: offline (mock) mode replaces the
credentials with dummies, the same isolation as tests/conftest.py, so a mock run can never
reach the real Groq / Supabase project, even when `.env` holds real keys
(`load_dotenv()` does not override variables that already exist).
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

DATASETS = Path(__file__).parent / "datasets"
RESULTS = Path(__file__).parent / "results"  # read by the Streamlit eval dashboard (ui/eval_data.py)
OFFLINE_ENV = {  # same dummies as tests/conftest.py, so importing an eval under pytest changes nothing
    "GROQ_API_KEY": "test-groq-key",
    "SUPABASE_URL": "http://127.0.0.1:54321",
    "SUPABASE_KEY": "test-supabase-key",
}


def configure(live: bool) -> None:
    if not live:
        os.environ.update(OFFLINE_ENV)


def load_jsonl(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# --------------------------------------------------------------------------- text matching
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def canonical_number(token: str) -> str:
    """'99,9' / '99.9' -> '99.9'; '6.0' -> '6'; '1.198,81' / '1,198.81' -> '1.198.81'."""
    token = token.replace(",", ".")
    try:
        value = float(token)
    except ValueError:
        return token  # grouped thousands: VN and EN spellings map to the same string
    return str(int(value)) if value.is_integer() else str(value)


def numbers(text: str) -> set[str]:
    return {canonical_number(n) for n in _NUMBER.findall(text or "")}


def _squash(text: str) -> str:
    return re.sub(r"[\s*`_]+", "", (text or "").lower())


def contains(text: str, expected: str | list[str]) -> bool:
    """
    True if `text` contains `expected` (a list = any of). A purely numeric expectation is
    matched as a number, so "99.9" matches "99,9 %" but NOT "99.95"; anything else is a
    case/whitespace/markdown-insensitive substring.
    """
    if isinstance(expected, list):
        return any(contains(text, e) for e in expected)
    if _NUMBER.fullmatch(expected):
        return canonical_number(expected) in numbers(text)
    return _squash(expected) in _squash(text)


def unsupported_numbers(answer: str, *sources: str) -> list[str]:
    """Numbers in `answer` that appear in none of `sources` (question, tool args/results)."""
    known = set().union(*(numbers(s) for s in sources))
    # shortcut: integers below 10 are skipped (list markers, "1 GB = 10^9"); a fabricated
    # single digit slips through, upgrade to a units-aware check if that ever matters.
    return sorted(n for n in numbers(answer) - known
                  if not (n.isdigit() and int(n) < 10))


# --------------------------------------------------------------------------- offline grader
_PROMPT_PARTS = re.compile(
    r"USER QUESTION:\n(.*?)\n\nRAW DATA[^\n]*:\n(.*?)\n\nFINAL ANSWER[^\n]*:\n(.*?)\n\nCheck:", re.S)


class HeuristicGraderLLM:
    """
    Offline stand-in for the grader LLM: FAIL iff the answer has a number found neither in
    the question nor in the raw data. A lexical BASELINE, not a model of the LLM grader
    (it cannot see invented names or certifications without digits).
    """

    def invoke(self, prompt: str, *_, **__):
        from langchain_core.messages import AIMessage

        question, raw, answer = _PROMPT_PARTS.search(prompt).groups()
        bad = unsupported_numbers(answer, question, raw)
        verdict = {"verdict": "FAIL" if bad else "PASS",
                   "issues": [f"number {n} not in the raw data" for n in bad]}
        return AIMessage(content=json.dumps(verdict))


# --------------------------------------------------------------------------- reporting
def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100); 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)], 3)


def rate(hits: int, total: int) -> float | None:
    return round(hits / total, 3) if total else None


def md_table(rows: list[dict]) -> str:
    if not rows:
        return "_(none)_\n"
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r.get(c, "")).replace("|", "\\|").replace("\n", " ") for c in cols) + " |"
              for r in rows]
    return "\n".join(lines) + "\n"


def gate_rows(gates: dict[str, tuple]) -> list[dict]:
    """{name: (value, target_text, ok)} -> table rows."""
    return [{"metric": k, "value": v, "target": t, "status": "✅" if ok else "❌"}
            for k, (v, t, ok) in gates.items()]


def default_out(kind: str) -> str:
    """evals/results/<kind>_<timestamp>.json: one file per run, so runs can be compared."""
    return str(RESULTS / f"{kind}_{datetime.now():%Y%m%d-%H%M%S}.json")


def write_report(out: str | Path, payload: dict, markdown: str) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    out.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(f"✅ {out}  ·  {out.with_suffix('.md')}")
