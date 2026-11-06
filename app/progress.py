"""
Plain facts about each graph step, for the UI's activity log.

`/chat/stream` sends a `progress` event after every node. This module adds a small `detail`
dict with what happened in that step — no internals, just facts a non-technical user can
be told about (the UI turns them into sentences):

    supervisor -> {"route": "calculator", "used_corrections": False}
    calculator / document / competitor
               -> {"tools": ["calculate_backup_and_cost"], "found_data": True}
    general    -> {"tools": []}
    grader     -> {"verdict": "pass" | "retry" | "warn" | "unverified" | "skipped",
                   "attempt": 1}
"""

from __future__ import annotations

import re

from app.graders.self_rag_grader import UNVERIFIED_WARNING

SPECIALISTS = {"calculator", "document", "competitor"}
_TOOL_CALL = re.compile(r"^(\w+)\(")
# Tool results that mean "nothing usable came back".
_NO_DATA = re.compile(r"No relevant|No matching|No mapping data|not found|'error'|\"error\"", re.IGNORECASE)


def tools_used(raw_tool_output: str | None) -> list[str]:
    """Tool names from raw_tool_output entries ("name(args) -> result", separated by ---)."""
    names: list[str] = []
    for entry in (raw_tool_output or "").split("\n---\n"):
        match = _TOOL_CALL.match(entry.strip())
        if match and match.group(1) not in names:
            names.append(match.group(1))
    return names


def found_data(raw_tool_output: str | None) -> bool:
    """True if at least one tool call returned usable data (not an error / empty search)."""
    entries = [e for e in (raw_tool_output or "").split("\n---\n") if e.strip()]
    return any(not _NO_DATA.search(e.split(" -> ", 1)[-1]) for e in entries)


def describe_step(node: str, update: dict, raw_before: str | None, attempt: int = 1) -> dict:
    """
    Detail for one `progress` event.

    `update` is the node's partial state update; `raw_before` is the raw tool output known
    before this node ran (the grader auto-passes when there was none); `attempt` is how many
    times the grader has run in this request (1, or 2 after a retry).
    """
    if not isinstance(update, dict):
        return {}
    if node == "supervisor":
        return {"route": update.get("route"), "used_corrections": bool(update.get("retrieved_experience"))}
    if node in SPECIALISTS:
        raw = update.get("raw_tool_output")
        return {"tools": tools_used(raw), "found_data": found_data(raw)}
    if node == "general":
        return {"tools": []}
    if node == "grader":
        if not (raw_before or "").strip():
            return {"verdict": "skipped", "attempt": attempt}
        if update.get("grading_passed"):
            return {"verdict": "pass", "attempt": attempt}
        final = update.get("final_response") or ""
        if UNVERIFIED_WARNING.strip() in final:
            return {"verdict": "unverified", "attempt": attempt}
        if update.get("grader_feedback"):
            return {"verdict": "retry", "attempt": attempt}
        return {"verdict": "warn", "attempt": attempt}
    return {}
