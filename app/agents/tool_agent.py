"""
Shared tool-calling loop for the three specialist agents (calculator, document, competitor).

Each agent only supplies its system prompt, its tools and its LLM; this module runs the
loop the same way for all of them:

1. Build the prompt: system prompt (+ reviewed corrections, + the grader's feedback on a
   retry), the conversation history, and the user's question.
2. Call the LLM; execute every tool call it asks for; feed the results back; repeat up to
   MAX_TOOL_ROUNDS times.
3. Record each tool call (name, arguments, result) in `raw_tool_output` for the Grounding
   Grader. The field is
   RESET at the start of every run, so a retry is graded against its own tool calls only.

Failure handling (these used to crash the request or return an empty answer):
- a tool that raises (bad arguments, DB down, …) becomes an {"error": ...} tool result the
  LLM can explain, instead of an exception that kills the whole graph;
- an unknown tool name becomes an error result too;
- if the LLM still wants tools after MAX_TOOL_ROUNDS, those last calls are executed and the
  LLM is asked once more with tool_choice="none" to answer from what it has gathered;
- only if that still yields no text does the agent return a short "could not complete"
  message (never an empty string).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.graph.state import GraphState
from app.tools.memory_tools import EXPERIENCE_USAGE_NOTE

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 3

INCOMPLETE_ANSWER = (
    "Sorry, I could not complete this request (the lookup/calculation did not finish). "
    "Please rephrase the question or provide more details."
)

FINAL_ANSWER_NOTE = (
    "Tool budget reached. Do not call any more tools: answer the user's question now, using "
    "only the tool results above. If they don't contain the answer, say so plainly."
)

RETRY_NOTE = (
    "IMPORTANT — your previous answer to this question failed an automatic fact check "
    "against the tool data:\n{feedback}\n"
    "Call the tools again and answer using ONLY the figures, service names and terms that "
    "appear in the tool results. Copy numbers exactly; do not round or convert them."
)


def message_text(message: Any) -> str:
    """Plain text of an LLM message whose content may be a string or a list of blocks."""
    content = getattr(message, "content", "") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


def build_system_prompt(base_prompt: str, state: GraphState) -> str:
    """System prompt + reviewed corrections (if any) + grader feedback on a retry (if any)."""
    parts = [base_prompt]
    experience = state.get("retrieved_experience") or ""
    if experience:
        parts.append(f"{experience}\n{EXPERIENCE_USAGE_NOTE}")
    feedback = state.get("grader_feedback") or ""
    if feedback:
        parts.append(RETRY_NOTE.format(feedback=feedback))
    return "\n\n".join(parts)


def _run_tool(tool_map: dict, tool_call: dict) -> Any:
    name = tool_call.get("name")
    tool = tool_map.get(name)
    if tool is None:
        return {"error": f"Tool {name!r} not found."}
    try:
        return tool.invoke(tool_call.get("args") or {})
    except Exception as exc:  # noqa: BLE001 - surfaced to the LLM as data, never raised
        return {"error": f"Tool {name!r} failed: {exc}"}


def run_tool_agent(
    state: GraphState,
    *,
    llm_with_tools: Any,
    tools: list,
    system_prompt: str,
    max_rounds: int = MAX_TOOL_ROUNDS,
) -> dict:
    """Run one specialist agent; returns the partial state update (answer + raw tool data)."""
    tool_map = {t.name: t for t in tools}
    messages: list[Any] = [
        SystemMessage(content=build_system_prompt(system_prompt, state)),
        *(state.get("messages") or []),
        HumanMessage(content=state["user_query"]),
    ]

    raw_outputs: list[str] = []  # fresh for every run (incl. a grader retry)

    def run_pending_tools(ai_message: Any) -> None:
        for tool_call in ai_message.tool_calls:
            result = _run_tool(tool_map, tool_call)
            # Record the call's inputs too: the grader must see that "10 GB" came from the
            # user's question / the tool arguments, not from the model's imagination.
            raw_outputs.append(f"{tool_call.get('name')}({tool_call.get('args') or {}}) -> {result}")
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    rounds = 0
    while getattr(ai_msg, "tool_calls", None) and rounds < max_rounds:
        run_pending_tools(ai_msg)
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        rounds += 1

    if getattr(ai_msg, "tool_calls", None):
        # Budget used up: answer the pending calls, then force a text answer.
        run_pending_tools(ai_msg)
        try:
            ai_msg = llm_with_tools.invoke(
                [*messages, SystemMessage(content=FINAL_ANSWER_NOTE)], tool_choice="none"
            )
        except Exception as exc:  # noqa: BLE001 - fall back to the "could not complete" text
            logger.warning("Forced final answer failed: %s", exc)
            ai_msg = None

    answer = message_text(ai_msg).strip() if ai_msg is not None else ""
    if ai_msg is None or getattr(ai_msg, "tool_calls", None) or not answer:
        answer = INCOMPLETE_ANSWER

    return {
        "raw_tool_output": "\n---\n".join(raw_outputs) or None,
        "agent_response": answer,
        "final_response": answer,
    }
