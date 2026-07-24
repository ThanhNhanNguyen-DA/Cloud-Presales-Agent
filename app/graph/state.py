"""
LangGraph State definition for the Cloud Presales Agent system.
"""

from typing import Any, TypedDict


class GraphState(TypedDict):
    """
    State maintained throughout the graph execution.
    
    Fields:
        messages: Prior conversation turns (LangChain messages, oldest first) loaded
            from the multi-turn memory for the current session. Read-only within the
            graph — the API layer fills it before invocation and persists the new turn
            afterwards. Empty for a stateless call (no session_id).
        session_id: Conversation identifier used to load/save multi-turn history.
        user_query: The original question from the user.
        route: The agent/handler chosen by supervisor (calculator, document, competitor, general).
        retrieved_experience: Lessons learned retrieved from feedback memory (Phase 7).
        raw_tool_output: Raw tool/RAG output for the Grounding Grader to cross-check (Phase 8).
        agent_response: The direct response from the sub-agent.
        final_response: The final answer after grading (Phase 8).
        grading_passed: Result of the Grounding Grader (Phase 8).
        retry_count: Number of grading retries performed, to bound the retry loop (Phase 8).
    """
    messages: list[Any]
    session_id: str | None
    user_query: str
    route: str | None
    retrieved_experience: str | None
    raw_tool_output: str | None
    agent_response: str | None
    final_response: str | None
    grading_passed: bool | None
    retry_count: int | None
