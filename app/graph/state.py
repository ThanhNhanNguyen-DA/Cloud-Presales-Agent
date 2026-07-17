"""
LangGraph State definition for the Cloud Presales Agent system.
"""

from typing import TypedDict, List, Optional, Any


class GraphState(TypedDict):
    """
    State maintained throughout the graph execution.
    
    Fields:
        messages: Conversation history (list of messages).
        user_query: The original question from the user.
        route: The agent/handler chosen by supervisor (calculator, document, competitor, general).
        retrieved_experience: Lessons learned retrieved from feedback memory (Phase 7).
        raw_tool_output: Raw tool/RAG output for the Grounding Grader to cross-check (Phase 8).
        agent_response: The direct response from the sub-agent.
        final_response: The final answer after grading (Phase 8).
        grading_passed: Result of the Grounding Grader (Phase 8).
        retry_count: Number of grading retries performed, to bound the retry loop (Phase 8).
    """
    messages: List[Any]
    user_query: str
    route: Optional[str]
    retrieved_experience: Optional[str]
    raw_tool_output: Optional[str]
    agent_response: Optional[str]
    final_response: Optional[str]
    grading_passed: Optional[bool]
    retry_count: Optional[int]
