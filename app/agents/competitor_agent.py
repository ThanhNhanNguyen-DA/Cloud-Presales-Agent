"""
Competitor Mapping Agent for AWS/Azure/GCP to local service equivalence lookup.
Uses direct database queries for structured competitor mapping data.
"""

from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from app.llm import get_chat_llm
from app.graph.state import GraphState
from app.tools.competitor_tools import find_competitor_mapping, list_all_competitor_services


TOOLS = [find_competitor_mapping, list_all_competitor_services]
TOOL_MAP = {t.name: t for t in TOOLS}

llm = get_chat_llm(temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = """You are the Competitor Mapping Agent for the Presales Cloud system.
Task: map AWS/Azure/GCP services to their local service equivalents.
You MUST use the tools to look up the mapping matrix — DO NOT guess or invent an equivalent service.
If nothing is found in the database, clearly say there is no data yet, and suggest the user contact the technical team to confirm.
Always answer in the SAME language as the user's question, highlighting similarities/differences and migration difficulty if available.

Format your answer as:
1. The mapping first, in bold (e.g. **AWS S3 → Object Storage**).
2. Key similarities/differences.
3. Migration difficulty if available; otherwise state it is not determined yet."""


def competitor_agent_node(state: GraphState) -> GraphState:
    """
    Competitor Agent node that uses structured database lookups for service mapping.
    
    Args:
        state: GraphState containing user_query and optional retrieved_experience
        
    Returns:
        Updated GraphState with agent_response and final_response filled
    """
    experience_context = state.get("retrieved_experience") or ""
    system_content = SYSTEM_PROMPT
    if experience_context:
        system_content += f"\n\n{experience_context}\nYou MUST follow the correction above if it is relevant."

    # Prior turns (if any) let the agent resolve follow-ups like "what about GCP?".
    history = state.get("messages") or []
    messages = [
        SystemMessage(content=system_content),
        *history,
        HumanMessage(content=state["user_query"])
    ]

    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    max_iterations = 3
    iteration = 0
    while ai_msg.tool_calls and iteration < max_iterations:
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_result = TOOL_MAP[tool_name].invoke(tool_args) if tool_name in TOOL_MAP else {"error": "Tool not found"}
            # Phase 8: keep the raw tool output so the Grounding Grader can cross-check numbers.
            existing = state.get("raw_tool_output") or ""
            state["raw_tool_output"] = (existing + "\n---\n" if existing else "") + str(tool_result)
            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            ))
        
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        iteration += 1

    state["agent_response"] = ai_msg.content
    state["final_response"] = ai_msg.content
    return state
