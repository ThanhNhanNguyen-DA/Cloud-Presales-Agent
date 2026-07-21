"""
Calculator & Sizing Agent for the Presales Cloud system.
Uses LLM function calling (Groq / Llama 3.3 70B) to invoke the 3 calculation tools.
"""

from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from app.llm import get_chat_llm
from app.graph.state import GraphState
from app.tools.presales_tools import (
    calculate_backup_and_cost,
    size_web_app_vm,
    size_database_storage
)


TOOLS = [calculate_backup_and_cost, size_web_app_vm, size_database_storage]
TOOL_MAP = {t.name: t for t in TOOLS}

llm = get_chat_llm(temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = """You are the Calculator & Sizing Agent for the Presales Cloud system.
Task: accurately calculate cost, backup duration, VM sizing, and storage/IOPS sizing.
You MUST use the provided tools to perform calculations — DO NOT estimate numbers yourself, to avoid errors.
If you are missing information needed to call a tool (e.g. missing data volume or bandwidth), ask the user instead of guessing.
After getting the tool result, present it naturally and clearly, in the SAME language as the user's question, for a Presales staff member, with concrete numbers.

Format your answer as:
1. Inputs/assumptions used (short bullet list).
2. The result figure(s), each with its **unit** in bold (e.g. **8 vCPU**, **16 GB RAM**, **0.23 USD/month**).
3. A one-line reminder to double-check the numbers before quoting them to the customer."""


def calculator_agent_node(state: GraphState) -> GraphState:
    """
    Calculator Agent node that uses LLM function calling to invoke sizing tools.
    
    Processes the user query by:
    1. Calling the LLM with available tools attached
    2. Executing any tool calls returned by the LLM
    3. Feeding results back to the LLM for synthesis
    4. Repeating until the LLM generates a final answer or max iterations reached
    
    Args:
        state: GraphState containing user_query and optional retrieved_experience
        
    Returns:
        Updated GraphState with agent_response and final_response filled
    """
    experience_context = state.get("retrieved_experience") or ""
    system_content = SYSTEM_PROMPT
    if experience_context:
        system_content += f"\n\n{experience_context}\nYou MUST follow the correction above if it is relevant."

    # Prior turns (if any) let the agent resolve follow-ups like "and for 20GB?".
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
            if tool_name in TOOL_MAP:
                tool_result = TOOL_MAP[tool_name].invoke(tool_args)
            else:
                tool_result = {"error": f"Tool {tool_name} not found"}

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

    final_text = ai_msg.content
    state["agent_response"] = final_text
    state["final_response"] = final_text
    return state
