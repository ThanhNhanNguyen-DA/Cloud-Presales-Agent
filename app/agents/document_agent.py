"""
Document Agent for RAG-based lookup of technical documentation, SLA, security, and compliance info.
Uses semantic search over cloud_documents table via Supabase vector similarity.
"""

from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from app.llm import get_chat_llm
from app.graph.state import GraphState
from app.tools.rag_tools import search_cloud_documents


TOOLS = [search_cloud_documents]
TOOL_MAP = {t.name: t for t in TOOLS}

llm = get_chat_llm(temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = """You are the Document Agent for the Presales Cloud system.
Task: look up technical documentation, SLA, security (ISO 27001, PCI-DSS), and legal compliance (Decree 13/ND-CP).
You MUST use the search_cloud_documents tool to look things up — DO NOT make up SLA terms or legal figures yourself.
If no relevant document is found, clearly say that the information is not available yet, do not guess.
Always answer in the SAME language as the user's question, clearly citing the service name and the source of the information found."""


def document_agent_node(state: GraphState) -> GraphState:
    """
    Document Agent node that uses RAG to look up documentation from cloud_documents table.
    
    Args:
        state: GraphState containing user_query and optional retrieved_experience
        
    Returns:
        Updated GraphState with agent_response and final_response filled
    """
    experience_context = state.get("retrieved_experience") or ""
    system_content = SYSTEM_PROMPT
    if experience_context:
        system_content += f"\n\n{experience_context}\nYou MUST follow the correction above if it is relevant."

    # Prior turns (if any) let the agent resolve follow-ups like "and its SLA?".
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
            # Phase 8: keep the raw tool/RAG output so the Grounding Grader can cross-check numbers.
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
