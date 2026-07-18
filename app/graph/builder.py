"""
LangGraph builder: constructs the graph with supervisor and specialized agents.
- Calculator agent (Phase 4): real calculator_agent_node
- Document agent (Phase 5): real document_agent_node  
- Competitor agent (Phase 6): real competitor_agent_node
- General: placeholder general_node
"""

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from app.llm import get_chat_llm
from app.graph.state import GraphState
from app.graph.supervisor import supervisor_node
from app.agents.calculator_agent import calculator_agent_node
from app.agents.document_agent import document_agent_node
from app.agents.competitor_agent import competitor_agent_node
from app.graders.self_rag_grader import grounding_grader_node


llm = get_chat_llm(temperature=0.3)

GENERAL_SYSTEM_PROMPT = """You are a helpful assistant for the Presales Cloud system.
Answer greetings and general questions naturally, in the SAME language as the user.
Use the conversation history to stay consistent with what was already discussed."""


def general_node(state: GraphState) -> GraphState:
    """
    General-purpose response node. Answers questions directly via the LLM.
    Used for general queries that don't require specialized agents.
    Includes the recent conversation so short follow-ups stay in context.
    """
    history = state.get("messages") or []
    messages = [
        SystemMessage(content=GENERAL_SYSTEM_PROMPT),
        *history,
        HumanMessage(content=state["user_query"]),
    ]
    response = llm.invoke(messages)
    state["agent_response"] = response.content
    state["final_response"] = response.content
    return state


def route_decision(state: GraphState) -> str:
    """Route decision based on supervisor classification."""
    return state["route"]


def grading_decision(state: GraphState) -> str:
    """
    Decide what to do after the Grounding Grader (Phase 8).

    Pure read-only routing: the actual `retry_count` increment happens inside
    `grounding_grader_node`, since LangGraph does not reliably persist state
    mutations made inside a conditional-edge function.

    - PASS -> end.
    - FAIL but retry budget remains (retry_count < 2 after the node bumped it)
      -> retry (re-route from the supervisor once).
    - FAIL and retry budget exhausted -> end (the node has already appended a
      warning to `final_response`, so the user still gets a non-empty answer).
    """
    if state.get("grading_passed"):
        return "end"
    if (state.get("retry_count") or 0) >= 2:
        return "end"
    return "retry"


def build_graph():
    """
    Build and compile the LangGraph graph.
    
    Structure:
    - Entry: supervisor_node (classifies intent into 4 groups)
    - Routing: conditional edges to appropriate handlers
      - calculator: real calculator_agent_node (Phase 4)
      - document: real document_agent_node (Phase 5)
      - competitor: real competitor_agent_node (Phase 6)
      - general: general_node
    - Exit: all paths lead to END
    """
    graph = StateGraph(GraphState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("calculator", calculator_agent_node)
    graph.add_node("document", document_agent_node)
    graph.add_node("competitor", competitor_agent_node)
    graph.add_node("general", general_node)
    graph.add_node("grader", grounding_grader_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_decision,
        {
            "calculator": "calculator",
            "document": "document",
            "competitor": "competitor",
            "general": "general",
        }
    )

    # The 3 specialized agents flow into the Grounding Grader (Phase 8).
    graph.add_edge("calculator", "grader")
    graph.add_edge("document", "grader")
    graph.add_edge("competitor", "grader")
    # General answers don't use a tool, so there's nothing to cross-check.
    graph.add_edge("general", END)

    # Grader either ends, or re-routes once via the supervisor (bounded by retry_count).
    graph.add_conditional_edges(
        "grader",
        grading_decision,
        {
            "retry": "supervisor",
            "end": END,
        }
    )

    return graph.compile()


