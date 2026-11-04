"""
LangGraph builder: constructs the graph with supervisor and specialized agents.
- Calculator agent (Phase 4): real calculator_agent_node
- Document agent (Phase 5): real document_agent_node  
- Competitor agent (Phase 6): real competitor_agent_node
- General: general_node (direct answer, no tools)
"""

from langgraph.graph import END, START, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.tool_agent import message_text
from app.llm import get_chat_llm
from app.graph.state import GraphState
from app.graph.supervisor import supervisor_node
from app.agents.calculator_agent import calculator_agent_node
from app.agents.document_agent import document_agent_node
from app.agents.competitor_agent import competitor_agent_node
from app.graders.self_rag_grader import MAX_GRADING_ATTEMPTS, grounding_grader_node


llm = get_chat_llm(temperature=0.3)

GENERAL_SYSTEM_PROMPT = """You are the front-desk assistant for the Presales Cloud system.
Context: the system has three specialists — cost/sizing calculation, documentation lookup
(SLA, security, compliance), and AWS/Azure/GCP competitor comparison. Use the conversation
history to stay consistent with what was already discussed.

Task: handle greetings, small talk, and meta questions about what the system can do. When the
user asks about cost/sizing, SLA/security/compliance, or a competitor comparison, briefly say
you can help with that and invite them to ask it directly (a specialist will handle it).

Do NOT invent technical facts, SLAs, prices, service names, or comparisons — those come only
from the specialists and their tools.

Answer in the SAME language as the user's question — concise but complete; a short paragraph is fine."""


def general_node(state: GraphState) -> dict:
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
    answer = message_text(response).strip()
    return {"agent_response": answer, "final_response": answer}


def route_decision(state: GraphState) -> str:
    """Route decision based on supervisor classification."""
    return state["route"]


SPECIALIST_ROUTES = ("calculator", "document", "competitor")


def grading_decision(state: GraphState) -> str:
    """
    Decide what to do after the Grounding Grader (Phase 8).

    Pure read-only routing: `retry_count` / `grader_feedback` are set inside
    `grounding_grader_node`, since LangGraph does not reliably persist state
    mutations made inside a conditional-edge function.

    - PASS -> end.
    - FAIL with retry budget left -> re-run the SAME specialist (it receives the grader's
      feedback). Routing and memory recall are not repeated: the question hasn't changed.
    - FAIL with the budget used (or the grader could not verify) -> end; the node has
      already appended a warning to `final_response`.
    """
    if state.get("grading_passed"):
        return "end"
    if (state.get("retry_count") or 0) >= MAX_GRADING_ATTEMPTS:
        return "end"
    route = state.get("route")
    return route if route in SPECIALIST_ROUTES else "end"


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

    graph.add_edge(START, "supervisor")

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

    # Grader either ends, or re-runs the same specialist once (bounded by retry_count).
    graph.add_conditional_edges(
        "grader",
        grading_decision,
        {
            "calculator": "calculator",
            "document": "document",
            "competitor": "competitor",
            "end": END,
        }
    )

    return graph.compile()


