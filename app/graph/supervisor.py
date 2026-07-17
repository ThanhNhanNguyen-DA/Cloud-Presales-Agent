"""
Supervisor node for routing user queries to appropriate handlers.
Includes mandatory experience retrieval before routing.
Uses the LLM (Groq / Llama 3.3 70B) to classify intent into one of 4 groups.
"""

from app.llm import get_chat_llm
from app.graph.state import GraphState
from app.tools.memory_tools import retrieve_past_experiences


llm = get_chat_llm(temperature=0)

ROUTING_PROMPT = """You are the Supervisor for the Presales Cloud Chatbot system.
Classify the following question into EXACTLY one of these groups: calculator, document, competitor, general.

- calculator: questions about cost calculation, backup duration, VM sizing, storage/IOPS sizing.
- document: questions about technical documentation, SLA, security, or legal compliance.
- competitor: questions comparing services with AWS/Azure/GCP.
- general: greetings or anything that doesn't fit the 3 groups above.

Question: {query}

Return ONLY one of the 4 words above, with no explanation."""


def supervisor_node(state: GraphState) -> GraphState:
    """
    Supervisor node that:
    1. Retrieves past experiences (mandatory, hardcoded, not LLM decision)
    2. Routes the user query to appropriate agent based on intent classification
    
    Args:
        state: GraphState containing user_query
        
    Returns:
        Updated GraphState with route set and retrieved_experience injected
    """
    query = state["user_query"]

    # Step 1: MANDATORY experience retrieval first, not dependent on LLM decision
    experience_context = retrieve_past_experiences(query)
    state["retrieved_experience"] = experience_context

    # Step 2: Routing as before
    prompt = ROUTING_PROMPT.format(query=query)
    response = llm.invoke(prompt)
    route = response.content.strip().lower()

    valid_routes = {"calculator", "document", "competitor", "general"}
    if route not in valid_routes:
        route = "general"

    state["route"] = route
    return state
