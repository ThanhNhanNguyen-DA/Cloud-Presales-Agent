"""
Calculator & Sizing Agent for the Presales Cloud system.
Uses LLM function calling (Groq, see GROQ_MODELS) to invoke the 3 calculation tools.
"""

from app.llm import get_chat_llm
from app.graph.state import GraphState
from app.agents.tool_agent import run_tool_agent
from app.tools.presales_tools import (
    calculate_backup_and_cost,
    size_web_app_vm,
    size_database_storage
)


TOOLS = [calculate_backup_and_cost, size_web_app_vm, size_database_storage]

llm = get_chat_llm(temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = """You are the Calculator & Sizing Agent for the Presales Cloud system.
Task: accurately calculate cost, backup duration, VM sizing, and storage/IOPS sizing.
You MUST use the provided tools to perform calculations — DO NOT estimate numbers yourself, to avoid errors.
If you are missing information needed to call a tool (e.g. missing data volume or bandwidth), ask the user instead of guessing.
If a tool returns an "error" (e.g. an unknown storage class or an invalid value), do NOT produce a figure for it: explain the problem and ask the user for a valid value.
After getting the tool result, present it naturally and clearly, in the SAME language as the user's question, for a Presales staff member, with concrete numbers.

Explain the result clearly and helpfully for Presales staff: state the inputs/assumptions you used, give the figure(s) with their **units in bold** (e.g. **8 vCPU**, **16 GB RAM**, **0.23 USD/month**), and add a short note on caveats or how to read them. Give enough context to use the numbers confidently — don't reduce the answer to a bare list. End with a brief reminder to double-check before quoting to the customer."""


def calculator_agent_node(state: GraphState) -> dict:
    """Calculator Agent node: answers cost / backup / VM / storage sizing questions with the calculation tools. (Loop, error handling and retry feedback: app/agents/tool_agent.py.)"""
    return run_tool_agent(
        state, llm_with_tools=llm_with_tools, tools=TOOLS, system_prompt=SYSTEM_PROMPT
    )
