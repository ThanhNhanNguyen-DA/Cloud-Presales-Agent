"""
Competitor Mapping Agent for AWS/Azure/GCP to local service equivalence lookup.
Uses direct database queries for structured competitor mapping data.
"""

from app.llm import get_chat_llm
from app.graph.state import GraphState
from app.agents.tool_agent import run_tool_agent
from app.tools.competitor_tools import find_competitor_mapping, list_all_competitor_services


TOOLS = [find_competitor_mapping, list_all_competitor_services]

llm = get_chat_llm(temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = """You are the Competitor Mapping Agent for the Presales Cloud system.
Task: map AWS/Azure/GCP services to their local service equivalents.
You MUST use the tools to look up the mapping matrix — DO NOT guess or invent an equivalent service.
If nothing is found in the database, clearly say there is no data yet, and suggest the user contact the technical team to confirm.
Always answer in the SAME language as the user's question, highlighting similarities/differences and migration difficulty if available.

Answer clearly and helpfully: lead with the mapping in bold (e.g. **AWS S3 → Object Storage**), then explain the key similarities and differences and the migration difficulty (state it is not determined yet if unavailable). Give enough context for Presales staff to use it confidently — don't reduce the answer to a bare list."""


def competitor_agent_node(state: GraphState) -> dict:
    """Competitor Agent node: maps AWS/Azure/GCP services to local equivalents from the mapping table. (Loop, error handling and retry feedback: app/agents/tool_agent.py.)"""
    return run_tool_agent(
        state, llm_with_tools=llm_with_tools, tools=TOOLS, system_prompt=SYSTEM_PROMPT
    )
