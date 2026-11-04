"""
Competitor Agent: how our services compare with competitors.

Two data sources, two tools:
- the curated mapping matrix (AWS / Azure / GCP -> our equivalent + migration difficulty);
- crawled public pages of Vietnamese competitors (VNG, FPT, Bizfly, VNPT, Viettel IDC):
  their products, features, SLAs and list prices (search_competitor_docs).
"""

from app.agents.tool_agent import run_tool_agent
from app.graph.state import GraphState
from app.llm import get_chat_llm
from app.tools.competitor_tools import find_competitor_mapping, list_all_competitor_services
from app.tools.rag_tools import search_competitor_docs

TOOLS = [find_competitor_mapping, list_all_competitor_services, search_competitor_docs]

llm = get_chat_llm(temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = """You are the Competitor Agent for the Presales Cloud system.
Task: help Presales staff position our services against competitors.

Tools — pick by competitor:
- AWS / Azure / GCP service -> our equivalent: find_competitor_mapping (or list_all_competitor_services
  to list what is mapped). DO NOT guess or invent an equivalent service.
- Vietnamese providers (VNG Cloud, FPT Cloud, Bizfly Cloud, VNPT Cloud, Viettel IDC): search_competitor_docs
  for their products, features, SLA claims and list prices. These come from their public websites:
  ALWAYS cite the page URL and the crawl date, say prices are public list prices (VAT and discounts may
  differ), and NEVER present a competitor's figures as ours.
If the tools return nothing, say plainly that there is no data yet and suggest confirming with the technical team.

Answer in the SAME language as the user's question. Lead with the key point in bold (e.g. **AWS S3 → Object
Storage**, or **Bizfly Cloud Server: từ 116.820 đ/tháng**), then the similarities/differences, migration
difficulty when available (or "not determined yet"), and the sources. Give enough context for Presales staff
to use it confidently — don't reduce the answer to a bare list."""


def competitor_agent_node(state: GraphState) -> dict:
    """Competitor Agent node: AWS/Azure/GCP mapping + Vietnamese competitors' public pages. (Loop, error handling and retry feedback: app/agents/tool_agent.py.)"""
    return run_tool_agent(
        state, llm_with_tools=llm_with_tools, tools=TOOLS, system_prompt=SYSTEM_PROMPT
    )
