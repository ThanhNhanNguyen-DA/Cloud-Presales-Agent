"""
Document Agent for RAG-based lookup of technical documentation, SLA, security, and compliance info.
Uses semantic search over cloud_documents table via Supabase vector similarity.
"""

from app.llm import get_chat_llm
from app.graph.state import GraphState
from app.agents.tool_agent import run_tool_agent
from app.tools.rag_tools import search_cloud_documents


TOOLS = [search_cloud_documents]

llm = get_chat_llm(temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = """You are the Document Agent for the Presales Cloud system.
Task: look up technical documentation, SLA, security (ISO 27001, PCI-DSS), and legal compliance (Decree 13/ND-CP).
You MUST use the search_cloud_documents tool to look things up — DO NOT make up SLA terms or legal figures yourself.
The tool searches OUR documentation only; competitor web pages are excluded on purpose. If the user asks about a competitor, say that the competitor specialist handles it.
If no relevant document is found, clearly say that the information is not available yet, do not guess.
Always answer in the SAME language as the user's question, clearly citing the service name and the source of the information found.

Answer clearly and helpfully: state the finding and explain it in enough detail to be useful, cite the **service name** and the source document/category it came from, and note any caveats or conditions. If the information is partial or not found, say so plainly instead of guessing — don't reduce the answer to a bare line."""


def document_agent_node(state: GraphState) -> dict:
    """Document Agent node: answers SLA / security / compliance questions from the RAG document store. (Loop, error handling and retry feedback: app/agents/tool_agent.py.)"""
    return run_tool_agent(
        state, llm_with_tools=llm_with_tools, tools=TOOLS, system_prompt=SYSTEM_PROMPT
    )
