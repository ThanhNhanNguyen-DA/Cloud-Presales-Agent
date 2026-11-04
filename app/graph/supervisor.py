"""
Supervisor node for routing user queries to appropriate handlers.
Includes mandatory experience retrieval before routing.
Uses the LLM (Groq, see GROQ_MODELS) to classify intent into one of 4 groups.
"""

import logging
import re

from langchain_core.messages import HumanMessage

from app.llm import get_chat_llm
from app.graph.state import GraphState, Route
from app.tools.memory_tools import retrieve_past_experiences

logger = logging.getLogger(__name__)


llm = get_chat_llm(temperature=0, role="supervisor")

ROUTING_PROMPT = """You are the Supervisor for the Presales Cloud Chatbot system.
Classify the LATEST question into EXACTLY one of these groups: calculator, document, competitor, general.

- calculator: compute a number from the user's inputs — backup duration, storage cost per month,
  VM sizing (vCPU/RAM for N concurrent users), database growth / storage tier for an IOPS need.
- document: anything about OUR OWN cloud services — what a service does, its features, specs, limits,
  how to create/configure/use it, SLA and uptime, security certifications, legal compliance.
  Our services include: Cloud Server, Object Storage, ELB (load balancer), Auto Scaling, VPC, VPN,
  RDS / RDS for MySQL, Key Management Service, Cloud Trace, Elastic IP, Elastic GPU, Cloud Kafka,
  DaaS, Terraform, Availability Zones.
- competitor: the question names a competitor — AWS / Azure / GCP services (what is our equivalent,
  compare, migrate) or Vietnamese providers (VNG Cloud, FPT Cloud, Bizfly, VNPT, Viettel IDC): their
  services, features, SLAs or prices.
- general: greetings, thanks, small talk, questions about this assistant, or off-topic requests.

Examples:
"Cách thêm listener HTTPS cho Load Balancer?" -> document
"Cloud Kafka giới hạn bao nhiêu topic?" -> document
"AWS DynamoDB bên mình có gì tương đương?" -> competitor
"Bảng giá Object Storage của Viettel IDC?" -> competitor
"Ứng dụng Go 300 người dùng đồng thời cần bao nhiêu RAM?" -> calculator
"Cảm ơn, vậy là đủ rồi" -> general
{history_section}
LATEST question: {query}

The LATEST question may be a follow-up (e.g. "and the monthly cost?", "what about GCP?").
If so, use the recent conversation above to understand what it refers to before classifying.

Return ONLY one of the 4 words above, with no explanation."""


def _format_history_for_routing(messages, max_messages: int = 4) -> str:
    """Render the last few turns as a compact block for the routing prompt (empty if none)."""
    if not messages:
        return ""
    recent = messages[-max_messages:]
    lines = []
    for m in recent:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        text = (m.content or "").strip().replace("\n", " ")
        if len(text) > 300:
            text = text[:300] + "…"
        lines.append(f"{role}: {text}")
    return "\nRecent conversation:\n" + "\n".join(lines) + "\n"


VALID_ROUTES: tuple[Route, ...] = ("calculator", "document", "competitor", "general")
_ROUTE_WORD = re.compile(r"\b(" + "|".join(VALID_ROUTES) + r")\b")


def parse_route(content) -> Route:
    """
    Extract the route from the classifier's reply.

    Tolerates casing, punctuation, quotes or extra words ("Calculator.", "**document**",
    "Route: competitor"); the first valid route word wins. Falls back to "general".
    """
    text = content if isinstance(content, str) else str(content or "")
    match = _ROUTE_WORD.search(text.lower())
    return match.group(1) if match else "general"


def supervisor_node(state: GraphState) -> dict:
    """
    Supervisor node:
    1. Retrieves past (reviewed) corrections — mandatory, not an LLM decision.
    2. Classifies the question into one of the 4 routes, using the recent conversation so
       follow-ups ("and the monthly cost?") classify correctly.

    Returns the partial state update {"route", "retrieved_experience"}.
    """
    query = state["user_query"]
    experience_context = retrieve_past_experiences(query)

    history_section = _format_history_for_routing(state.get("messages") or [])
    prompt = ROUTING_PROMPT.format(query=query, history_section=history_section)
    route = parse_route(llm.invoke(prompt).content)
    logger.info("route=%s corrections=%s", route, "yes" if experience_context else "no")
    return {"route": route, "retrieved_experience": experience_context}
