"""
Multi-turn conversation memory.

Unlike the *feedback* memory (app/tools/memory_tools.py), which stores lessons learned
from human corrections and is retrieved by vector similarity, this module stores the
*literal* turns of an ongoing conversation, grouped by `session_id`, so a follow-up
question ("and the monthly cost?", "what about for GCP?") can be answered with the
context of what was already said.

Storage is a plain Supabase table (`cloud_conversation_history`), one row per message.
Retrieval is a raw sliding window: the most recent `max_turns` message-pairs, returned
as LangChain messages ready to splice into an agent's prompt.

Neither function is a `@tool`: they are called directly by the app (the API layer
loads history into the graph state, and saves the new turn after the answer is ready),
not chosen by the LLM. Both are best-effort — any DB error is logged and swallowed so a
memory hiccup never breaks the answer.
"""

from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.db.supabase_client import supabase


def load_conversation_history(session_id: Optional[str], max_turns: int = 5) -> List[BaseMessage]:
    """
    Load the most recent turns of a conversation as LangChain messages.

    Args:
        session_id: The conversation identifier. If falsy, returns [] (stateless call).
        max_turns: How many recent user/assistant *pairs* to keep (the sliding window).

    Returns:
        A list of HumanMessage/AIMessage in chronological order (oldest first), at most
        `2 * max_turns` items. Empty on no history or any error.
    """
    if not session_id:
        return []
    try:
        response = (
            supabase.table("cloud_conversation_history")
            .select("role, content")
            .eq("session_id", session_id)
            .order("created_at", desc=True)  # newest first, so LIMIT keeps the recent tail…
            .limit(max_turns * 2)
            .execute()
        )
        rows = list(reversed(response.data or []))  # …then flip back to chronological order.
        messages: List[BaseMessage] = []
        for row in rows:
            if row["role"] == "user":
                messages.append(HumanMessage(content=row["content"]))
            else:
                messages.append(AIMessage(content=row["content"]))
        return messages
    except Exception as e:
        print(f"Error loading conversation history: {str(e)}")
        return []


def save_conversation_turn(
    session_id: Optional[str],
    role: str,
    content: str,
    route: Optional[str] = None,
) -> None:
    """
    Append one message to a conversation's history.

    Args:
        session_id: The conversation identifier. If falsy, this is a no-op (stateless call).
        role: 'user' or 'assistant'.
        content: The message text.
        route: The route chosen for an assistant turn (ignored for user turns).
    """
    if not session_id or not content:
        return
    try:
        supabase.table("cloud_conversation_history").insert({
            "session_id": session_id,
            "role": role,
            "content": content,
            "route": route,
        }).execute()
    except Exception as e:
        print(f"Error saving conversation turn: {str(e)}")
