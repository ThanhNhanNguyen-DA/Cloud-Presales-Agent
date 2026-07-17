"""
Memory tools for episodic feedback storage and retrieval.
Enables self-learning from user corrections without LLM fine-tuning.
"""

from typing import Optional

from langchain_core.tools import tool
from app.db.supabase_client import supabase
from app.tools.embedding_tool import embed_text


@tool
def save_user_feedback(
    query: str,
    response: str,
    is_positive: bool,
    correction: Optional[str] = None
) -> str:
    """
    Stores a Presales staff member's feedback or correction into the database,
    to serve as the AI's learning memory.
    
    Args:
        query: The user's original question.
        response: The system-generated response that was evaluated.
        is_positive: Whether the feedback is positive (True) or negative (False).
        correction: The correct information provided by the human (only needed when is_positive=False).
        
    Returns:
        Success/error message.
    """
    try:
        embedding = embed_text(query)
        data = {
            "user_query": query,
            "system_response": response,
            "is_positive": is_positive,
            "user_correction": correction,
            "query_embedding": embedding
        }
        supabase.table("cloud_feedback_memory").insert(data).execute()
        return "Feedback recorded successfully! The system will remember this to self-correct on similar future questions."
    except Exception as e:
        return f"Error saving feedback: {str(e)}"


def retrieve_past_experiences(
    query: str,
    similarity_threshold: float = 0.75,
    match_count: int = 2
) -> str:
    """
    Internal function (NOT a @tool — called directly by the Supervisor, not via LLM function-calling)
    that searches for similar past lessons learned based on vector distance.
    
    Reason it's not left to the LLM to decide: the experience-retrieval step MUST always run before
    routing, regardless of whether the LLM "thinks" to call this tool or not.
    
    Args:
        query: The user's question to search for similar past experiences.
        similarity_threshold: Minimum similarity score (0-1) to retrieve.
        match_count: Maximum number of past experiences to retrieve.
        
    Returns:
        Formatted context string with lessons learned, or empty string if none found.
    """
    try:
        query_embedding = embed_text(query)
        response = supabase.rpc(
            "match_past_experience",
            {
                "query_embedding": query_embedding,
                "match_threshold": similarity_threshold,
                "match_count": match_count
            }
        ).execute()

        experiences = response.data
        if not experiences:
            return ""

        context_experience = "### LESSONS LEARNED FROM PAST HUMAN FEEDBACK:\n"
        found_any = False
        for idx, exp in enumerate(experiences):
            if not exp["is_positive"] and exp["user_correction"]:
                found_any = True
                context_experience += (
                    f"Case {idx+1}:\n"
                    f"- Similar past question: '{exp['user_query']}'\n"
                    f"- AI previously answered incorrectly: '{exp['system_response']}'\n"
                    f"- A Presales expert corrected it to: '{exp['user_correction']}' -> YOU MUST USE THIS INFORMATION.\n"
                )
        return context_experience if found_any else ""
    except Exception as e:
        print(f"Error retrieving experience memory: {str(e)}")
        return ""
