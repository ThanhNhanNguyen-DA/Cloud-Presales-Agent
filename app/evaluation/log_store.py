"""
Read-side of the interaction log — deliberately lightweight.

Kept separate from `interaction_log.py` (the write side) so the Streamlit monitoring
dashboard can import it WITHOUT pulling in the RAGAS evaluator and, through it, the
local embedding model (~1.2 GB). This module imports only the Supabase client.
"""

from typing import List

from app.db.supabase_client import supabase


def load_recent_logs(limit: int = 500) -> List[dict]:
    """
    Fetch the most recent interaction logs (newest first) for the monitoring dashboard.

    Returns [] on any error so the dashboard degrades gracefully when the table is empty,
    not yet migrated, or unreachable.
    """
    try:
        response = (
            supabase.table("cloud_interaction_logs")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data or []
    except Exception as e:  # noqa: BLE001
        print(f"Error loading interaction logs: {str(e)}")
        return []
