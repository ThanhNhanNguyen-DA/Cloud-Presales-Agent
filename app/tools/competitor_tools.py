"""
Competitor mapping tools for looking up AWS/Azure/GCP service equivalents.
Uses direct SQL filtering for structured data (no vector search needed).
"""

from langchain_core.tools import tool
from app.db.supabase_client import supabase


@tool
def find_competitor_mapping(
    competitor_name: str | None = None,
    competitor_service_name: str | None = None
) -> str:
    """
    Finds the local service equivalent to a competitor's service (AWS, Azure, GCP).
    
    Args:
        competitor_name: Name of the competitor provider ('AWS', 'Azure', 'GCP'). Can be left blank if unknown.
        competitor_service_name: Specific competitor service name (e.g. 'S3', 'EC2', 'RDS', 'Blob Storage'). Should be provided for the most accurate lookup.
        
    Returns:
        Formatted mapping results showing local service equivalent and migration difficulty.
    """
    query = supabase.table("cloud_competitor_mapping").select("*")

    if competitor_name:
        query = query.ilike("competitor_name", f"%{competitor_name}%")
    if competitor_service_name:
        query = query.ilike("competitor_service_name", f"%{competitor_service_name}%")

    response = query.execute()
    results = response.data

    if not results:
        return "No matching competitor service found in the mapping database. Please provide more details or check the service name."

    formatted = "### COMPETITOR SERVICE MAPPING RESULTS:\n"
    for idx, row in enumerate(results):
        formatted += (
            f"\n[{idx+1}] {row['competitor_name']} {row['competitor_service_name']} "
            f"↔ {row['local_service_name']}\n"
            f"Feature comparison: {row['feature_comparison'] or 'No detailed description available'}\n"
            f"Migration difficulty: {row['migration_difficulty'] or 'Not determined'}\n"
        )
    return formatted


@tool
def list_all_competitor_services(competitor_name: str) -> str:
    """
    Lists all services of a specific competitor that already exist in the mapping matrix.
    
    Args:
        competitor_name: Name of the competitor provider ('AWS', 'Azure', 'GCP').
        
    Returns:
        Formatted list of all mapped services for the competitor.
    """
    response = supabase.table("cloud_competitor_mapping").select("*").ilike(
        "competitor_name", f"%{competitor_name}%"
    ).execute()
    results = response.data

    if not results:
        return f"No mapping data exists yet for {competitor_name}."

    formatted = f"### LIST OF {competitor_name.upper()} SERVICES ALREADY MAPPED:\n"
    for row in results:
        formatted += f"- {row['competitor_service_name']} ↔ {row['local_service_name']}\n"
    return formatted
