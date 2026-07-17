"""
Core presales calculation tools for Cloud Presales Agent.
These are pure calculation functions decorated with @tool for LangChain integration.
No LangGraph dependencies in this phase.
"""

from langchain_core.tools import tool
import math


@tool
def calculate_backup_and_cost(
    data_volume_gb: float,
    bandwidth_mbps: float,
    storage_class: str = "Standard"
) -> dict:
    """
    Calculates the estimated backup duration to the Cloud and the monthly storage cost.
    
    Args:
        data_volume_gb: Data volume to be backed up (GB).
        bandwidth_mbps: Network bandwidth (Mbps).
        storage_class: Storage tier ('Standard', 'Cool', 'Archive').
    
    Returns:
        Dictionary with backup duration (hours/minutes) and monthly cost estimate.
    """
    pricing = {"Standard": 0.023, "Cool": 0.012, "Archive": 0.004}
    price_per_gb = pricing.get(storage_class, 0.023)
    monthly_cost = data_volume_gb * price_per_gb

    total_bits = data_volume_gb * 8 * 1024 * 1024 * 1024
    effective_bandwidth_bps = bandwidth_mbps * 1024 * 1024 * 0.8

    duration_seconds = total_bits / effective_bandwidth_bps
    duration_hours = duration_seconds / 3600

    return {
        "calculated_monthly_cost_usd": round(monthly_cost, 2),
        "estimated_hours": round(duration_hours, 2),
        "estimated_minutes": round(duration_hours * 60, 1),
        "storage_class_used": storage_class,
        "network_efficiency": "80%"
    }


@tool
def size_web_app_vm(
    ccu: int,
    app_type: str = "java"
) -> dict:
    """
    Recommends the optimal VM configuration (vCPU, RAM) based on the number of concurrent users (CCU).
    
    Args:
        ccu: Number of concurrent users accessing the system at once.
        app_type: Application runtime type ('java', 'nodejs', 'php', 'python').
    
    Returns:
        Dictionary with recommended vCPU and RAM configuration.
    """
    specs = {
        "java": {"cpu_per_ccu": 0.008, "ram_mb_per_ccu": 16, "base_ram_gb": 2.0},
        "nodejs": {"cpu_per_ccu": 0.005, "ram_mb_per_ccu": 8, "base_ram_gb": 1.0},
        "php": {"cpu_per_ccu": 0.006, "ram_mb_per_ccu": 10, "base_ram_gb": 1.0},
        "python": {"cpu_per_ccu": 0.007, "ram_mb_per_ccu": 12, "base_ram_gb": 1.5}
    }

    config = specs.get(app_type.lower(), specs["java"])

    raw_cpu = ccu * config["cpu_per_ccu"]
    raw_ram = (ccu * config["ram_mb_per_ccu"]) / 1024 + config["base_ram_gb"]

    recommended_vcpu = max(2, int(math.ceil(raw_cpu / 2) * 2))
    recommended_ram_gb = max(4, int(math.ceil(raw_ram / 4) * 4))

    return {
        "app_type": app_type,
        "target_ccu": ccu,
        "recommended_vcpu": recommended_vcpu,
        "recommended_ram_gb": recommended_ram_gb,
        "sizing_note": f"Recommendation based on the runtime load of the {app_type} engine/runtime."
    }


@tool
def size_database_storage(
    current_size_gb: float,
    monthly_growth_rate_percent: float,
    months: int,
    required_iops: int
) -> dict:
    """
    Estimates the projected database size after N months of growth and recommends a suitable
    storage tier based on the required IOPS.
    
    Args:
        current_size_gb: Current data size (GB).
        monthly_growth_rate_percent: Monthly growth rate as a percentage (e.g. 15 means 15%/month).
        months: Number of months to project.
        required_iops: Required IOPS for the application's read/write workload.
    
    Returns:
        Dictionary with projected size and storage tier recommendation.
    """
    growth_rate = monthly_growth_rate_percent / 100
    projected_size_gb = current_size_gb * ((1 + growth_rate) ** months)

    if required_iops < 3000:
        storage_recommendation = "SATA SSD or Standard Cloud Storage"
        tier = "Standard"
    elif required_iops < 10000:
        storage_recommendation = "High-Performance NVMe SSD"
        tier = "High-Performance"
    else:
        storage_recommendation = "Dedicated Storage or IOPS Provisioned Volume"
        tier = "Dedicated/Provisioned"

    return {
        "current_size_gb": current_size_gb,
        "projected_size_gb_after_months": round(projected_size_gb, 2),
        "months_projected": months,
        "required_iops": required_iops,
        "storage_tier_recommendation": tier,
        "storage_recommendation_detail": storage_recommendation
    }
