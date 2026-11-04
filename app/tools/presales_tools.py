"""
Core presales calculation tools for Cloud Presales Agent.
These are pure calculation functions decorated with @tool for LangChain integration.

These numbers go straight into customer quotes, so the tools never guess silently:
- option names (storage class, runtime) are matched case-insensitively, and an unknown
  value returns an `error` instead of quietly falling back to a default;
- inputs are validated (positive / finite), and invalid ones return an `error` the agent
  can relay to the user ("please give the bandwidth") instead of raising;
- units follow the networking/billing convention: 1 GB = 10^9 bytes, 1 Mbps = 10^6 bit/s.
"""

import math

from langchain_core.tools import tool

# USD per GB-month by storage tier (keys are canonical names).
STORAGE_PRICING_USD_PER_GB = {"Standard": 0.023, "Cool": 0.012, "Archive": 0.004}
# Share of the nominal bandwidth a long transfer achieves in practice (TCP/protocol overhead).
NETWORK_EFFICIENCY = 0.8

VM_SPECS = {
    "java": {"cpu_per_ccu": 0.008, "ram_mb_per_ccu": 16, "base_ram_gb": 2.0},
    "nodejs": {"cpu_per_ccu": 0.005, "ram_mb_per_ccu": 8, "base_ram_gb": 1.0},
    "php": {"cpu_per_ccu": 0.006, "ram_mb_per_ccu": 10, "base_ram_gb": 1.0},
    "python": {"cpu_per_ccu": 0.007, "ram_mb_per_ccu": 12, "base_ram_gb": 1.5},
}
# Common spellings the LLM may pass for a runtime.
RUNTIME_ALIASES = {"node": "nodejs", "node.js": "nodejs", "js": "nodejs", "py": "python"}

MAX_PROJECTION_MONTHS = 120


def _error(message: str) -> dict:
    """Uniform error payload: the agent shows it / asks the user instead of inventing a number."""
    return {"error": message}


def _check_number(name: str, value: float, *, minimum: float = 0.0, strict: bool = True) -> str | None:
    """Return an error message if `value` is not a finite number above (or at) `minimum`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return f"{name} must be a finite number (got {value!r})."
    if strict and value <= minimum:
        return f"{name} must be greater than {minimum:g} (got {value:g})."
    if not strict and value < minimum:
        return f"{name} must be at least {minimum:g} (got {value:g})."
    return None


def _match_option(value: str, options, aliases: dict | None = None) -> str | None:
    """Case/whitespace-insensitive lookup of `value` among `options`; returns the canonical key."""
    key = (value or "").strip().lower()
    key = (aliases or {}).get(key, key)
    for option in options:
        if option.lower() == key:
            return option
    return None


@tool
def calculate_backup_and_cost(
    data_volume_gb: float,
    bandwidth_mbps: float,
    storage_class: str = "Standard"
) -> dict:
    """
    Calculates the estimated backup duration to the Cloud and the monthly storage cost.

    Args:
        data_volume_gb: Data volume to be backed up (GB, must be > 0).
        bandwidth_mbps: Network bandwidth (Mbps, must be > 0).
        storage_class: Storage tier: 'Standard', 'Cool' or 'Archive' (case-insensitive).

    Returns:
        Dictionary with backup duration (hours/minutes) and monthly cost estimate,
        or {"error": ...} if an input is invalid.
    """
    for name, value in (("data_volume_gb", data_volume_gb), ("bandwidth_mbps", bandwidth_mbps)):
        problem = _check_number(name, value)
        if problem:
            return _error(problem)

    tier = _match_option(storage_class, STORAGE_PRICING_USD_PER_GB)
    if tier is None:
        valid = ", ".join(STORAGE_PRICING_USD_PER_GB)
        return _error(f"Unknown storage_class {storage_class!r}. Valid options: {valid}.")

    monthly_cost = data_volume_gb * STORAGE_PRICING_USD_PER_GB[tier]

    total_bits = data_volume_gb * 8 * 10**9
    effective_bandwidth_bps = bandwidth_mbps * 10**6 * NETWORK_EFFICIENCY
    duration_hours = total_bits / effective_bandwidth_bps / 3600

    return {
        "calculated_monthly_cost_usd": round(monthly_cost, 2),
        "price_per_gb_month_usd": STORAGE_PRICING_USD_PER_GB[tier],
        "estimated_hours": round(duration_hours, 2),
        "estimated_minutes": round(duration_hours * 60, 1),
        "storage_class_used": tier,
        "network_efficiency": f"{NETWORK_EFFICIENCY:.0%}",
        "unit_convention": "1 GB = 10^9 bytes, 1 Mbps = 10^6 bit/s",
    }


@tool
def size_web_app_vm(
    ccu: int,
    app_type: str = "java"
) -> dict:
    """
    Recommends the optimal VM configuration (vCPU, RAM) based on the number of concurrent users (CCU).

    Args:
        ccu: Number of concurrent users accessing the system at once (>= 0).
        app_type: Application runtime: 'java', 'nodejs', 'php' or 'python' (case-insensitive).

    Returns:
        Dictionary with recommended vCPU and RAM configuration, or {"error": ...}.
    """
    problem = _check_number("ccu", ccu, strict=False)
    if problem:
        return _error(problem)

    runtime = _match_option(app_type, VM_SPECS, RUNTIME_ALIASES)
    if runtime is None:
        valid = ", ".join(VM_SPECS)
        return _error(f"Unknown app_type {app_type!r}. Valid options: {valid}.")

    config = VM_SPECS[runtime]
    raw_cpu = ccu * config["cpu_per_ccu"]
    raw_ram = (ccu * config["ram_mb_per_ccu"]) / 1024 + config["base_ram_gb"]

    recommended_vcpu = max(2, int(math.ceil(raw_cpu / 2) * 2))
    recommended_ram_gb = max(4, int(math.ceil(raw_ram / 4) * 4))

    return {
        "app_type": runtime,
        "target_ccu": ccu,
        "recommended_vcpu": recommended_vcpu,
        "recommended_ram_gb": recommended_ram_gb,
        "sizing_note": f"Recommendation based on the runtime load of the {runtime} engine/runtime."
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
        current_size_gb: Current data size (GB, > 0).
        monthly_growth_rate_percent: Monthly growth rate as a percentage (e.g. 15 means 15%/month; 0 or more).
        months: Number of months to project (0-120).
        required_iops: Required IOPS for the application's read/write workload (>= 0).

    Returns:
        Dictionary with projected size and storage tier recommendation, or {"error": ...}.
    """
    checks = (
        _check_number("current_size_gb", current_size_gb),
        _check_number("monthly_growth_rate_percent", monthly_growth_rate_percent, strict=False),
        _check_number("months", months, strict=False),
        _check_number("required_iops", required_iops, strict=False),
    )
    problem = next((c for c in checks if c), None)
    if problem:
        return _error(problem)
    if months > MAX_PROJECTION_MONTHS:
        return _error(f"months must be at most {MAX_PROJECTION_MONTHS} (got {months:g}).")

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
