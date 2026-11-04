"""
Pytest tests for presales calculation tools.
Tests all 3 tools with normal and edge cases.
"""

import pytest
from app.tools.presales_tools import (
    calculate_backup_and_cost,
    size_web_app_vm,
    size_database_storage
)


class TestCalculateBackupAndCost:
    """Tests for calculate_backup_and_cost tool."""

    def test_backup_normal_case(self):
        """Test backup calculation with normal inputs."""
        result = calculate_backup_and_cost.invoke({
            "data_volume_gb": 10,
            "bandwidth_mbps": 100,
            "storage_class": "Standard"
        })
        assert result["calculated_monthly_cost_usd"] > 0
        assert result["estimated_hours"] > 0
        assert result["estimated_minutes"] > 0
        assert result["storage_class_used"] == "Standard"
        assert result["network_efficiency"] == "80%"
        # 10 GB * 0.023 USD/GB = 0.23 USD
        assert result["calculated_monthly_cost_usd"] == 0.23

    def test_backup_edge_case_small_bandwidth(self):
        """Test backup with very small bandwidth (slow connection)."""
        result = calculate_backup_and_cost.invoke({
            "data_volume_gb": 5,
            "bandwidth_mbps": 1,
            "storage_class": "Cool"
        })
        assert result["calculated_monthly_cost_usd"] > 0
        assert result["estimated_hours"] > 0
        # 5 GB * 0.012 USD/GB = 0.06 USD
        assert result["calculated_monthly_cost_usd"] == 0.06
        assert result["storage_class_used"] == "Cool"

    def test_backup_archive_storage(self):
        """Test backup with Archive storage tier."""
        result = calculate_backup_and_cost.invoke({
            "data_volume_gb": 100,
            "bandwidth_mbps": 500,
            "storage_class": "Archive"
        })
        assert result["calculated_monthly_cost_usd"] > 0
        assert result["storage_class_used"] == "Archive"
        # 100 GB * 0.004 USD/GB = 0.4 USD
        assert result["calculated_monthly_cost_usd"] == 0.4

    def test_backup_large_volume(self):
        """Test backup with large data volume."""
        result = calculate_backup_and_cost.invoke({
            "data_volume_gb": 1000,
            "bandwidth_mbps": 200,
            "storage_class": "Standard"
        })
        assert result["calculated_monthly_cost_usd"] == 23.0
        assert result["estimated_hours"] > 0


class TestSizeWebAppVm:
    """Tests for size_web_app_vm tool."""

    def test_sizing_normal_case_java(self):
        """Test VM sizing for Java app with normal CCU."""
        result = size_web_app_vm.invoke({
            "ccu": 100,
            "app_type": "java"
        })
        assert result["app_type"] == "java"
        assert result["target_ccu"] == 100
        assert result["recommended_vcpu"] >= 2
        assert result["recommended_ram_gb"] >= 4
        assert "java" in result["sizing_note"]

    def test_sizing_zero_ccu(self):
        """Test VM sizing with zero concurrent users (edge case)."""
        result = size_web_app_vm.invoke({
            "ccu": 0,
            "app_type": "java"
        })
        # Should still return minimum: 2 vCPU, 4 GB RAM
        assert result["recommended_vcpu"] >= 2
        assert result["recommended_ram_gb"] >= 4

    def test_sizing_nodejs_runtime(self):
        """Test VM sizing for Node.js runtime (more efficient)."""
        result = size_web_app_vm.invoke({
            "ccu": 100,
            "app_type": "nodejs"
        })
        assert result["app_type"] == "nodejs"
        assert result["target_ccu"] == 100
        assert result["recommended_vcpu"] >= 2
        assert result["recommended_ram_gb"] >= 4

    def test_sizing_python_runtime(self):
        """Test VM sizing for Python runtime."""
        result = size_web_app_vm.invoke({
            "ccu": 200,
            "app_type": "python"
        })
        assert result["app_type"] == "python"
        assert result["target_ccu"] == 200
        assert result["recommended_vcpu"] >= 2
        assert result["recommended_ram_gb"] >= 4

    def test_sizing_php_runtime(self):
        """Test VM sizing for PHP runtime."""
        result = size_web_app_vm.invoke({
            "ccu": 150,
            "app_type": "php"
        })
        assert result["app_type"] == "php"
        assert result["recommended_vcpu"] >= 2
        assert result["recommended_ram_gb"] >= 4

    def test_sizing_large_ccu(self):
        """Test VM sizing with large number of concurrent users."""
        result = size_web_app_vm.invoke({
            "ccu": 1000,
            "app_type": "java"
        })
        assert result["recommended_vcpu"] >= 2
        assert result["recommended_ram_gb"] >= 4
        # With large CCU, sizing should scale up significantly
        assert result["recommended_vcpu"] > 2 or result["recommended_ram_gb"] > 4


class TestSizeDatabaseStorage:
    """Tests for size_database_storage tool."""

    def test_storage_sizing_normal_case(self):
        """Test database storage sizing with normal inputs."""
        result = size_database_storage.invoke({
            "current_size_gb": 100,
            "monthly_growth_rate_percent": 10,
            "months": 12,
            "required_iops": 2000
        })
        assert result["current_size_gb"] == 100
        assert result["projected_size_gb_after_months"] > 100
        assert result["months_projected"] == 12
        assert result["required_iops"] == 2000
        assert result["storage_tier_recommendation"] == "Standard"
        assert "SATA SSD" in result["storage_recommendation_detail"]

    def test_storage_sizing_zero_growth(self):
        """Test database storage with zero growth rate (edge case)."""
        result = size_database_storage.invoke({
            "current_size_gb": 50,
            "monthly_growth_rate_percent": 0,
            "months": 24,
            "required_iops": 1000
        })
        # With 0% growth, size should remain the same
        assert result["projected_size_gb_after_months"] == 50.0
        assert result["storage_tier_recommendation"] == "Standard"

    def test_storage_sizing_high_iops(self):
        """Test database storage with high IOPS requirement."""
        result = size_database_storage.invoke({
            "current_size_gb": 200,
            "monthly_growth_rate_percent": 15,
            "months": 6,
            "required_iops": 15000
        })
        assert result["required_iops"] == 15000
        assert result["storage_tier_recommendation"] == "Dedicated/Provisioned"
        assert "Dedicated Storage" in result["storage_recommendation_detail"]

    def test_storage_sizing_medium_iops(self):
        """Test database storage with medium IOPS (3000-10000 range)."""
        result = size_database_storage.invoke({
            "current_size_gb": 150,
            "monthly_growth_rate_percent": 5,
            "months": 12,
            "required_iops": 5000
        })
        assert result["required_iops"] == 5000
        assert result["storage_tier_recommendation"] == "High-Performance"
        assert "NVMe SSD" in result["storage_recommendation_detail"]

    def test_storage_sizing_rapid_growth(self):
        """Test database storage with rapid growth rate."""
        result = size_database_storage.invoke({
            "current_size_gb": 10,
            "monthly_growth_rate_percent": 50,
            "months": 12,
            "required_iops": 2000
        })
        # 10 * (1.5^12) = 10 * ~129.75 ≈ 1297.5 GB
        assert result["projected_size_gb_after_months"] > 100
        assert result["storage_tier_recommendation"] == "Standard"

    def test_storage_sizing_exact_iops_threshold(self):
        """Test IOPS thresholds at exact boundary values."""
        # Test at boundary: 3000 IOPS
        result_boundary = size_database_storage.invoke({
            "current_size_gb": 100,
            "monthly_growth_rate_percent": 10,
            "months": 6,
            "required_iops": 3000
        })
        assert result_boundary["storage_tier_recommendation"] == "High-Performance"

        # Test just below boundary: 2999 IOPS
        result_below = size_database_storage.invoke({
            "current_size_gb": 100,
            "monthly_growth_rate_percent": 10,
            "months": 6,
            "required_iops": 2999
        })
        assert result_below["storage_tier_recommendation"] == "Standard"


class TestQuoteSafety:
    """
    Regression tests for the Day-1 fixes: these numbers go into customer quotes, so a
    tool must never price one tier while labelling it another, or crash on bad input.
    """

    @pytest.mark.parametrize("given, canonical, price", [
        ("cool", "Cool", 0.12),
        ("ARCHIVE", "Archive", 0.04),
        ("  standard ", "Standard", 0.23),
    ])
    def test_storage_class_is_case_insensitive(self, given, canonical, price):
        result = calculate_backup_and_cost.invoke(
            {"data_volume_gb": 10, "bandwidth_mbps": 100, "storage_class": given})
        assert result["storage_class_used"] == canonical
        assert result["calculated_monthly_cost_usd"] == price

    def test_unknown_storage_class_returns_error_not_standard_price(self):
        result = calculate_backup_and_cost.invoke(
            {"data_volume_gb": 10, "bandwidth_mbps": 100, "storage_class": "Glacier"})
        assert "error" in result
        assert "calculated_monthly_cost_usd" not in result
        assert "Standard, Cool, Archive" in result["error"]

    def test_backup_duration_uses_decimal_units(self):
        # 10 GB = 8e10 bit; 100 Mbps * 80% = 8e7 bit/s -> 1000 s = 16.7 min.
        result = calculate_backup_and_cost.invoke({"data_volume_gb": 10, "bandwidth_mbps": 100})
        assert result["estimated_minutes"] == 16.7
        assert result["estimated_hours"] == 0.28

    @pytest.mark.parametrize("volume, bandwidth", [
        (10, 0), (10, -5), (0, 100), (-1, 100), (float("nan"), 100), (10, float("inf")),
    ])
    def test_backup_rejects_invalid_numbers(self, volume, bandwidth):
        result = calculate_backup_and_cost.invoke(
            {"data_volume_gb": volume, "bandwidth_mbps": bandwidth})
        assert "error" in result

    @pytest.mark.parametrize("given, canonical", [
        ("Java", "java"), ("NodeJS", "nodejs"), ("node.js", "nodejs"), ("PY", "python"),
    ])
    def test_app_type_is_case_insensitive_with_aliases(self, given, canonical):
        result = size_web_app_vm.invoke({"ccu": 100, "app_type": given})
        assert result["app_type"] == canonical

    def test_unknown_app_type_returns_error_not_java(self):
        result = size_web_app_vm.invoke({"ccu": 100, "app_type": "golang"})
        assert "error" in result
        assert "recommended_vcpu" not in result

    def test_negative_ccu_returns_error(self):
        assert "error" in size_web_app_vm.invoke({"ccu": -10, "app_type": "java"})

    @pytest.mark.parametrize("args", [
        {"current_size_gb": 0, "monthly_growth_rate_percent": 10, "months": 12, "required_iops": 100},
        {"current_size_gb": 100, "monthly_growth_rate_percent": -5, "months": 12, "required_iops": 100},
        {"current_size_gb": 100, "monthly_growth_rate_percent": 10, "months": -1, "required_iops": 100},
        {"current_size_gb": 100, "monthly_growth_rate_percent": 10, "months": 121, "required_iops": 100},
        {"current_size_gb": 100, "monthly_growth_rate_percent": 10, "months": 12, "required_iops": -1},
    ])
    def test_database_sizing_rejects_invalid_inputs(self, args):
        assert "error" in size_database_storage.invoke(args)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
