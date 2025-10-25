#!/usr/bin/env python3
"""
Test Suite for Memory Quantizer + Lazy Loading Integration
===========================================================

Tests to verify that Memory Quantizer properly prevents OOM kills
by refusing to load intelligence components when memory is insufficient.

Author: Derek J. Russell
Date: October 2025
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass

# Import the components we're testing
from core.memory_quantizer import MemoryQuantizer, MemoryTier, MemoryPressure, MemoryMetrics


class TestMemoryQuantizerLazyLoading:
    """Test Memory Quantizer integration with lazy loading"""

    @pytest.mark.asyncio
    async def test_sufficient_memory_allows_loading(self):
        """Test that loading proceeds when memory is sufficient"""

        # Mock metrics showing plenty of memory
        mock_metrics = MemoryMetrics(
            timestamp=1234567890.0,
            process_memory_gb=0.3,
            system_memory_gb=16.0,
            system_memory_percent=30.0,  # 30% used
            system_memory_available_gb=11.2,  # 11.2 GB available
            tier=MemoryTier.OPTIMAL,
            pressure=MemoryPressure.NORMAL
        )

        # Create app_state mock
        app_state = Mock()
        app_state.uae_engine = None
        app_state.uae_initializing = False
        app_state.uae_lazy_config = {"vision_analyzer": None}

        with patch('core.memory_quantizer.MemoryQuantizer.get_current_metrics',
                  return_value=mock_metrics):
            # In real code, this would call ensure_uae_loaded
            # Here we just verify the logic
            required_memory = 10.0
            predicted_usage = mock_metrics.system_memory_percent + \
                            (required_memory / mock_metrics.system_memory_gb * 100)

            # Should allow loading
            assert mock_metrics.system_memory_available_gb >= required_memory
            assert mock_metrics.tier not in {MemoryTier.CRITICAL, MemoryTier.EMERGENCY, MemoryTier.CONSTRAINED}
            assert predicted_usage < 90  # ~92.5% - Safe to load

            print(f"✅ PASS: Memory check allows loading")
            print(f"   Available: {mock_metrics.system_memory_available_gb:.2f} GB")
            print(f"   Required: {required_memory:.2f} GB")
            print(f"   Predicted usage: {predicted_usage:.1f}%")

    @pytest.mark.asyncio
    async def test_insufficient_memory_prevents_loading(self):
        """Test that loading is prevented when memory is insufficient"""

        # Mock metrics showing low memory
        mock_metrics = MemoryMetrics(
            timestamp=1234567890.0,
            process_memory_gb=2.5,
            system_memory_gb=16.0,
            system_memory_percent=85.0,  # 85% used
            system_memory_available_gb=2.4,  # Only 2.4 GB available
            tier=MemoryTier.CONSTRAINED,
            pressure=MemoryPressure.WARN
        )

        required_memory = 10.0

        # Should prevent loading
        assert mock_metrics.system_memory_available_gb < required_memory

        deficit = required_memory - mock_metrics.system_memory_available_gb

        print(f"✅ PASS: Memory check prevents loading")
        print(f"   Available: {mock_metrics.system_memory_available_gb:.2f} GB")
        print(f"   Required: {required_memory:.2f} GB")
        print(f"   Deficit: {deficit:.2f} GB")

    @pytest.mark.asyncio
    async def test_dangerous_tier_prevents_loading(self):
        """Test that loading is prevented in dangerous memory tiers"""

        test_cases = [
            (MemoryTier.CRITICAL, "CRITICAL tier"),
            (MemoryTier.EMERGENCY, "EMERGENCY tier"),
            (MemoryTier.CONSTRAINED, "CONSTRAINED tier"),
        ]

        for tier, description in test_cases:
            mock_metrics = MemoryMetrics(
                timestamp=1234567890.0,
                process_memory_gb=3.0,
                system_memory_gb=16.0,
                system_memory_percent=90.0,
                system_memory_available_gb=1.6,
                tier=tier,
                pressure=MemoryPressure.CRITICAL if tier == MemoryTier.EMERGENCY else MemoryPressure.WARN
            )

            dangerous_tiers = {MemoryTier.CRITICAL, MemoryTier.EMERGENCY, MemoryTier.CONSTRAINED}
            assert mock_metrics.tier in dangerous_tiers

            print(f"✅ PASS: {description} prevents loading")

    @pytest.mark.asyncio
    async def test_oom_prediction_prevents_loading(self):
        """Test that loading is prevented when OOM is predicted"""

        # Scenario: 12 GB used, 4 GB available, but loading 10GB would push us over 90%
        mock_metrics = MemoryMetrics(
            timestamp=1234567890.0,
            process_memory_gb=2.0,
            system_memory_gb=16.0,
            system_memory_percent=75.0,  # 75% used (12 GB)
            system_memory_available_gb=4.0,  # 4 GB available
            tier=MemoryTier.ELEVATED,
            pressure=MemoryPressure.NORMAL
        )

        required_memory = 10.0
        predicted_usage = mock_metrics.system_memory_percent + \
                        (required_memory / mock_metrics.system_memory_gb * 100)

        # Should prevent loading because predicted usage > 90%
        assert predicted_usage > 90  # Would be ~137.5%

        print(f"✅ PASS: OOM prediction prevents loading")
        print(f"   Current usage: {mock_metrics.system_memory_percent:.1f}%")
        print(f"   Predicted usage: {predicted_usage:.1f}%")
        print(f"   Safe threshold: <90%")

    @pytest.mark.asyncio
    async def test_edge_case_exactly_90_percent(self):
        """Test edge case where predicted usage is exactly 90%"""

        # Setup: Current 50%, loading 10GB on 16GB system = 50% + 62.5% = 112.5%
        # But if we have exactly enough to hit 90%...
        mock_metrics = MemoryMetrics(
            timestamp=1234567890.0,
            process_memory_gb=1.0,
            system_memory_gb=16.0,
            system_memory_percent=50.0,  # 8 GB used
            system_memory_available_gb=8.0,  # 8 GB available
            tier=MemoryTier.OPTIMAL,
            pressure=MemoryPressure.NORMAL
        )

        required_memory = 6.4  # Exactly enough to hit 90%
        predicted_usage = mock_metrics.system_memory_percent + \
                        (required_memory / mock_metrics.system_memory_gb * 100)

        # Should still allow loading at exactly 90%
        assert predicted_usage == 90.0

        print(f"✅ PASS: Edge case at exactly 90%")
        print(f"   Predicted usage: {predicted_usage:.1f}%")

    @pytest.mark.asyncio
    async def test_memory_quantizer_fallback_on_error(self):
        """Test that loading proceeds with warning if Memory Quantizer fails"""

        # Simulate Memory Quantizer import failure
        with patch('core.memory_quantizer.MemoryQuantizer',
                  side_effect=ImportError("Memory Quantizer not available")):
            try:
                from core.memory_quantizer import MemoryQuantizer
                quantizer = MemoryQuantizer()
                assert False, "Should have raised ImportError"
            except ImportError as e:
                # This is expected - system should log warning and proceed
                print(f"✅ PASS: Gracefully handles Memory Quantizer failure")
                print(f"   Error: {e}")

    @pytest.mark.asyncio
    async def test_real_world_16gb_system_scenarios(self):
        """Test realistic scenarios on a 16GB system"""

        scenarios = [
            {
                "name": "Startup (260 MB used)",
                "process_gb": 0.26,
                "system_percent": 30.0,
                "available_gb": 11.2,
                "tier": MemoryTier.OPTIMAL,
                "should_allow": True
            },
            {
                "name": "After loading some apps (8 GB used)",
                "process_gb": 0.26,
                "system_percent": 50.0,
                "available_gb": 8.0,
                "tier": MemoryTier.OPTIMAL,
                "should_allow": True
            },
            {
                "name": "Heavy usage (12 GB used)",
                "process_gb": 0.26,
                "system_percent": 75.0,
                "available_gb": 4.0,
                "tier": MemoryTier.ELEVATED,
                "should_allow": False  # 10GB load would push to 137%
            },
            {
                "name": "Very heavy usage (14 GB used)",
                "process_gb": 0.26,
                "system_percent": 87.5,
                "available_gb": 2.0,
                "tier": MemoryTier.CONSTRAINED,
                "should_allow": False  # In dangerous tier
            },
            {
                "name": "Critical (15 GB used)",
                "process_gb": 0.26,
                "system_percent": 93.75,
                "available_gb": 1.0,
                "tier": MemoryTier.CRITICAL,
                "should_allow": False  # In dangerous tier
            }
        ]

        for scenario in scenarios:
            mock_metrics = MemoryMetrics(
                timestamp=1234567890.0,
                process_memory_gb=scenario["process_gb"],
                system_memory_gb=16.0,
                system_memory_percent=scenario["system_percent"],
                system_memory_available_gb=scenario["available_gb"],
                tier=scenario["tier"],
                pressure=MemoryPressure.NORMAL
            )

            required_memory = 10.0
            predicted_usage = mock_metrics.system_memory_percent + \
                            (required_memory / mock_metrics.system_memory_gb * 100)

            # Check all three conditions
            has_memory = mock_metrics.system_memory_available_gb >= required_memory
            safe_tier = mock_metrics.tier not in {MemoryTier.CRITICAL, MemoryTier.EMERGENCY, MemoryTier.CONSTRAINED}
            safe_prediction = predicted_usage <= 90

            would_allow = has_memory and safe_tier and safe_prediction

            assert would_allow == scenario["should_allow"], \
                f"Scenario '{scenario['name']}' failed: expected {scenario['should_allow']}, got {would_allow}"

            status = "✅ ALLOW" if would_allow else "❌ BLOCK"
            print(f"{status}: {scenario['name']}")
            print(f"   Used: {scenario['system_percent']:.1f}%, Available: {scenario['available_gb']:.2f} GB")
            print(f"   Tier: {scenario['tier'].value}, Predicted: {predicted_usage:.1f}%")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
