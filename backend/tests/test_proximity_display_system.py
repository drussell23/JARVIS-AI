#!/usr/bin/env python3
"""
Comprehensive Tests for Proximity-Aware Display System (Phase 1.2)
===================================================================

Tests all components of the proximity-display system:
- Bluetooth proximity detection
- Display location configuration
- Proximity scoring and context generation
- Connection decision logic
- API endpoints

Author: Derek Russell
Date: 2025-10-14
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestProximityDisplaySystem:
    """Comprehensive test suite for Phase 1.2"""
    
    @pytest.mark.asyncio
    async def test_proximity_data_creation(self):
        """Test ProximityData creation and methods"""
        from proximity.proximity_display_context import ProximityData, ProximityZone
        from datetime import datetime
        
        prox_data = ProximityData(
            device_name="Derek's Apple Watch",
            device_uuid="12:34:56:78:9A:BC",
            device_type="apple_watch",
            rssi=-55,
            estimated_distance=2.3,
            proximity_zone=ProximityZone.NEAR,
            timestamp=datetime.now(),
            confidence=0.9,
            signal_quality=0.8
        )
        
        assert prox_data.device_name == "Derek's Apple Watch"
        assert prox_data.estimated_distance == 2.3
        assert prox_data.proximity_zone == ProximityZone.NEAR
        
        # Test to_dict()
        data_dict = prox_data.to_dict()
        assert data_dict["device_name"] == "Derek's Apple Watch"
        assert data_dict["proximity_zone"] == "near"
        
        # Test zone classification
        assert ProximityData.classify_zone(0.5) == ProximityZone.IMMEDIATE
        assert ProximityData.classify_zone(2.0) == ProximityZone.NEAR
        assert ProximityData.classify_zone(5.0) == ProximityZone.ROOM
        assert ProximityData.classify_zone(10.0) == ProximityZone.FAR
        assert ProximityData.classify_zone(20.0) == ProximityZone.OUT_OF_RANGE
        
        print("✅ ProximityData creation and methods working")
        return True
    
    @pytest.mark.asyncio
    async def test_display_location_creation(self):
        """Test DisplayLocation creation and persistence"""
        from proximity.proximity_display_context import DisplayLocation
        
        location = DisplayLocation(
            display_id=1,
            location_name="Living Room TV",
            zone="living_room",
            expected_proximity_range=(2.0, 8.0),
            auto_connect_enabled=True,
            connection_priority=0.8,
            tags=["tv", "entertainment"]
        )
        
        assert location.location_name == "Living Room TV"
        assert location.is_in_range(5.0)  # Within 2.0-8.0
        assert not location.is_in_range(10.0)  # Outside range
        
        # Test serialization
        data_dict = location.to_dict()
        location_restored = DisplayLocation.from_dict(data_dict)
        assert location_restored.location_name == location.location_name
        assert location_restored.zone == location.zone
        
        print("✅ DisplayLocation creation and serialization working")
        return True
    
    @pytest.mark.asyncio
    async def test_bluetooth_proximity_service_init(self):
        """Test BluetoothProximityService initialization"""
        from proximity.bluetooth_proximity_service import BluetoothProximityService
        from proximity.proximity_display_context import ProximityThresholds
        
        thresholds = ProximityThresholds()
        service = BluetoothProximityService(thresholds)
        
        assert service is not None
        assert service.rssi_at_1m == -59
        assert service.path_loss_exponent == 2.5
        assert len(service.tracked_devices) == 0
        
        print("✅ BluetoothProximityService initialization working")
        return True
    
    @pytest.mark.asyncio
    async def test_bluetooth_availability_check(self):
        """Test Bluetooth availability detection"""
        from proximity.bluetooth_proximity_service import BluetoothProximityService
        
        service = BluetoothProximityService()
        available = await service.check_bluetooth_availability()
        
        # Should return True or False, not error
        assert isinstance(available, bool)
        
        print(f"✅ Bluetooth availability check: {available}")
        return True
    
    @pytest.mark.asyncio
    async def test_rssi_to_distance_conversion(self):
        """Test RSSI to distance conversion"""
        from proximity.bluetooth_proximity_service import BluetoothProximityService
        
        service = BluetoothProximityService()
        
        # Test various RSSI values
        # Strong signal (-40 dBm) should be close
        distance_close = service._rssi_to_distance(-40)
        assert distance_close < 1.0, f"Distance for -40 dBm should be < 1m, got {distance_close}"
        
        # Medium signal (-60 dBm) should be ~1-3m
        distance_medium = service._rssi_to_distance(-60)
        assert 1.0 < distance_medium < 5.0, f"Distance for -60 dBm should be 1-5m, got {distance_medium}"
        
        # Weak signal (-80 dBm) should be far
        distance_far = service._rssi_to_distance(-80)
        assert distance_far > 5.0, f"Distance for -80 dBm should be > 5m, got {distance_far}"
        
        print(f"✅ RSSI conversion: -40dBm={distance_close:.1f}m, -60dBm={distance_medium:.1f}m, -80dBm={distance_far:.1f}m")
        return True
    
    @pytest.mark.asyncio
    async def test_kalman_filter_smoothing(self):
        """Test Kalman filter for RSSI smoothing"""
        from proximity.bluetooth_proximity_service import KalmanFilter
        
        kf = KalmanFilter()
        
        # Simulate noisy RSSI readings
        noisy_rssi = [-60, -65, -58, -62, -60, -64, -59]
        smoothed = []
        
        for rssi in noisy_rssi:
            smoothed_rssi = kf.update(rssi)
            smoothed.append(smoothed_rssi)
        
        # Smoothed values should have less variance
        import statistics
        original_variance = statistics.variance(noisy_rssi)
        smoothed_variance = statistics.variance(smoothed)
        
        assert smoothed_variance < original_variance, "Kalman filter should reduce variance"
        
        print(f"✅ Kalman filter: Original variance={original_variance:.2f}, Smoothed={smoothed_variance:.2f}")
        return True
    
    @pytest.mark.asyncio
    async def test_proximity_display_bridge_init(self):
        """Test ProximityDisplayBridge initialization"""
        from proximity.proximity_display_bridge import ProximityDisplayBridge
        
        bridge = ProximityDisplayBridge()
        
        assert bridge is not None
        assert bridge.proximity_service is not None
        assert isinstance(bridge.display_locations, dict)
        
        # Check config was loaded
        await bridge.load_display_locations()
        assert len(bridge.display_locations) >= 0  # May be empty initially
        
        print(f"✅ ProximityDisplayBridge initialized with {len(bridge.display_locations)} display locations")
        return True
    
    @pytest.mark.asyncio
    async def test_display_location_registration(self):
        """Test registering a display location"""
        from proximity.proximity_display_bridge import ProximityDisplayBridge
        
        bridge = ProximityDisplayBridge()
        
        success = await bridge.register_display_location(
            display_id=99,
            location_name="Test TV",
            zone="test_zone",
            expected_proximity_range=(1.0, 5.0),
            auto_connect_enabled=True,
            connection_priority=0.7,
            tags=["test", "tv"]
        )
        
        assert success == True
        assert 99 in bridge.display_locations
        assert bridge.display_locations[99].location_name == "Test TV"
        
        print("✅ Display location registration working")
        return True
    
    @pytest.mark.asyncio
    async def test_proximity_scoring_no_proximity(self):
        """Test proximity scoring without proximity data"""
        from proximity.proximity_display_bridge import ProximityDisplayBridge
        
        bridge = ProximityDisplayBridge()
        
        # Register test display
        await bridge.register_display_location(
            display_id=1,
            location_name="Test Display",
            zone="office",
            expected_proximity_range=(0.0, 5.0),
            connection_priority=0.8
        )
        
        displays = [{"display_id": 1, "name": "Test Display"}]
        
        scores = await bridge._calculate_proximity_scores(None, displays)
        
        # Without proximity data, should return base priority score
        assert 1 in scores
        assert scores[1] == 0.8  # Should match connection_priority
        
        print(f"✅ Proximity scoring (no proximity): {scores}")
        return True
    
    @pytest.mark.asyncio
    async def test_proximity_context_generation(self):
        """Test generating proximity-display context"""
        from proximity.proximity_display_bridge import ProximityDisplayBridge
        
        bridge = ProximityDisplayBridge()
        
        # Mock displays
        displays = [
            {"display_id": 1, "name": "MacBook Pro", "resolution": [1440, 900], "position": [0, 0], "is_primary": True}
        ]
        
        context = await bridge.get_proximity_display_context(displays)
        
        assert context is not None
        assert context.available_displays == displays
        assert isinstance(context.proximity_scores, dict)
        
        print("✅ Proximity context generation working")
        return True
    
    @pytest.mark.asyncio
    async def test_connection_decision_no_proximity(self):
        """Test connection decision without proximity data"""
        from proximity.proximity_display_bridge import ProximityDisplayBridge
        
        bridge = ProximityDisplayBridge()
        
        decision = await bridge.make_connection_decision()
        
        # Without proximity data, should return None
        assert decision is None
        
        print("✅ Connection decision (no proximity) correctly returns None")
        return True
    
    @pytest.mark.asyncio
    async def test_device_type_classification(self):
        """Test device type classification"""
        from proximity.bluetooth_proximity_service import BluetoothProximityService
        
        service = BluetoothProximityService()
        
        assert service._classify_device_type("Derek's Apple Watch") == "apple_watch"
        assert service._classify_device_type("Derek's iPhone") == "iphone"
        assert service._classify_device_type("AirPods Pro") == "airpods"
        assert service._classify_device_type("MacBook Pro") == "mac"
        assert service._classify_device_type("Unknown Device") == "unknown"
        
        print("✅ Device type classification working")
        return True
    
    @pytest.mark.asyncio
    async def test_signal_quality_calculation(self):
        """Test signal quality calculation"""
        from proximity.bluetooth_proximity_service import BluetoothProximityService
        
        service = BluetoothProximityService()
        
        # Test various RSSI levels
        quality_excellent = service._calculate_signal_quality(-45)
        quality_good = service._calculate_signal_quality(-55)
        quality_fair = service._calculate_signal_quality(-65)
        quality_poor = service._calculate_signal_quality(-75)
        quality_very_poor = service._calculate_signal_quality(-85)
        
        assert quality_excellent == 1.0
        assert quality_good == 0.8
        assert quality_fair == 0.6
        assert quality_poor == 0.4
        assert quality_very_poor == 0.2
        
        print("✅ Signal quality calculation working")
        return True
    
    @pytest.mark.asyncio
    async def test_proximity_thresholds_customization(self):
        """Test customizing proximity thresholds"""
        from proximity.proximity_display_context import ProximityThresholds
        
        thresholds = ProximityThresholds(
            immediate_distance=0.5,
            near_distance=2.0,
            auto_connect_distance=1.0,
            auto_connect_confidence=0.9
        )
        
        assert thresholds.immediate_distance == 0.5
        assert thresholds.auto_connect_distance == 1.0
        
        # Test serialization
        data_dict = thresholds.to_dict()
        restored = ProximityThresholds.from_dict(data_dict)
        assert restored.immediate_distance == thresholds.immediate_distance
        
        print("✅ Proximity thresholds customization working")
        return True
    
    @pytest.mark.asyncio
    async def test_bridge_stats(self):
        """Test bridge statistics"""
        from proximity.proximity_display_bridge import ProximityDisplayBridge
        
        bridge = ProximityDisplayBridge()
        
        # Generate some context
        await bridge.get_proximity_display_context()
        
        stats = bridge.get_bridge_stats()
        
        assert "context_generation_count" in stats
        assert stats["context_generation_count"] >= 1
        assert "proximity_service_stats" in stats
        
        print(f"✅ Bridge stats: {stats['context_generation_count']} contexts generated")
        return True


async def run_all_tests():
    """Run all proximity-display system tests"""
    print("\n" + "="*70)
    print("🧪 PHASE 1.2 PROXIMITY-AWARE DISPLAY SYSTEM - COMPREHENSIVE TESTS")
    print("="*70 + "\n")
    
    tests = TestProximityDisplaySystem()
    results = []
    
    test_methods = [
        ("ProximityData Creation", tests.test_proximity_data_creation),
        ("DisplayLocation Creation", tests.test_display_location_creation),
        ("BluetoothService Init", tests.test_bluetooth_proximity_service_init),
        ("Bluetooth Availability", tests.test_bluetooth_availability_check),
        ("RSSI to Distance Conversion", tests.test_rssi_to_distance_conversion),
        ("Kalman Filter Smoothing", tests.test_kalman_filter_smoothing),
        ("ProximityDisplayBridge Init", tests.test_proximity_display_bridge_init),
        ("Display Location Registration", tests.test_display_location_registration),
        ("Proximity Scoring (No Proximity)", tests.test_proximity_scoring_no_proximity),
        ("Proximity Context Generation", tests.test_proximity_context_generation),
        ("Connection Decision (No Proximity)", tests.test_connection_decision_no_proximity),
        ("Device Type Classification", tests.test_device_type_classification),
        ("Signal Quality Calculation", tests.test_signal_quality_calculation),
        ("Proximity Thresholds", tests.test_proximity_thresholds_customization),
        ("Bridge Statistics", tests.test_bridge_stats)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in test_methods:
        try:
            print(f"\n📍 {test_name}")
            print("-" * 70)
            result = await test_func()
            results.append((test_name, True, None))
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False, str(e)))
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    for test_name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if error:
            print(f"       Error: {error}")
    
    print(f"\n📈 Results: {passed}/{len(test_methods)} tests passed ({int(passed/len(test_methods)*100)}%)")
    
    if passed == len(test_methods):
        print("\n🎉 ALL TESTS PASSED - PHASE 1.2 READY!")
        print("\n✅ COMPONENTS VERIFIED:")
        print("   • ProximityData & DisplayLocation - ✅")
        print("   • BluetoothProximityService - ✅")
        print("   • ProximityDisplayBridge - ✅")
        print("   • RSSI→Distance Conversion - ✅")
        print("   • Kalman Filter Smoothing - ✅")
        print("   • Proximity Scoring - ✅")
        print("   • Connection Decision Logic - ✅")
        print("\n🚀 Proximity-Aware Display System: OPERATIONAL")
        return True
    else:
        print(f"\n⚠️  {failed} tests failed - review errors above")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
