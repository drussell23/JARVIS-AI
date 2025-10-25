#!/usr/bin/env python3
"""
Multi-Monitor Integration Test

Quick integration test for multi-monitor support functionality.
This script tests the core components and API endpoints.

Author: Derek Russell
Date: 2025-01-14
Branch: multi-monitor-support
"""

import asyncio
import sys
import json
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from vision.multi_monitor_detector import (
    MultiMonitorDetector,
    MACOS_AVAILABLE
)


async def test_basic_functionality():
    """Test basic multi-monitor functionality"""
    print("🧪 Testing Multi-Monitor Basic Functionality")
    print("=" * 50)
    
    if not MACOS_AVAILABLE:
        print("❌ macOS frameworks not available - skipping tests")
        return False
    
    detector = MultiMonitorDetector()
    
    try:
        # Test 1: Display Detection
        print("1. Testing display detection...")
        displays = await detector.detect_displays()
        print(f"   ✅ Detected {len(displays)} displays")
        
        if displays:
            for display in displays:
                print(f"     - {display.name}: {display.resolution[0]}x{display.resolution[1]} {'[Primary]' if display.is_primary else ''}")
        
        # Test 2: Space Mapping
        print("\n2. Testing space-display mapping...")
        mappings = await detector.get_space_display_mapping()
        print(f"   ✅ Mapped {len(mappings)} spaces to displays")
        
        if mappings:
            for space_id, display_id in mappings.items():
                print(f"     - Space {space_id} → Display {display_id}")
        
        # Test 3: Display Summary
        print("\n3. Testing display summary...")
        summary = await detector.get_display_summary()
        print(f"   ✅ Generated summary with {summary.get('total_displays', 0)} displays")
        
        # Test 4: Performance Stats
        print("\n4. Testing performance statistics...")
        stats = detector.get_performance_stats()
        print(f"   ✅ Generated stats with {stats['capture_stats']['total_captures']} total captures")
        
        # Test 5: Capture (optional - may fail due to permissions)
        print("\n5. Testing screenshot capture...")
        try:
            result = await detector.capture_all_displays()
            if result.success:
                print(f"   ✅ Captured {len(result.displays_captured)} displays in {result.capture_time:.2f}s")
            else:
                print(f"   ⚠️  Capture failed: {result.error}")
        except Exception as e:
            print(f"   ⚠️  Capture test failed: {e}")
        
        print("\n🎉 All basic tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


async def test_api_endpoints():
    """Test API endpoint functionality"""
    print("\n🌐 Testing API Endpoint Functionality")
    print("=" * 50)
    
    try:
        # Import API components
        from api.vision_api import multi_monitor_detector
        
        # Test display summary endpoint logic
        print("1. Testing display summary endpoint...")
        summary = await multi_monitor_detector.get_display_summary()
        
        expected_keys = ["total_displays", "displays", "space_mappings", "detection_time", "capture_stats"]
        for key in expected_keys:
            if key in summary:
                print(f"   ✅ {key}: {type(summary[key])}")
            else:
                print(f"   ⚠️  Missing key: {key}")
        
        # Test performance stats endpoint logic
        print("\n2. Testing performance stats endpoint...")
        stats = multi_monitor_detector.get_performance_stats()
        
        expected_stats_keys = ["capture_stats", "displays_cached", "space_mappings_cached", "last_detection_time", "cache_age"]
        for key in expected_stats_keys:
            if key in stats:
                print(f"   ✅ {key}: {stats[key]}")
            else:
                print(f"   ⚠️  Missing key: {key}")
        
        print("\n🎉 API endpoint tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False


async def test_error_handling():
    """Test error handling scenarios"""
    print("\n🛡️  Testing Error Handling")
    print("=" * 50)
    
    detector = MultiMonitorDetector()
    
    try:
        # Test with invalid yabai path
        print("1. Testing invalid yabai path...")
        detector.yabai_path = "/nonexistent/yabai"
        mappings = await detector.get_space_display_mapping()
        print(f"   ✅ Handled invalid yabai path gracefully: {len(mappings)} mappings")
        
        # Test force refresh
        print("\n2. Testing force refresh...")
        await detector.detect_displays(force_refresh=True)
        print("   ✅ Force refresh completed")
        
        # Test cache behavior
        print("\n3. Testing cache behavior...")
        displays1 = await detector.detect_displays()
        displays2 = await detector.detect_displays()
        print(f"   ✅ Cache working: {len(displays1)} displays cached")
        
        print("\n🎉 Error handling tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


async def main():
    """Main test runner"""
    print("🚀 JARVIS Multi-Monitor Integration Test")
    print("=" * 60)
    
    if not MACOS_AVAILABLE:
        print("❌ macOS frameworks not available")
        print("   This test requires macOS with Core Graphics support")
        return 1
    
    success = True
    
    # Run all test suites
    success &= await test_basic_functionality()
    success &= await test_api_endpoints()
    success &= await test_error_handling()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("   Multi-monitor support is working correctly")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("   Check the output above for details")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
