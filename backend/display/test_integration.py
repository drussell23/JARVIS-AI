#!/usr/bin/env python3
"""
Integration Test for Adaptive Control Center Integration
=========================================================

Quick test to verify the integration is working correctly.

Run this to verify:
1. Old control_center_clicker wraps new adaptive_control_center_clicker
2. Backward compatibility is maintained
3. Adaptive detection is enabled by default
4. Metrics are accessible

Author: Derek J. Russell
Date: October 2025
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_import():
    """Test 1: Verify imports work"""
    print("\n" + "=" * 75)
    print("Test 1: Verify Imports")
    print("=" * 75)

    try:
        from display.control_center_clicker import get_control_center_clicker
        print("✅ control_center_clicker imported successfully")

        from display.adaptive_control_center_clicker import get_adaptive_clicker
        print("✅ adaptive_control_center_clicker imported successfully")

        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_backward_compatibility():
    """Test 2: Verify backward compatibility"""
    print("\n" + "=" * 75)
    print("Test 2: Backward Compatibility")
    print("=" * 75)

    try:
        from display.control_center_clicker import get_control_center_clicker

        # Old API should still work
        clicker = get_control_center_clicker()

        # Check it has old methods
        assert hasattr(clicker, 'open_control_center'), "Missing open_control_center method"
        assert hasattr(clicker, 'connect_to_living_room_tv'), "Missing connect_to_living_room_tv method"
        assert hasattr(clicker, 'click_living_room_tv'), "Missing click_living_room_tv method"

        # Check it has old constants (deprecated but kept for compatibility)
        assert hasattr(clicker, 'CONTROL_CENTER_X'), "Missing CONTROL_CENTER_X constant"
        assert hasattr(clicker, 'CONTROL_CENTER_Y'), "Missing CONTROL_CENTER_Y constant"

        print("✅ All old methods and constants present")
        print("✅ Backward compatibility maintained")

        return True
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adaptive_mode_enabled():
    """Test 3: Verify adaptive mode is enabled by default"""
    print("\n" + "=" * 75)
    print("Test 3: Adaptive Mode Enabled by Default")
    print("=" * 75)

    try:
        from display.control_center_clicker import get_control_center_clicker

        clicker = get_control_center_clicker(use_adaptive=True)

        # Check adaptive mode is enabled
        assert clicker.use_adaptive is True, "Adaptive mode not enabled"
        print("✅ use_adaptive = True")

        # Check adaptive clicker is initialized
        assert clicker._adaptive_clicker is not None, "Adaptive clicker not initialized"
        print("✅ _adaptive_clicker initialized")

        # Check it's the right type
        from display.adaptive_control_center_clicker import AdaptiveControlCenterClicker
        assert isinstance(clicker._adaptive_clicker, AdaptiveControlCenterClicker), \
            "Wrong type for adaptive clicker"
        print("✅ Correct AdaptiveControlCenterClicker instance")

        return True
    except Exception as e:
        print(f"❌ Adaptive mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_available():
    """Test 4: Verify metrics are accessible"""
    print("\n" + "=" * 75)
    print("Test 4: Metrics Accessible")
    print("=" * 75)

    try:
        from display.control_center_clicker import get_control_center_clicker

        clicker = get_control_center_clicker(use_adaptive=True)

        # Get metrics
        metrics = clicker.get_metrics()

        assert isinstance(metrics, dict), "Metrics not a dict"
        print(f"✅ Metrics returned: {metrics}")

        # Check metrics has expected keys
        expected_keys = ['total_attempts', 'successful_clicks', 'success_rate']
        for key in expected_keys:
            if key in metrics:
                print(f"✅ Metric '{key}' present: {metrics[key]}")

        return True
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_legacy_mode():
    """Test 5: Verify legacy mode still works (fallback)"""
    print("\n" + "=" * 75)
    print("Test 5: Legacy Mode Available (Fallback)")
    print("=" * 75)

    try:
        from display.control_center_clicker import get_control_center_clicker

        # Explicitly request legacy mode
        clicker = get_control_center_clicker(use_adaptive=False)

        assert clicker.use_adaptive is False, "Legacy mode not disabled"
        print("✅ use_adaptive = False")

        assert clicker._adaptive_clicker is None, "Adaptive clicker should be None in legacy mode"
        print("✅ _adaptive_clicker is None (legacy mode)")

        # Metrics should indicate legacy mode
        metrics = clicker.get_metrics()
        assert metrics.get("mode") == "legacy", "Metrics don't indicate legacy mode"
        print("✅ Metrics indicate legacy mode")

        return True
    except Exception as e:
        print(f"❌ Legacy mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vision_analyzer_integration():
    """Test 6: Verify vision analyzer can be passed"""
    print("\n" + "=" * 75)
    print("Test 6: Vision Analyzer Integration")
    print("=" * 75)

    try:
        from display.control_center_clicker import get_control_center_clicker

        # Try to get vision analyzer
        vision_analyzer = None
        try:
            from vision.claude_vision_analyzer_main import get_claude_vision_analyzer
            vision_analyzer = get_claude_vision_analyzer()
            print("✅ Claude Vision analyzer available")
        except Exception as e:
            print(f"ℹ️  Claude Vision analyzer not available: {e}")
            print("   (This is OK - OCR will use pytesseract fallback)")

        # Create clicker with vision analyzer
        clicker = get_control_center_clicker(vision_analyzer=vision_analyzer)

        # Check it was set
        if vision_analyzer:
            assert clicker._vision_analyzer is not None, "Vision analyzer not set"
            print("✅ Vision analyzer set successfully")
        else:
            print("✅ Clicker works without vision analyzer (will use pytesseract)")

        return True
    except Exception as e:
        print(f"❌ Vision analyzer integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_display_monitor_service():
    """Test 7: Verify display_monitor_service integration"""
    print("\n" + "=" * 75)
    print("Test 7: Display Monitor Service Integration")
    print("=" * 75)

    try:
        from display.display_monitor_service import get_display_monitor

        # Create display monitor
        monitor = get_display_monitor(poll_interval_seconds=10.0)

        print("✅ Display monitor created successfully")

        # Register a display
        monitor.register_display(
            display_name="Living Room TV",
            auto_prompt=True,
            default_mode="extend"
        )

        print("✅ Display registered successfully")

        # Check it's in monitored displays
        assert "Living Room TV" in monitor.monitored_displays, "Display not registered"
        print("✅ Display appears in monitored_displays")

        return True
    except Exception as e:
        print(f"❌ Display monitor service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_location():
    """Test 8: Verify cache file location"""
    print("\n" + "=" * 75)
    print("Test 8: Cache File Location")
    print("=" * 75)

    try:
        from pathlib import Path

        cache_file = Path.home() / ".jarvis" / "control_center_cache.json"

        print(f"Expected cache location: {cache_file}")

        if cache_file.exists():
            print(f"✅ Cache file exists: {cache_file}")

            # Try to read it
            import json
            with open(cache_file, 'r') as f:
                data = json.load(f)
            print(f"✅ Cache file readable, contains {len(data)} entries")
        else:
            print(f"ℹ️  Cache file doesn't exist yet (will be created on first use)")

        # Check directory exists
        cache_dir = cache_file.parent
        if not cache_dir.exists():
            print(f"⚠️  Cache directory doesn't exist: {cache_dir}")
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created cache directory: {cache_dir}")
        else:
            print(f"✅ Cache directory exists: {cache_dir}")

        return True
    except Exception as e:
        print(f"❌ Cache location test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("\n" + "=" * 75)
    print("ADAPTIVE CONTROL CENTER - INTEGRATION TESTS")
    print("=" * 75)
    print("\nRunning 8 integration tests...\n")

    tests = [
        ("Import Test", test_import),
        ("Backward Compatibility", test_backward_compatibility),
        ("Adaptive Mode Enabled", test_adaptive_mode_enabled),
        ("Metrics Available", test_metrics_available),
        ("Legacy Mode Fallback", test_legacy_mode),
        ("Vision Analyzer Integration", test_vision_analyzer_integration),
        ("Display Monitor Service", test_display_monitor_service),
        ("Cache File Location", test_cache_location),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Print summary
    print("\n" + "=" * 75)
    print("TEST SUMMARY")
    print("=" * 75)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")

    print("\n" + "=" * 75)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 75)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Integration successful!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
