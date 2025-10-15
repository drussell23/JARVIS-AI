#!/usr/bin/env python3
"""
Comprehensive Test Suite for JARVIS Display Monitor
====================================================

Tests all components of the advanced display monitoring system:
- Configuration management
- Display detection
- Voice integration
- Event callbacks
- Error handling

Usage:
    python3 test_advanced_display_monitor.py [--verbose] [--quick]

Author: Derek Russell
Date: 2025-10-15
Version: 1.0
"""

import asyncio
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, List
import argparse

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import logging

# Test results tracker
test_results = []


class TestResult:
    """Track individual test results"""
    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name = name
        self.passed = passed
        self.message = message


def test_header(name: str):
    """Print test header"""
    print(f"\n{'='*80}")
    print(f"🧪 TEST: {name}")
    print(f"{'='*80}")


def test_result(name: str, passed: bool, message: str = ""):
    """Record and print test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    result = TestResult(name, passed, message)
    test_results.append(result)

    print(f"{status}: {name}")
    if message:
        print(f"  {message}")

    return passed


async def test_imports():
    """Test 1: Verify all imports"""
    test_header("Module Imports")

    try:
        from display.advanced_display_monitor import (
            AdvancedDisplayMonitor,
            DetectionMethod,
            DisplayType,
            ConnectionMode,
            DisplayInfo,
            MonitoredDisplay,
            get_display_monitor
        )
        test_result("Import AdvancedDisplayMonitor", True)
    except Exception as e:
        test_result("Import AdvancedDisplayMonitor", False, str(e))
        return False

    try:
        from display.display_config_manager import (
            DisplayConfigManager,
            get_config_manager
        )
        test_result("Import DisplayConfigManager", True)
    except Exception as e:
        test_result("Import DisplayConfigManager", False, str(e))
        return False

    try:
        from display.display_voice_handler import (
            DisplayVoiceHandler,
            create_voice_handler
        )
        test_result("Import DisplayVoiceHandler", True)
    except Exception as e:
        test_result("Import DisplayVoiceHandler", False, str(e))
        return False

    return True


async def test_config_manager():
    """Test 2: Configuration Manager"""
    test_header("Configuration Manager")

    from display.display_config_manager import DisplayConfigManager

    # Create temp config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_config_path = f.name
        json.dump({
            "display_monitoring": {
                "enabled": True,
                "check_interval_seconds": 5.0
            },
            "displays": {
                "monitored_displays": []
            },
            "voice_integration": {
                "enabled": True
            }
        }, f)

    try:
        # Test load
        manager = DisplayConfigManager(temp_config_path)
        test_result("Load configuration", True)

        # Test get
        enabled = manager.get('display_monitoring.enabled')
        test_result("Get config value", enabled == True, f"Got: {enabled}")

        # Test set
        manager.set('display_monitoring.check_interval_seconds', 15.0, save=False)
        value = manager.get('display_monitoring.check_interval_seconds')
        test_result("Set config value", value == 15.0, f"Set to 15.0, got: {value}")

        # Test add display
        display_config = {
            'id': 'test_display',
            'name': 'Test Display',
            'display_type': 'airplay',
            'aliases': ['Test'],
            'auto_connect': False,
            'auto_prompt': True,
            'connection_mode': 'extend',
            'priority': 1,
            'enabled': True
        }
        manager.add_display(display_config, save=False)
        displays = manager.get_monitored_displays()
        test_result("Add display", len(displays) == 1, f"Added 1 display, found {len(displays)}")

        # Test remove display
        manager.remove_display('test_display', save=False)
        displays = manager.get_monitored_displays()
        test_result("Remove display", len(displays) == 0, f"Removed display, {len(displays)} remaining")

        # Test preset
        manager.apply_preset('minimal', save=False)
        test_result("Apply preset", True)

        # Test export/import
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as export_file:
            export_path = export_file.name

        manager.export_config(export_path)
        test_result("Export config", Path(export_path).exists())

        # Cleanup
        Path(temp_config_path).unlink()
        Path(export_path).unlink()

    except Exception as e:
        test_result("Configuration Manager", False, str(e))
        return False

    return True


async def test_voice_handler():
    """Test 3: Voice Handler"""
    test_header("Voice Handler")

    from display.display_voice_handler import DisplayVoiceHandler

    try:
        handler = DisplayVoiceHandler()
        test_result("Create voice handler", True)

        # Test get available voices
        voices = handler.get_available_voices()
        test_result("Get available voices", len(voices) > 0, f"Found {len(voices)} voices")

        # Test settings
        handler.set_enabled(False)
        test_result("Disable voice", handler.voice_enabled == False)

        handler.set_enabled(True)
        test_result("Enable voice", handler.voice_enabled == True)

        handler.set_voice_rate(1.2)
        test_result("Set voice rate", handler.voice_rate == 1.2)

        handler.set_voice_name("Samantha")
        test_result("Set voice name", handler.voice_name == "Samantha")

    except Exception as e:
        test_result("Voice Handler", False, str(e))
        return False

    return True


async def test_display_detection():
    """Test 4: Display Detection"""
    test_header("Display Detection")

    from display.advanced_display_monitor import (
        AppleScriptDetector,
        CoreGraphicsDetector,
        YabaiDetector
    )

    # Test AppleScript Detector
    try:
        config = {
            'timeout_seconds': 5.0,
            'retry_attempts': 2,
            'retry_delay_seconds': 0.5,
            'filter_system_items': ['Turn Display Mirroring Off', '']
        }
        detector = AppleScriptDetector(config)
        displays = await detector.detect_displays()
        test_result(
            "AppleScript detection",
            True,
            f"Detected {len(displays)} display(s): {displays}"
        )
    except Exception as e:
        test_result("AppleScript detection", False, str(e))

    # Test CoreGraphics Detector
    try:
        config = {
            'max_displays': 32,
            'exclude_builtin': True,
            'detect_airplay': True,
            'detect_external': True
        }
        detector = CoreGraphicsDetector(config)
        displays = await detector.detect_displays()
        test_result(
            "CoreGraphics detection",
            True,
            f"Detected {len(displays)} display(s)"
        )
    except Exception as e:
        test_result("CoreGraphics detection", False, str(e))

    # Test Yabai Detector (may not be installed)
    try:
        config = {'command_timeout': 3.0}
        detector = YabaiDetector(config)
        displays = await detector.detect_displays()
        test_result(
            "Yabai detection",
            True,
            f"Detected {len(displays)} display(s) (Yabai {'installed' if displays else 'not installed'})"
        )
    except Exception as e:
        test_result("Yabai detection", False, str(e))

    return True


async def test_monitor_lifecycle():
    """Test 5: Monitor Lifecycle"""
    test_header("Monitor Lifecycle")

    from display.advanced_display_monitor import get_display_monitor

    # Create temp config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_config_path = f.name
        json.dump({
            "display_monitoring": {
                "enabled": True,
                "check_interval_seconds": 30.0,
                "startup_delay_seconds": 0.1,
                "detection_methods": ["applescript"],
                "preferred_detection_method": "applescript"
            },
            "displays": {
                "monitored_displays": [{
                    "id": "test_tv",
                    "name": "Test TV",
                    "display_type": "airplay",
                    "aliases": ["Test"],
                    "auto_connect": False,
                    "auto_prompt": True,
                    "connection_mode": "extend",
                    "priority": 1,
                    "enabled": True
                }],
                "ignored_displays": []
            },
            "voice_integration": {"enabled": False},
            "applescript": {
                "enabled": True,
                "timeout_seconds": 5.0,
                "retry_attempts": 2,
                "retry_delay_seconds": 0.5,
                "filter_system_items": ["Turn Display Mirroring Off", ""]
            },
            "coregraphics": {"enabled": True},
            "yabai": {"enabled": False},
            "caching": {"enabled": True, "display_list_ttl_seconds": 5},
            "performance": {"parallel_detection": True},
            "notifications": {"enabled": True},
            "logging": {"level": "INFO"},
            "security": {"require_user_consent_first_time": False},
            "advanced": {"multi_monitor_support": True}
        }, f)

    try:
        # Create monitor
        monitor = get_display_monitor(temp_config_path, voice_handler=None)
        test_result("Create monitor", True)

        # Test status before starting
        status = monitor.get_status()
        test_result("Get status (not running)", status['is_monitoring'] == False)

        # Test callback registration
        callback_called = {'detected': False, 'lost': False}

        async def on_detected(display, detected_name):
            callback_called['detected'] = True

        async def on_lost(display):
            callback_called['lost'] = True

        monitor.register_callback('display_detected', on_detected)
        monitor.register_callback('display_lost', on_lost)
        test_result("Register callbacks", True)

        # Start monitor
        await monitor.start()
        await asyncio.sleep(0.5)  # Let it start
        status = monitor.get_status()
        test_result("Start monitor", status['is_monitoring'] == True)

        # Let it run for a bit
        await asyncio.sleep(1.0)

        # Stop monitor
        await monitor.stop()
        await asyncio.sleep(0.3)
        status = monitor.get_status()
        test_result("Stop monitor", status['is_monitoring'] == False)

        # Cleanup
        Path(temp_config_path).unlink()

    except Exception as e:
        test_result("Monitor Lifecycle", False, str(e))
        return False

    return True


async def test_event_callbacks():
    """Test 6: Event Callbacks"""
    test_header("Event Callbacks")

    from display.advanced_display_monitor import AdvancedDisplayMonitor

    # Create minimal config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_config_path = f.name
        json.dump({
            "display_monitoring": {"enabled": True, "check_interval_seconds": 30.0},
            "displays": {"monitored_displays": []},
            "voice_integration": {"enabled": False},
            "applescript": {"enabled": True},
            "coregraphics": {"enabled": True},
            "yabai": {"enabled": False},
            "caching": {"enabled": True, "display_list_ttl_seconds": 5},
            "performance": {},
            "notifications": {},
            "logging": {},
            "security": {},
            "advanced": {}
        }, f)

    try:
        monitor = AdvancedDisplayMonitor(temp_config_path, voice_handler=None)

        # Test multiple callbacks for same event
        calls = {'count': 0}

        async def callback1(**kwargs):
            calls['count'] += 1

        async def callback2(**kwargs):
            calls['count'] += 10

        monitor.register_callback('display_detected', callback1)
        monitor.register_callback('display_detected', callback2)

        # Emit test event
        await monitor._emit_event('display_detected')

        test_result("Multiple callbacks", calls['count'] == 11, f"Expected 11, got {calls['count']}")

        # Cleanup
        Path(temp_config_path).unlink()

    except Exception as e:
        test_result("Event Callbacks", False, str(e))
        return False

    return True


async def test_error_handling():
    """Test 7: Error Handling"""
    test_header("Error Handling")

    from display.display_config_manager import DisplayConfigManager

    # Test loading non-existent config
    try:
        manager = DisplayConfigManager("/tmp/nonexistent_config_12345.json")
        test_result("Load non-existent config", False, "Should have raised exception")
    except:
        test_result("Load non-existent config", True, "Correctly raised exception")

    # Test invalid JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("{ invalid json }")
        invalid_path = f.name

    try:
        manager = DisplayConfigManager(invalid_path)
        test_result("Load invalid JSON", False, "Should have raised exception")
    except:
        test_result("Load invalid JSON", True, "Correctly raised exception")
        Path(invalid_path).unlink()

    return True


async def test_caching():
    """Test 8: Caching System"""
    test_header("Caching System")

    from display.advanced_display_monitor import DisplayCache

    try:
        cache = DisplayCache(ttl_seconds=2)

        # Test set and get
        cache.set("test_key", ["display1", "display2"])
        result = cache.get("test_key")
        test_result("Cache set/get", result == ["display1", "display2"])

        # Test expiration
        await asyncio.sleep(2.5)
        result = cache.get("test_key")
        test_result("Cache expiration", result is None, "Cache should have expired")

        # Test clear
        cache.set("key1", ["value1"])
        cache.set("key2", ["value2"])
        cache.clear()
        test_result("Cache clear", cache.get("key1") is None and cache.get("key2") is None)

    except Exception as e:
        test_result("Caching System", False, str(e))
        return False

    return True


def print_summary():
    """Print test summary"""
    print(f"\n{'='*80}")
    print("📊 TEST SUMMARY")
    print(f"{'='*80}\n")

    passed = sum(1 for r in test_results if r.passed)
    failed = sum(1 for r in test_results if not r.passed)
    total = len(test_results)

    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {(passed/total*100):.1f}%\n")

    if failed > 0:
        print("Failed Tests:")
        for result in test_results:
            if not result.passed:
                print(f"  ❌ {result.name}")
                if result.message:
                    print(f"     {result.message}")
        print()

    print(f"{'='*80}\n")

    return failed == 0


async def run_all_tests(verbose=False, quick=False):
    """Run all tests"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "JARVIS DISPLAY MONITOR TEST SUITE" + " "*25 + "║")
    print("╚" + "="*78 + "╝")

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Run tests
    await test_imports()
    await test_config_manager()
    await test_voice_handler()
    await test_display_detection()

    if not quick:
        await test_monitor_lifecycle()
        await test_event_callbacks()
        await test_error_handling()
        await test_caching()

    # Print summary
    all_passed = print_summary()

    return all_passed


async def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Test JARVIS Display Monitor")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick tests only")

    args = parser.parse_args()

    all_passed = await run_all_tests(verbose=args.verbose, quick=args.quick)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test runner error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
