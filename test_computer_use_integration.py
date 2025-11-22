#!/usr/bin/env python3
"""
Test Script for JARVIS Computer Use Integration

This script tests the Claude Computer Use API integration for dynamic,
vision-based UI automation without hardcoded coordinates.

Usage:
    # Test with a specific display name
    python test_computer_use_integration.py "Living Room TV"

    # Test with default display
    python test_computer_use_integration.py

    # Test in silent mode (no voice)
    python test_computer_use_integration.py --silent

Requirements:
    - ANTHROPIC_API_KEY environment variable set
    - anthropic package installed (pip install anthropic)
    - pyautogui package installed

Author: JARVIS AI System
"""

import asyncio
import argparse
import logging
import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_requirements():
    """Check that all requirements are met."""
    issues = []

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        issues.append("ANTHROPIC_API_KEY environment variable not set")

    # Check packages
    try:
        import anthropic
    except ImportError:
        issues.append("anthropic package not installed (pip install anthropic)")

    try:
        import pyautogui
    except ImportError:
        issues.append("pyautogui package not installed (pip install pyautogui)")

    try:
        from PIL import Image
    except ImportError:
        issues.append("Pillow package not installed (pip install Pillow)")

    return issues


async def test_computer_use_connector():
    """Test the Computer Use connector directly."""
    print("\n" + "=" * 60)
    print("TEST 1: Computer Use Connector Direct Test")
    print("=" * 60)

    try:
        from backend.display.computer_use_connector import (
            ClaudeComputerUseConnector,
            get_computer_use_connector
        )

        connector = get_computer_use_connector()
        print("✅ Computer Use connector created successfully")

        # Test screenshot capture
        screenshot, base64_img = await connector.screen_capture.capture()
        print(f"✅ Screenshot captured: {screenshot.size}")

        return True

    except Exception as e:
        print(f"❌ Computer Use connector test failed: {e}")
        return False


async def test_vision_navigator_integration():
    """Test the Vision Navigator with Computer Use integration."""
    print("\n" + "=" * 60)
    print("TEST 2: Vision Navigator Integration Test")
    print("=" * 60)

    try:
        from backend.display.vision_ui_navigator import VisionUINavigator

        navigator = VisionUINavigator()
        print("✅ Vision Navigator created")

        # Check if Computer Use connector is available
        connector = await navigator._get_computer_use_connector()
        if connector:
            print("✅ Computer Use connector loaded via Vision Navigator")
        else:
            print("⚠️  Computer Use connector not available (will use fallback)")

        # Get stats
        stats = navigator.get_stats()
        print(f"📊 Navigator stats: {stats}")

        return True

    except Exception as e:
        print(f"❌ Vision Navigator test failed: {e}")
        return False


async def test_jarvis_integration(display_name: str, silent: bool = False):
    """Test the full JARVIS Computer Use integration."""
    print("\n" + "=" * 60)
    print("TEST 3: Full JARVIS Computer Use Integration Test")
    print("=" * 60)

    try:
        from backend.display.jarvis_computer_use_integration import (
            JARVISComputerUse,
            ExecutionMode,
            get_jarvis_computer_use
        )

        mode = ExecutionMode.SILENT if silent else ExecutionMode.FULL_VOICE
        print(f"📢 Execution mode: {mode.value}")

        jarvis = JARVISComputerUse(execution_mode=mode)
        initialized = await jarvis.initialize()

        if initialized:
            print("✅ JARVIS Computer Use initialized")
        else:
            print("⚠️  JARVIS Computer Use partially initialized")

        # Get stats
        stats = jarvis.get_stats()
        print(f"📊 Integration stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")

        return True

    except Exception as e:
        print(f"❌ JARVIS integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_display_connection(display_name: str, silent: bool = False):
    """Test actual display connection."""
    print("\n" + "=" * 60)
    print(f"TEST 4: Display Connection Test - '{display_name}'")
    print("=" * 60)

    print("\n⚠️  This test will attempt to interact with your screen!")
    print("    Make sure no important work is in progress.")
    response = input("\nProceed with connection test? (y/n): ")

    if response.lower() != 'y':
        print("Skipping connection test.")
        return True

    try:
        from backend.display.jarvis_computer_use_integration import (
            JARVISComputerUse,
            ExecutionMode
        )

        mode = ExecutionMode.SILENT if silent else ExecutionMode.FULL_VOICE
        jarvis = JARVISComputerUse(execution_mode=mode)
        await jarvis.initialize()

        print(f"\n🚀 Attempting to connect to '{display_name}'...")
        print("   (Watch your screen for automated actions)")

        result = await jarvis.connect_to_display(display_name, narrate=not silent)

        print("\n" + "-" * 40)
        print("📋 CONNECTION RESULT:")
        print("-" * 40)
        print(f"   Success: {result.success}")
        print(f"   Message: {result.message}")
        print(f"   Confidence: {result.confidence:.2%}")
        print(f"   Duration: {result.duration_seconds:.2f}s")
        print(f"   Method: {result.method_used}")

        if result.narration_transcript:
            print(f"\n📜 Narration Transcript:")
            for line in result.narration_transcript:
                print(f"   🔊 {line}")

        if result.learning_insights:
            print(f"\n💡 Learning Insights:")
            for insight in result.learning_insights:
                print(f"   • {insight}")

        if result.error:
            print(f"\n❌ Error: {result.error}")

        return result.success

    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_custom_task(silent: bool = False):
    """Test custom task execution."""
    print("\n" + "=" * 60)
    print("TEST 5: Custom Task Execution Test")
    print("=" * 60)

    print("\n⚠️  This test will attempt to interact with your screen!")
    response = input("\nProceed with custom task test? (y/n): ")

    if response.lower() != 'y':
        print("Skipping custom task test.")
        return True

    try:
        from backend.display.jarvis_computer_use_integration import (
            JARVISComputerUse,
            ExecutionMode
        )

        mode = ExecutionMode.SILENT if silent else ExecutionMode.FULL_VOICE
        jarvis = JARVISComputerUse(execution_mode=mode)
        await jarvis.initialize()

        # Simple task: take a screenshot and describe what's visible
        task = "Look at the current screen and describe what application windows are open"

        print(f"\n🚀 Executing task: '{task}'")

        result = await jarvis.execute_custom_task(task, narrate=not silent)

        print("\n" + "-" * 40)
        print("📋 TASK RESULT:")
        print("-" * 40)
        print(f"   Success: {result.success}")
        print(f"   Message: {result.message}")
        print(f"   Duration: {result.duration_seconds:.2f}s")

        return result.success

    except Exception as e:
        print(f"❌ Custom task test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests(display_name: str, silent: bool = False, skip_interactive: bool = False):
    """Run all tests."""
    print("\n" + "=" * 60)
    print("JARVIS COMPUTER USE INTEGRATION TEST SUITE")
    print("=" * 60)

    # Check requirements first
    issues = check_requirements()
    if issues:
        print("\n❌ Requirements check failed:")
        for issue in issues:
            print(f"   • {issue}")
        return False

    print("\n✅ All requirements met")

    results = {}

    # Test 1: Computer Use connector
    results["connector"] = await test_computer_use_connector()

    # Test 2: Vision Navigator integration
    results["navigator"] = await test_vision_navigator_integration()

    # Test 3: JARVIS integration
    results["jarvis"] = await test_jarvis_integration(display_name, silent)

    if not skip_interactive:
        # Test 4: Actual connection
        results["connection"] = await test_display_connection(display_name, silent)

        # Test 5: Custom task
        results["custom_task"] = await test_custom_task(silent)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"   {test_name}: {status}")

    print(f"\n{'✅' if passed == total else '⚠️'} {passed}/{total} tests passed")

    return passed == total


def main():
    parser = argparse.ArgumentParser(
        description="Test JARVIS Computer Use Integration"
    )
    parser.add_argument(
        "display_name",
        nargs="?",
        default="Test Display",
        help="Name of display to connect to (default: 'Test Display')"
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Run without voice narration"
    )
    parser.add_argument(
        "--skip-interactive",
        action="store_true",
        help="Skip interactive tests (connection, custom task)"
    )

    args = parser.parse_args()

    print(f"\n🎯 Target display: {args.display_name}")
    print(f"🔇 Silent mode: {args.silent}")
    print(f"⏭️  Skip interactive: {args.skip_interactive}")

    success = asyncio.run(
        run_all_tests(
            args.display_name,
            silent=args.silent,
            skip_interactive=args.skip_interactive
        )
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
