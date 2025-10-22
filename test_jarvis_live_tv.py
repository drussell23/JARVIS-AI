#!/usr/bin/env python3
"""
Test JARVIS live with Living Room TV connection
Ensures fresh instances with drag fix are used
"""

import asyncio
import sys
from pathlib import Path
import time

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

async def test_live_jarvis():
    """Test JARVIS with fresh instances"""
    print("\n" + "=" * 80)
    print("TESTING LIVE JARVIS WITH FRESH INSTANCES")
    print("=" * 80)

    # Step 1: Clear singletons
    print("\n1. Clearing singletons to force fresh instances...")
    try:
        import backend.display.adaptive_control_center_clicker as acc
        import backend.display.sai_enhanced_control_center_clicker as sai
        import backend.display.uae_enhanced_control_center_clicker as uae
        import backend.display.advanced_display_monitor as adm

        acc._adaptive_clicker = None
        sai._sai_clicker = None
        uae._uae_clicker = None
        adm._monitor_instance = None
        print("   ✅ All singletons cleared")
    except Exception as e:
        print(f"   ⚠️  Could not clear all singletons: {e}")

    # Step 2: Get fresh clicker
    print("\n2. Getting fresh clicker instance...")
    from backend.display.control_center_clicker_factory import get_best_clicker

    clicker = get_best_clicker(
        vision_analyzer=None,
        cache_ttl=0,  # No cache
        enable_verification=False
    )
    print(f"   ✅ Got fresh {clicker.__class__.__name__}")

    # Step 3: Verify drag fix is present
    import inspect
    source = inspect.getsource(clicker.click)
    has_dragto = 'dragTo' in source
    print(f"   ✅ Drag fix present: {has_dragto}")

    # Step 4: Test Control Center opening
    print("\n3. Testing Control Center with drag motion...")
    result = await clicker.open_control_center()

    if result.success:
        print(f"   ✅ Control Center opened!")
        print(f"      Method: {result.method_used}")
        print(f"      Coordinates: {result.coordinates}")

        # Close it
        import pyautogui
        await asyncio.sleep(1)
        pyautogui.press('escape')
        print("   ✅ Closed Control Center")

        # Step 5: Test full connection
        print("\n4. Testing full Living Room TV connection...")
        await asyncio.sleep(1)

        tv_result = await clicker.connect_to_device("Living Room TV")
        if tv_result['success']:
            print(f"   ✅ Successfully connected to Living Room TV!")
            print(f"      Duration: {tv_result['duration']:.2f}s")
            return True
        else:
            print(f"   ❌ Failed to connect: {tv_result.get('message', 'Unknown error')}")
            return False
    else:
        print(f"   ❌ Failed to open Control Center: {result.error}")
        return False


async def main():
    """Run the test"""
    print("\n" + "=" * 80)
    print("JARVIS LIVE TEST WITH DRAG FIX")
    print("=" * 80)
    print("\nThis test will:")
    print("1. Clear all singleton instances")
    print("2. Get a fresh clicker with the drag fix")
    print("3. Test Control Center opening")
    print("4. Test Living Room TV connection")

    try:
        success = await test_live_jarvis()

        print("\n" + "=" * 80)
        print("TEST RESULTS")
        print("=" * 80)

        if success:
            print("✅ JARVIS is working correctly with the drag fix!")
            print("\nThe Living Room TV connection is functioning properly.")
            print("JARVIS should now respond correctly when you ask it to")
            print("connect to the Living Room TV during normal usage.")
        else:
            print("❌ There may still be issues.")
            print("\nTry the following:")
            print("1. Restart JARVIS completely")
            print("2. Clear the cache: rm -rf ~/.jarvis/*.json")
            print("3. Check if Control Center is accessible manually")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())