#!/usr/bin/env python3
"""
Test the 3-step display connection process
"""

import asyncio
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_display_connection():
    """Test connecting to Living Room TV"""

    # Import the factory
    from backend.display.control_center_clicker_factory import get_best_clicker

    print("\n" + "="*60)
    print("🎯 Testing Display Connection to Living Room TV")
    print("="*60)

    print("\n📍 Expected coordinates:")
    print("  1. Control Center: (1236, 12)")
    print("  2. Screen Mirroring: (1396, 177)")
    print("  3. Living Room TV: (1223, 115)")
    print()

    # Get the best clicker
    print("🔧 Getting clicker...")
    clicker = get_best_clicker(
        enable_verification=True,
        prefer_uae=False  # Use simple adaptive for testing
    )
    print(f"   Using: {clicker.__class__.__name__}")

    # Test the 3-step process
    print("\n🚀 Starting 3-step connection process...")
    print("-" * 40)

    try:
        # Step 1: Click Control Center
        print("\n📍 Step 1: Clicking Control Center (1236, 12)...")
        result = await clicker.click("control_center")
        if hasattr(result, 'success'):
            print(f"   Result: {'✅ Success' if result.success else '❌ Failed'}")
            if result.success:
                print(f"   Clicked at: {result.coordinates}")
                print(f"   Method: {result.method_used}")

        await asyncio.sleep(0.5)  # Wait for menu to open

        # Step 2: Click Screen Mirroring
        print("\n📍 Step 2: Clicking Screen Mirroring (1396, 177)...")
        result = await clicker.click("screen_mirroring")
        if hasattr(result, 'success'):
            print(f"   Result: {'✅ Success' if result.success else '❌ Failed'}")
            if result.success:
                print(f"   Clicked at: {result.coordinates}")
                print(f"   Method: {result.method_used}")

        await asyncio.sleep(0.5)  # Wait for submenu

        # Step 3: Click Living Room TV
        print("\n📍 Step 3: Clicking Living Room TV (1223, 115)...")
        result = await clicker.click("Living Room TV")
        if hasattr(result, 'success'):
            print(f"   Result: {'✅ Success' if result.success else '❌ Failed'}")
            if result.success:
                print(f"   Clicked at: {result.coordinates}")
                print(f"   Method: {result.method_used}")

        print("\n" + "="*60)
        print("✅ Test Complete!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

        # Try to close any open menus
        print("\n🧹 Cleaning up...")
        import pyautogui
        pyautogui.press('escape')
        await asyncio.sleep(0.2)
        pyautogui.press('escape')

if __name__ == "__main__":
    asyncio.run(test_display_connection())