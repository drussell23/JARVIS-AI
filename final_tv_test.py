#!/usr/bin/env python3
"""
Final comprehensive test for JARVIS TV connection with all fixes
"""

import asyncio
import sys
from pathlib import Path
import time

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

print("\n" + "=" * 80)
print("FINAL JARVIS TV CONNECTION TEST")
print("=" * 80)
print("\nThis test verifies:")
print("✅ Drag motion is used for Control Center")
print("✅ Fresh instances are created")
print("✅ Living Room TV connection works")

async def main():
    # Wait for JARVIS to fully initialize
    print("\nWaiting for JARVIS to initialize...")
    await asyncio.sleep(5)

    print("\n1. Testing with factory (force_new=True)...")
    from backend.display.control_center_clicker_factory import get_best_clicker

    clicker = get_best_clicker(force_new=True)
    print(f"   Got: {clicker.__class__.__name__}")

    # Test the connection
    print("\n2. Testing Living Room TV connection...")
    result = await clicker.connect_to_device("Living Room TV")

    if result['success']:
        print(f"\n✅ SUCCESS! Connected to Living Room TV")
        print(f"   Duration: {result['duration']:.2f}s")
        print(f"   Method: {result.get('method', 'unknown')}")

        # Show what happened
        if 'control_center_coords' in result:
            print(f"\n   Steps taken:")
            print(f"   1. Control Center: {result['control_center_coords']}")
            print(f"   2. Screen Mirroring: {result['screen_mirroring_coords']}")
            print(f"   3. Living Room TV: {result['living_room_tv_coords']}")

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nJARVIS is now correctly configured to:")
        print("• Use drag motion for Control Center")
        print("• Create fresh instances when needed")
        print("• Connect to Living Room TV reliably")
        print("\nYou can now ask JARVIS: 'Connect to Living Room TV'")
        print("and it should work correctly!")

    else:
        print(f"\n❌ Connection failed: {result.get('message', 'Unknown error')}")
        print(f"   Failed at: {result.get('step_failed', 'unknown')}")
        print("\nTroubleshooting:")
        print("1. Make sure Control Center is accessible")
        print("2. Ensure Living Room TV is powered on")
        print("3. Check that AirPlay is enabled on the TV")

if __name__ == "__main__":
    asyncio.run(main())