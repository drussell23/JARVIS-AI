#!/usr/bin/env python3
"""
Test the EXACT method JARVIS uses for clicking
This bypasses everything and directly uses the AdaptiveControlCenterClicker
"""

import asyncio
import sys
sys.path.insert(0, '/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')

async def test_exact_jarvis_method():
    """Test using the exact same method as JARVIS"""

    print("\n" + "="*60)
    print("🎯 Testing EXACT JARVIS Method")
    print("="*60)

    # Import exactly as JARVIS does
    from backend.display.control_center_clicker_factory import get_best_clicker

    print("\n📍 Getting best clicker (same as JARVIS)...")
    cc_clicker = get_best_clicker(
        vision_analyzer=None,  # No vision for this test
        enable_verification=True,
        prefer_uae=True
    )

    print(f"✅ Using clicker: {cc_clicker.__class__.__name__}")

    print("\n🚀 Calling connect_to_device('Living Room TV')...")
    print("   This is the EXACT method JARVIS calls")
    print("   Expected sequence:")
    print("   1. Control Center (1236, 12)")
    print("   2. Screen Mirroring (1396, 177)")
    print("   3. Living Room TV (1223, 115)")
    print()

    # This is EXACTLY what JARVIS calls
    result = await cc_clicker.connect_to_device("Living Room TV")

    print("\n" + "="*60)
    print("📊 Result:")
    print("="*60)
    print(f"Success: {result.get('success')}")
    print(f"Message: {result.get('message')}")
    print(f"Duration: {result.get('duration', 0):.2f}s")

    if not result.get('success'):
        print(f"\n❌ Failed: {result.get('error', 'Unknown error')}")
        print(f"Step failed: {result.get('step_failed')}")
    else:
        print("\n✅ Connection successful!")

    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_exact_jarvis_method())