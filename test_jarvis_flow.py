#!/usr/bin/env python3
"""
Test the exact flow JARVIS uses when connecting to Living Room TV
This simulates what happens when you say "living room tv" to JARVIS
"""
import sys
import os
import asyncio
import logging

# Setup path
sys.path.insert(0, os.path.dirname(__file__))

# Setup logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_jarvis_flow():
    """Test the exact flow used by JARVIS"""
    print("\n" + "="*80)
    print("TESTING JARVIS DISPLAY CONNECTION FLOW")
    print("This simulates: User says 'living room tv'")
    print("="*80 + "\n")

    # Import the advanced display monitor (what JARVIS uses)
    from backend.display.advanced_display_monitor import AdvancedDisplayMonitor

    # Create monitor instance
    monitor = AdvancedDisplayMonitor()

    # Check available displays (no initialization needed)
    print("\nChecking available displays...")
    displays = monitor.get_available_display_details()

    print(f"Found {len(displays)} displays:")
    for display in displays:
        print(f"  - {display['display_name']} (id: {display['display_id']})")

    # Find Living Room TV
    living_room_id = None
    for display in displays:
        if "living room" in display['display_name'].lower():
            living_room_id = display['display_id']
            print(f"\n✅ Found Living Room TV with ID: {living_room_id}")
            break

    if not living_room_id:
        print("\n❌ Living Room TV not found in available displays")
        print("Available displays:", [d['display_name'] for d in displays])
        return

    # Now connect (this is what happens when JARVIS processes "living room tv")
    print("\n" + "-"*80)
    print("EXECUTING CONNECTION (same as JARVIS voice command)")
    print("-"*80 + "\n")

    result = await monitor.connect_display(living_room_id)

    print("\n" + "="*80)
    print("RESULT:")
    if result.get('success'):
        print("✅ SUCCESS!")
        print(f"  Message: {result.get('message', 'Connected')}")
        print(f"  Method: {result.get('method', 'unknown')}")
        if 'control_center_coords' in result:
            print(f"  Control Center: {result['control_center_coords']}")
        if 'screen_mirroring_coords' in result:
            print(f"  Screen Mirroring: {result['screen_mirroring_coords']}")
        if 'living_room_tv_coords' in result:
            print(f"  Living Room TV: {result['living_room_tv_coords']}")
    else:
        print(f"❌ FAILED: {result.get('message', 'Unknown error')}")
        print(f"  Error: {result.get('error', 'None')}")
    print("="*80 + "\n")

if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_jarvis_flow())