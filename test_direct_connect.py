#!/usr/bin/env python3
"""
Test direct connection to Living Room TV using the display ID from config
"""
import sys
import os
import asyncio
import logging

# Setup path
sys.path.insert(0, os.path.dirname(__file__))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_direct_connection():
    """Test direct connection using the display ID from config"""
    print("\n" + "="*80)
    print("TESTING DIRECT CONNECTION TO LIVING ROOM TV")
    print("Using display_id: living_room_tv (from config)")
    print("="*80 + "\n")

    # Import the advanced display monitor
    from backend.display.advanced_display_monitor import AdvancedDisplayMonitor

    # Create monitor instance
    monitor = AdvancedDisplayMonitor()

    # The display ID from the config is "living_room_tv"
    display_id = "living_room_tv"

    print(f"Connecting to display_id: {display_id}")
    print("This should use the simple clicker with coordinates:")
    print("  Control Center: (1236, 12)")
    print("  Screen Mirroring: (1393, 177)")
    print("  Living Room TV: (1221, 116)")
    print("\n" + "-"*80 + "\n")

    # Connect directly (this is what JARVIS does internally)
    result = await monitor.connect_display(display_id)

    print("\n" + "="*80)
    print("RESULT:")
    if result.get('success'):
        print("✅ SUCCESS!")
        print(f"  Message: {result.get('message', 'Connected')}")
        print(f"  Method: {result.get('method', 'unknown')}")
        print(f"  Already connected: {result.get('already_connected', False)}")
        print(f"  Cached: {result.get('cached', False)}")
    else:
        print(f"❌ FAILED: {result.get('message', 'Unknown error')}")
        if 'error' in result:
            print(f"  Error: {result['error']}")
        if 'strategies_attempted' in result:
            print(f"  Strategies attempted: {result['strategies_attempted']}")
    print("="*80 + "\n")

if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_direct_connection())