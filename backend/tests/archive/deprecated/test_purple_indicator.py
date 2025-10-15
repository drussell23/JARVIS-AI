#!/usr/bin/env python3
"""
Test script to verify purple indicator functionality
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vision.direct_swift_capture import start_direct_swift_capture, stop_direct_swift_capture

async def test_purple_indicator():
    print("\n🟣 Testing Purple Indicator")
    print("=" * 60)
    print("\nThis test will show the macOS purple recording indicator.")
    print("The indicator should appear in your menu bar (top-right).\n")
    
    # Start capture
    print("Starting direct Swift capture...")
    success = await start_direct_swift_capture()
    
    if success:
        print("✅ Capture started successfully!")
        print("🟣 CHECK YOUR MENU BAR - Purple indicator should be visible!\n")
        
        print("The indicator will stay on for 10 seconds...")
        for i in range(10, 0, -1):
            print(f"\rTime remaining: {i} seconds   ", end='', flush=True)
            await asyncio.sleep(1)
        
        print("\n\nStopping capture...")
        stop_direct_swift_capture()
        print("✅ Capture stopped - purple indicator should disappear!")
        print("\nTest completed successfully! 🎉")
    else:
        print("❌ Failed to start capture")
        print("Make sure you have granted screen recording permission.")

if __name__ == "__main__":
    asyncio.run(test_purple_indicator())