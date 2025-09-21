#!/usr/bin/env python3
"""
Test direct Swift capture - should show purple indicator
"""

import asyncio
import sys

sys.path.insert(0, '/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')

from vision.direct_swift_capture import (
    start_direct_swift_capture, 
    stop_direct_swift_capture,
    is_direct_capturing
)

async def test():
    print("\n🟣 TESTING DIRECT SWIFT CAPTURE")
    print("=" * 60)
    print("This should show the purple recording indicator immediately!\n")
    
    # Start capture
    print("1️⃣ Starting direct Swift capture...")
    success = await start_direct_swift_capture()
    
    if success:
        print("✅ Capture started!")
        print("🟣 LOOK FOR PURPLE INDICATOR IN MENU BAR NOW!")
        print(f"📊 Is capturing: {is_direct_capturing()}")
        
        # Keep running for 15 seconds
        print("\n⏳ Recording for 15 seconds...")
        for i in range(15):
            await asyncio.sleep(1)
            print(f"   {15-i} seconds remaining...")
        
        # Stop capture
        print("\n2️⃣ Stopping capture...")
        stop_direct_swift_capture()
        print("✅ Capture stopped - purple indicator should disappear")
        
    else:
        print("❌ Failed to start capture")
        print("Make sure you have screen recording permissions enabled!")
    
    print("\n" + "=" * 60)
    print("Test complete!")

if __name__ == "__main__":
    asyncio.run(test())