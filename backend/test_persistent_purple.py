#!/usr/bin/env python3
"""
Test persistent Swift capture with purple indicator
"""

import asyncio
import sys

sys.path.insert(0, '/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')

from vision.swift_video_capture_persistent import (
    start_persistent_video_capture,
    stop_persistent_video_capture,
    is_video_capturing
)

async def test():
    print("\n🟣 Testing Persistent Swift Capture\n")
    print("=" * 60)
    
    # Start capture
    print("1️⃣ Starting persistent video capture...")
    success = await start_persistent_video_capture()
    
    if success:
        print("✅ Capture started!")
        print("🟣 LOOK FOR PURPLE INDICATOR IN MENU BAR!")
        
        # Check status
        is_capturing = await is_video_capturing()
        print(f"\n📊 Is capturing: {is_capturing}")
        
        # Keep running for 15 seconds
        print("\n⏳ Capturing for 15 seconds...")
        for i in range(15):
            await asyncio.sleep(1)
            print(f"   {15-i} seconds remaining...")
            
            # Check status every 5 seconds
            if i % 5 == 0:
                is_capturing = await is_video_capturing()
                print(f"   📊 Still capturing: {is_capturing}")
        
        # Stop capture
        print("\n2️⃣ Stopping video capture...")
        await stop_persistent_video_capture()
        print("✅ Capture stopped - purple indicator should disappear")
        
    else:
        print("❌ Failed to start capture")
    
    print("\n" + "=" * 60)
    print("Test complete!")

if __name__ == "__main__":
    asyncio.run(test())