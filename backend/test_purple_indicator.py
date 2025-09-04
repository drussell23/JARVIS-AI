#!/usr/bin/env python3
"""
Test if Swift video capture shows purple indicator
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, '/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')

from vision.swift_video_bridge import SwiftVideoBridge, SwiftCaptureConfig

async def test_purple_indicator():
    """Test if purple indicator appears with Swift capture"""
    print("\n🎥 Testing Swift Video Capture Purple Indicator\n")
    print("=" * 60)
    
    # Create bridge
    config = SwiftCaptureConfig(
        display_id=0,
        fps=30,
        resolution="1920x1080"
    )
    
    bridge = SwiftVideoBridge(config)
    
    # Check permission
    print("1️⃣ Checking screen recording permission...")
    permission_result = await bridge.check_permission()
    print(f"   Permission status: {permission_result.get('permissionStatus')}")
    
    if permission_result.get('permissionStatus') != 'authorized':
        print("\n⚠️  Screen recording permission not granted!")
        print("   Please grant permission in System Preferences")
        return
    
    # Start capture
    print("\n2️⃣ Starting Swift video capture...")
    print("   🟣 LOOK FOR PURPLE INDICATOR IN MENU BAR! 🟣")
    
    start_result = await bridge.start_capture()
    
    if start_result.get('success'):
        print(f"\n✅ Video capture started successfully!")
        print(f"   - Message: {start_result.get('message')}")
        print(f"   - Is Capturing: {start_result.get('isCapturing')}")
        
        print("\n⏳ Capturing for 10 seconds...")
        print("   The purple indicator should be visible in your menu bar")
        
        # Keep capturing for 10 seconds
        for i in range(10):
            await asyncio.sleep(1)
            print(f"   {10-i} seconds remaining...")
            
            # Get status every 3 seconds
            if i % 3 == 0:
                status = await bridge.get_status()
                print(f"   📊 Status: Capturing={status.get('isCapturing')}, Frames={status.get('framesCaptured')}")
        
        # Stop capture
        print("\n3️⃣ Stopping video capture...")
        stop_result = await bridge.stop_capture()
        print(f"   - Success: {stop_result.get('success')}")
        print(f"   - Purple indicator should disappear now")
        
    else:
        print(f"\n❌ Failed to start video capture")
        print(f"   - Error: {start_result.get('error')}")
    
    print("\n" + "=" * 60)
    print("Test complete!")

if __name__ == "__main__":
    asyncio.run(test_purple_indicator())