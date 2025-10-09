#!/usr/bin/env python3
"""
Test Swift video capture permissions and functionality
"""

import asyncio
import logging
import sys
import os

# Add backend to path
sys.path.insert(0, '/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')

from vision.swift_video_bridge import SwiftVideoBridge, SwiftCaptureConfig

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_swift_permissions():
    """Test Swift video capture permissions"""
    print("\n🧪 Swift Video Capture Permission Test\n")
    print("=" * 60)
    
    # Create bridge
    config = SwiftCaptureConfig(
        display_id=0,
        fps=30,
        resolution="1920x1080"
    )
    
    bridge = SwiftVideoBridge(config)
    
    # Test 1: Check permission status
    print("\n1️⃣ Checking current permission status...")
    permission_result = await bridge.check_permission()
    
    print(f"✅ Permission check completed:")
    print(f"   - Status: {permission_result.get('permissionStatus', 'Unknown')}")
    print(f"   - Message: {permission_result.get('message', '')}")
    
    current_status = permission_result.get('permissionStatus', '')
    
    # Test 2: Request permission if needed
    if current_status != 'authorized':
        print("\n2️⃣ Requesting screen recording permission...")
        print("   ⚠️  Please check the permission dialog that appears")
        print("   ⚠️  You may need to go to System Preferences > Security & Privacy > Privacy > Screen Recording")
        
        request_result = await bridge.request_permission()
        
        print(f"\n✅ Permission request completed:")
        print(f"   - Success: {request_result.get('success', False)}")
        print(f"   - Message: {request_result.get('message', '')}")
        
        if request_result.get('error'):
            print(f"   - Error: {request_result.get('error')}")
    else:
        print("\n2️⃣ Screen recording permission already granted ✅")
    
    # Test 3: Try to start capture
    print("\n3️⃣ Testing video capture start...")
    start_result = await bridge.start_capture()
    
    print(f"\n✅ Capture start result:")
    print(f"   - Success: {start_result.get('success', False)}")
    print(f"   - Message: {start_result.get('message', '')}")
    print(f"   - Is Capturing: {start_result.get('isCapturing', False)}")
    
    if start_result.get('error'):
        print(f"   - Error: {start_result.get('error')}")
    
    if start_result.get('success'):
        # Test 4: Get status
        print("\n4️⃣ Getting capture status...")
        await asyncio.sleep(2)  # Let it capture for 2 seconds
        
        status_result = await bridge.get_status()
        print(f"\n✅ Capture status:")
        print(f"   - Is Capturing: {status_result.get('isCapturing', False)}")
        print(f"   - Frames Captured: {status_result.get('framesCaptured', 0)}")
        
        # Test 5: Stop capture
        print("\n5️⃣ Stopping video capture...")
        stop_result = await bridge.stop_capture()
        
        print(f"\n✅ Capture stop result:")
        print(f"   - Success: {stop_result.get('success', False)}")
        print(f"   - Message: {stop_result.get('message', '')}")
    
    print("\n" + "=" * 60)
    
    # Summary
    if start_result.get('success'):
        print("\n🎉 Success! Swift video capture is working properly.")
        print("   The purple recording indicator should have appeared in your menu bar.")
    else:
        print("\n⚠️  Swift video capture could not start.")
        print("\n📝 To fix this:")
        print("   1. Open System Preferences")
        print("   2. Go to Security & Privacy > Privacy > Screen Recording")
        print("   3. Add Terminal (or your Python app) to the list")
        print("   4. Check the checkbox next to it")
        print("   5. Restart your terminal and try again")

if __name__ == "__main__":
    asyncio.run(test_swift_permissions())