#!/usr/bin/env python3
"""
Test Purple Indicator and Vision Status Integration
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vision.direct_swift_capture import DirectSwiftCapture
from vision.vision_status_manager import get_vision_status_manager

async def test_purple_indicator_status():
    """Test that purple indicator stays on and vision status updates"""
    print("\n🟣 TESTING PURPLE INDICATOR + VISION STATUS")
    print("=" * 60)
    
    # Get vision status manager
    status_manager = get_vision_status_manager()
    
    # Set up status change monitoring
    status_changes = []
    
    def status_callback(connected: bool):
        status = "CONNECTED" if connected else "DISCONNECTED"
        print(f"\n📡 Vision Status Changed: {status}")
        status_changes.append((connected, asyncio.get_event_loop().time()))
    
    status_manager.add_status_callback(status_callback)
    
    # Create direct capture instance
    capture = DirectSwiftCapture()
    
    # Set up vision status callback
    async def vision_status_callback(connected: bool):
        await status_manager.update_vision_status(connected)
    
    capture.set_vision_status_callback(vision_status_callback)
    
    # Start capture
    print("\n🚀 Starting capture with purple indicator...")
    success = await capture.start_capture()
    
    if not success:
        print("❌ Failed to start capture")
        return
    
    print("✅ Capture started successfully")
    print("🟣 Purple indicator should be visible")
    print("📡 Waiting for vision status update...\n")
    
    # Monitor for 30 seconds
    start_time = asyncio.get_event_loop().time()
    check_interval = 5  # seconds
    total_duration = 30  # seconds
    
    while asyncio.get_event_loop().time() - start_time < total_duration:
        await asyncio.sleep(check_interval)
        
        elapsed = asyncio.get_event_loop().time() - start_time
        current_status = status_manager.get_status()
        
        print(f"⏱️  Time: {elapsed:.0f}s")
        print(f"   Status: {current_status['text']}")
        print(f"   Color: {current_status['color']}")
        print(f"   Indicator: {current_status['indicator']}")
        print(f"   Is Capturing: {capture.is_capturing}")
        print()
    
    # Stop capture
    print("\n🛑 Stopping capture...")
    capture.stop_capture()
    
    # Wait for status update
    await asyncio.sleep(2)
    
    # Check final status
    final_status = status_manager.get_status()
    print("\n📊 Final Status:")
    print(f"   Connected: {final_status['connected']}")
    print(f"   Text: {final_status['text']}")
    print(f"   Status Changes: {len(status_changes)}")
    
    # Analyze results
    print("\n📈 Test Results:")
    if len(status_changes) >= 2:
        print("✅ Vision status updates working correctly")
        for i, (connected, time) in enumerate(status_changes):
            print(f"   Change {i+1}: {'Connected' if connected else 'Disconnected'} at {time:.1f}s")
    else:
        print("⚠️  Expected at least 2 status changes (connect/disconnect)")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_purple_indicator_status())