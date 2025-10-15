#!/usr/bin/env python3
"""
Test script to diagnose video streaming permissions issue
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_video_streaming():
    print("\n🔍 Testing Video Streaming Permissions")
    print("=" * 60)
    
    # Test 1: Check if direct Swift capture works
    print("\n1️⃣ Testing Direct Swift Capture...")
    try:
        from vision.direct_swift_capture import start_direct_swift_capture, stop_direct_swift_capture
        
        success = await start_direct_swift_capture()
        if success:
            print("✅ Direct Swift capture started - purple indicator should be visible!")
            await asyncio.sleep(3)
            stop_direct_swift_capture()
            print("✅ Direct Swift capture stopped")
        else:
            print("❌ Direct Swift capture failed")
    except Exception as e:
        print(f"❌ Direct Swift capture error: {e}")
    
    # Test 2: Check persistent capture Swift script
    print("\n2️⃣ Testing Swift Script Directly...")
    try:
        import subprocess
        swift_script = "vision/persistent_capture.swift"
        
        if os.path.exists(swift_script):
            # Test if Swift can access screen recording
            result = subprocess.run(
                ["swift", swift_script, "--test"],
                capture_output=True,
                text=True,
                timeout=5
            )
            print(f"Swift output: {result.stdout}")
            if result.stderr:
                print(f"Swift errors: {result.stderr}")
        else:
            print(f"❌ Swift script not found at {swift_script}")
    except Exception as e:
        print(f"❌ Swift test error: {e}")
    
    # Test 3: Check macOS permissions
    print("\n3️⃣ Checking macOS Permissions...")
    try:
        import subprocess
        
        # Check if terminal has screen recording permission
        result = subprocess.run(
            ["tccutil", "check", "ScreenCapture"],
            capture_output=True,
            text=True
        )
        print(f"Screen recording permission check: {result.stdout or 'No output'}")
    except:
        print("ℹ️  Cannot check permissions directly (tccutil requires admin)")
    
    # Test 4: Test video streaming initialization
    print("\n4️⃣ Testing Video Streaming Module...")
    try:
        from vision.video_stream_capture import VideoStreamCapture
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        
        # Create analyzer
        api_key = os.getenv('ANTHROPIC_API_KEY', 'test-key')
        analyzer = ClaudeVisionAnalyzer(api_key, enable_realtime=True)
        
        # Get video streaming
        video_streaming = await analyzer.get_video_streaming()
        
        if video_streaming:
            print("✅ Video streaming module created")
            
            # Try to start
            print("Attempting to start video streaming...")
            result = await analyzer.start_video_streaming()
            
            print(f"Result: {result}")
            
            if result.get('success'):
                print("✅ Video streaming started successfully!")
                await asyncio.sleep(3)
                await analyzer.stop_video_streaming()
            else:
                print(f"❌ Video streaming failed: {result.get('error')}")
        else:
            print("❌ Failed to create video streaming module")
            
    except Exception as e:
        print(f"❌ Video streaming test error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Check environment variables
    print("\n5️⃣ Checking Environment Variables...")
    env_vars = [
        'VISION_VIDEO_STREAMING',
        'VISION_PREFER_VIDEO',
        'VIDEO_STREAM_FPS',
        'VIDEO_STREAM_RESOLUTION'
    ]
    
    for var in env_vars:
        value = os.getenv(var, 'not set')
        print(f"  {var}: {value}")
    
    print("\n" + "=" * 60)
    print("📋 Diagnosis Complete!")
    print("\n🔧 Common Fixes:")
    print("1. Grant screen recording permission:")
    print("   System Preferences > Security & Privacy > Screen Recording")
    print("   ✓ Terminal.app (or your terminal)")
    print("   ✓ Python")
    print("\n2. Reset permissions if needed:")
    print("   tccutil reset ScreenCapture")
    print("\n3. Try the purple indicator test:")
    print("   cd backend")
    print("   python test_purple_indicator.py")

if __name__ == "__main__":
    asyncio.run(test_video_streaming())