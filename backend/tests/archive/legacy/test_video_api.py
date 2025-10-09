#!/usr/bin/env python3
"""
Quick test to verify video streaming is available in the API
"""

import asyncio
import os
from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer, VisionConfig

async def test_video_api():
    """Test video streaming through the API"""
    print("Testing Video Streaming API Integration")
    print("=" * 40)
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set")
        return
    
    # Create analyzer with video enabled
    config = VisionConfig(
        enable_video_streaming=True,
        prefer_video_over_screenshots=True
    )
    
    analyzer = ClaudeVisionAnalyzer(api_key, config)
    
    # Test 1: Check if video streaming is available
    print("\n1. Checking video streaming availability...")
    try:
        from vision.video_stream_capture import MACOS_CAPTURE_AVAILABLE
        if MACOS_CAPTURE_AVAILABLE:
            print("✅ Native macOS capture available")
        else:
            print("⚠️ Using fallback capture mode")
    except Exception as e:
        print(f"❌ Error checking capture: {e}")
    
    # Test 2: Start video streaming
    print("\n2. Starting video streaming...")
    try:
        result = await analyzer.start_video_streaming()
        if result['success']:
            print(f"✅ Video streaming started")
            print(f"   Method: {result['metrics']['capture_method']}")
            if result['metrics']['capture_method'] == 'macos_native':
                print("   🟣 Purple indicator should be visible")
            
            # Test 3: Capture a frame
            print("\n3. Capturing a frame...")
            frame = await analyzer.capture_screen()
            if frame:
                print(f"✅ Frame captured: {frame.size}")
            
            # Stop streaming
            await analyzer.stop_video_streaming()
            print("\n✅ Video streaming stopped")
        else:
            print("❌ Failed to start video streaming")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await analyzer.cleanup_all_components()
    
    print("\n✅ Test completed")

if __name__ == "__main__":
    asyncio.run(test_video_api())