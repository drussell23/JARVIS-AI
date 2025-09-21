#!/usr/bin/env python3
"""Debug video streaming issue"""

import asyncio
import logging
import os

# Set up logging to see all debug info
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_video_streaming():
    """Test video streaming initialization and startup"""
    
    # Import the vision analyzer
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    
    # Create analyzer instance
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ No ANTHROPIC_API_KEY found")
        return
    
    print("1️⃣ Creating ClaudeVisionAnalyzer...")
    analyzer = ClaudeVisionAnalyzer(api_key)
    
    print(f"2️⃣ Video streaming enabled: {analyzer.config.enable_video_streaming}")
    print(f"   Video streaming config: {analyzer._video_streaming_config}")
    
    # Get video streaming
    print("\n3️⃣ Getting video streaming manager...")
    video_streaming = await analyzer.get_video_streaming()
    print(f"   Video streaming manager: {video_streaming}")
    
    if not video_streaming:
        print("❌ Failed to get video streaming manager")
        return
    
    # Try to start streaming
    print("\n4️⃣ Starting video streaming...")
    try:
        result = await analyzer.start_video_streaming()
        print(f"   Result: {result}")
        
        if result.get('success'):
            print("\n✅ Video streaming started successfully!")
            print("🟣 Purple indicator should be visible")
            
            # Wait a bit
            await asyncio.sleep(5)
            
            # Stop streaming
            print("\n5️⃣ Stopping video streaming...")
            stop_result = await analyzer.stop_video_streaming()
            print(f"   Stop result: {stop_result}")
        else:
            print(f"\n❌ Failed to start: {result.get('error')}")
            
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_video_streaming())