#!/usr/bin/env python3
"""
Test JARVIS purple indicator fix
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_video_streaming():
    print("\n🟣 Testing JARVIS Video Streaming with Purple Indicator")
    print("=" * 60)
    
    # Import after path setup
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    
    # Get API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set")
        return
        
    # Create analyzer
    analyzer = ClaudeVisionAnalyzer(api_key)
    
    print("1️⃣ Starting video streaming...")
    result = await analyzer.start_video_streaming()
    
    print(f"\nResult: {result}")
    
    if result.get('success'):
        print("✅ Video streaming started successfully!")
        print("🟣 CHECK YOUR MENU BAR FOR PURPLE INDICATOR!")
        
        metrics = result.get('metrics', {})
        print(f"\nCapture method: {metrics.get('capture_method')}")
        print(f"Is capturing: {metrics.get('is_capturing')}")
        
        print("\n⏳ Monitoring for 10 seconds...")
        await asyncio.sleep(10)
        
        print("\n2️⃣ Stopping video streaming...")
        stop_result = await analyzer.stop_video_streaming()
        print(f"Stop result: {stop_result}")
        
    else:
        print(f"❌ Failed to start video streaming: {result.get('error')}")
        
        # Try to get more info
        if hasattr(analyzer, 'video_streaming'):
            print(f"Video streaming exists: {analyzer.video_streaming is not None}")
    
    print("\n" + "=" * 60)
    print("Test complete!")

if __name__ == "__main__":
    asyncio.run(test_video_streaming())