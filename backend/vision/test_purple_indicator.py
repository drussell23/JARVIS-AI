#!/usr/bin/env python3
"""
Test script to verify macOS purple screen recording indicator
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

async def test_purple_indicator():
    """Test that shows the purple indicator for 30 seconds"""
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer, VisionConfig
    
    print("🎥 Purple Screen Recording Indicator Test")
    print("=" * 50)
    print()
    
    # Configuration
    config = VisionConfig(
        enable_video_streaming=True,
        prefer_video_over_screenshots=True
    )
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        return
        
    analyzer = ClaudeVisionAnalyzer(api_key, config)
    
    try:
        # Start video streaming
        print("Starting native macOS video capture...")
        result = await analyzer.start_video_streaming()
        
        if result['success'] and result['metrics']['capture_method'] == 'macos_native':
            print()
            print("✅ Native macOS video capture started!")
            print("🟣 LOOK FOR THE PURPLE SCREEN RECORDING INDICATOR")
            print("   (Should appear in the menu bar)")
            print()
            print("The indicator will remain visible for 30 seconds...")
            print()
            
            # Keep capturing for 30 seconds
            for i in range(30, 0, -1):
                print(f"\r⏰ Time remaining: {i} seconds  ", end='', flush=True)
                await asyncio.sleep(1)
            
            print("\n\nStopping video capture...")
            await analyzer.stop_video_streaming()
            print("✅ Video capture stopped - purple indicator should disappear")
        else:
            print(f"❌ Failed to start native capture: {result}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await analyzer.cleanup_all_components()
        print("\n✅ Test completed")

if __name__ == "__main__":
    print("This test will activate the macOS screen recording indicator for 30 seconds.")
    print("Make sure to grant screen recording permission if prompted.")
    print()
    input("Press Enter to start...")
    
    asyncio.run(test_purple_indicator())