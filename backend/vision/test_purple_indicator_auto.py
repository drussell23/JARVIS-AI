#!/usr/bin/env python3
"""
Test script to verify macOS purple screen recording indicator (non-interactive)
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

async def test_purple_indicator():
    """Test that shows the purple indicator for 10 seconds"""
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer, VisionConfig
    
    print("🎥 Purple Screen Recording Indicator Test (Automatic)")
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
            print("The indicator will remain visible for 10 seconds...")
            print()
            
            # Keep capturing for 10 seconds
            for i in range(10, 0, -1):
                print(f"\r⏰ Time remaining: {i} seconds  ", end='', flush=True)
                await asyncio.sleep(1)
            
            print("\n\nStopping video capture...")
            await analyzer.stop_video_streaming()
            print("✅ Video capture stopped - purple indicator should disappear")
        else:
            print(f"❌ Failed to start native capture")
            print(f"   Capture method: {result['metrics'].get('capture_method', 'unknown')}")
            if result['metrics']['capture_method'] != 'macos_native':
                print("   ℹ️  Using fallback mode - no purple indicator")
                print("   To enable native capture, install: pip3 install pyobjc")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await analyzer.cleanup_all_components()
        print("\n✅ Test completed")

if __name__ == "__main__":
    print("This test will activate the macOS screen recording indicator for 10 seconds.")
    print("Starting automatically...")
    print()
    
    asyncio.run(test_purple_indicator())