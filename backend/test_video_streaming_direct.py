#!/usr/bin/env python3
"""
Direct test of video streaming initialization
"""

import asyncio
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add backend directory to path
sys.path.insert(0, '/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')

async def test_video_streaming():
    """Test video streaming directly"""
    print("\n🧪 Direct Video Streaming Test\n")
    print("=" * 60)
    
    try:
        # Import the vision analyzer
        print("1️⃣ Importing Claude Vision Analyzer...")
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        print("✅ Imported successfully")
        
        # Create instance
        print("\n2️⃣ Creating vision analyzer instance...")
        import os
        api_key = os.getenv('ANTHROPIC_API_KEY', '')
        if not api_key:
            print("❌ ANTHROPIC_API_KEY not set")
            return
        analyzer = ClaudeVisionAnalyzer(api_key)
        print("✅ Created successfully")
        
        # Test video streaming initialization
        print("\n3️⃣ Getting video streaming manager...")
        video_streaming = await analyzer.get_video_streaming()
        
        if video_streaming:
            print("✅ Video streaming manager created")
            print(f"   - Type: {type(video_streaming).__name__}")
            print(f"   - Is capturing: {video_streaming.is_capturing}")
            
            # Try to start streaming
            print("\n4️⃣ Starting video streaming...")
            result = await analyzer.start_video_streaming()
            
            print(f"\n📊 Result:")
            print(f"   - Success: {result.get('success', False)}")
            print(f"   - Message: {result.get('message', result.get('error', 'Unknown'))}")
            
            if 'metrics' in result:
                metrics = result['metrics']
                print(f"\n📈 Metrics:")
                print(f"   - Capture method: {metrics.get('capture_method', 'Unknown')}")
                print(f"   - Is capturing: {metrics.get('is_capturing', False)}")
                
            # Stop streaming if started
            if result.get('success'):
                print("\n5️⃣ Stopping video streaming...")
                await analyzer.stop_video_streaming()
                print("✅ Stopped successfully")
                
        else:
            print("❌ Video streaming manager not available")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test complete!")

if __name__ == "__main__":
    asyncio.run(test_video_streaming())