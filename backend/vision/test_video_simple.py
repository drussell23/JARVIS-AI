#!/usr/bin/env python3
"""
Simple video streaming test that demonstrates the integration
"""

import asyncio
import os
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_video_streaming_simple():
    """Simple test to demonstrate video streaming integration"""
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer, VisionConfig
    
    # Configuration
    config = VisionConfig(
        enable_video_streaming=True,
        prefer_video_over_screenshots=True,
        process_memory_limit_mb=2048
    )
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("Please set ANTHROPIC_API_KEY environment variable")
        return
        
    analyzer = ClaudeVisionAnalyzer(api_key, config)
    
    try:
        logger.info("=== Video Streaming Integration Test ===")
        
        # 1. Start video streaming
        logger.info("\n1. Starting video streaming...")
        result = await analyzer.start_video_streaming()
        
        if result['success']:
            logger.info("✅ Video streaming started successfully!")
            logger.info(f"   Capture method: {result['metrics']['capture_method']}")
            
            # Note about macOS indicator
            capture_method = result['metrics']['capture_method']
            if capture_method == 'macos_native':
                logger.info("   🟣 Check for purple screen recording indicator on macOS")
            elif capture_method == 'native':
                logger.info("   🟣 Using native capture (may show indicator)")
            else:
                logger.info("   ⚠️  Using fallback mode - no purple indicator")
        else:
            logger.error(f"❌ Failed to start: {result.get('message', 'Unknown error')}")
            logger.error(f"   Error details: {result.get('error', 'No details')}")
            return
            
        # 2. Check video streaming is active
        await asyncio.sleep(1)
        status = await analyzer.get_video_streaming_status()
        logger.info(f"\n2. Video status: Active = {status['active']}")
        
        # 3. Capture and analyze a frame
        logger.info("\n3. Testing capture and analysis...")
        
        # Capture screen (will use video frame if available)
        screenshot = await analyzer.capture_screen()
        if screenshot:
            logger.info(f"   ✅ Captured frame: {screenshot.size}")
            
            # Analyze it
            try:
                result = await analyzer.analyze_screenshot(
                    screenshot, 
                    "What application is currently in focus?"
                )
                
                # Handle both tuple and dict returns
                if isinstance(result, tuple):
                    success, response = result
                    if success:
                        logger.info(f"   ✅ Analysis result: {str(response)[:100]}...")
                    else:
                        logger.error(f"   ❌ Analysis failed: {response}")
                elif isinstance(result, dict) and result.get('success'):
                    logger.info(f"   ✅ Analysis result: {result['description'][:100]}...")
                else:
                    logger.error(f"   ❌ Analysis failed: {result}")
            except Exception as e:
                logger.error(f"   ❌ Analysis error: {e}")
        else:
            logger.error("   ❌ Failed to capture frame")
            
        # 4. Test switching between video/screenshot mode
        logger.info("\n4. Testing mode switching...")
        
        # Switch to screenshot mode
        await analyzer.switch_to_screenshot_mode()
        logger.info("   Switched to screenshot mode")
        
        # Switch back to video
        await analyzer.switch_to_video_mode()
        logger.info("   Switched back to video mode")
        
        # 5. Stop video streaming
        logger.info("\n5. Stopping video streaming...")
        stop_result = await analyzer.stop_video_streaming()
        if stop_result['success']:
            logger.info("✅ Video streaming stopped successfully")
            # Check the initial result for capture method
            video_status = await analyzer.get_video_streaming_status()
            if not video_status['active'] and 'native' in str(video_status):
                logger.info("   🟣 Purple indicator should have disappeared")
        
        # 6. Show memory usage
        stats = analyzer.get_all_memory_stats()
        logger.info(f"\nMemory usage: {stats['system']['process_mb']:.1f}MB")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
    finally:
        await analyzer.cleanup_all_components()
        logger.info("\n✅ Test completed - all components cleaned up")

if __name__ == "__main__":
    asyncio.run(test_video_streaming_simple())