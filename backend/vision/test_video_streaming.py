#!/usr/bin/env python3
"""
Test script for video streaming functionality
Tests real-time video capture with Claude Vision analysis
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_video_streaming():
    """Test video streaming with Claude Vision"""
    
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer, VisionConfig
    
    # Configuration for video streaming
    config = VisionConfig(
        enable_video_streaming=True,
        prefer_video_over_screenshots=True,
        enable_continuous_monitoring=False,  # We'll use video instead
        max_concurrent_requests=5,
        process_memory_limit_mb=2048,  # 2GB limit
        memory_threshold_percent=60
    )
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return
    
    analyzer = ClaudeVisionAnalyzer(api_key, config)
    
    try:
        logger.info("=== Testing Video Streaming Integration ===")
        
        # 1. Check memory health
        health = await analyzer.check_memory_health()
        logger.info(f"Memory health: {health}")
        
        if not health['healthy']:
            logger.error("Memory not healthy, aborting test")
            return
        
        # 2. Start video streaming
        logger.info("\n--- Starting video streaming ---")
        result = await analyzer.start_video_streaming()
        logger.info(f"Start result: {result}")
        
        if not result['success']:
            logger.error("Failed to start video streaming")
            return
        
        logger.info("\n🎥 VIDEO STREAMING ACTIVE - Check for macOS screen recording indicator!")
        
        # 3. Get status
        await asyncio.sleep(2)
        status = await analyzer.get_video_streaming_status()
        logger.info(f"\nVideo status: {status}")
        
        # 4. Test real-time analysis
        logger.info("\n--- Testing real-time video analysis ---")
        
        # Register a callback to see frames being analyzed
        analyzed_count = {'count': 0}
        
        async def on_frame_analyzed(data):
            analyzed_count['count'] += 1
            logger.info(f"Frame {data['frame_number']} analyzed at {data.get('timestamp', 'unknown')}")
        
        video = await analyzer.get_video_streaming()
        if video:
            video.register_callback('frame_analyzed', on_frame_analyzed)
        
        # Analyze for 10 seconds
        logger.info("Analyzing video stream for 10 seconds...")
        analysis_result = await analyzer.analyze_video_stream(
            "What is happening on the screen? Describe any changes or movement.",
            duration_seconds=10.0
        )
        
        logger.info(f"\nAnalysis complete:")
        logger.info(f"- Frames analyzed: {analysis_result.get('frames_analyzed', 0)}")
        logger.info(f"- Total callbacks: {analyzed_count['count']}")
        
        # Show some results
        if 'results' in analysis_result and analysis_result['results']:
            logger.info("\nSample analyses:")
            for i, result in enumerate(analysis_result['results'][:3]):
                logger.info(f"\nFrame {result['frame_number']} at {result['timestamp']:.1f}s:")
                logger.info(f"Analysis: {str(result['analysis'])[:200]}...")
        
        # 5. Test switching capture methods
        logger.info("\n--- Testing capture method switching ---")
        
        # Test getting a screenshot while video is running
        screenshot = await analyzer.capture_screen()
        logger.info(f"Screenshot captured while streaming: {screenshot.size if screenshot else 'None'}")
        
        # 6. Check memory usage
        final_stats = analyzer.get_all_memory_stats()
        logger.info(f"\nFinal memory stats:")
        logger.info(f"- Process: {final_stats['system']['process_mb']:.1f}MB")
        logger.info(f"- Video streaming: {final_stats['components'].get('video_streaming', {})}")
        
        # 7. Stop video streaming
        logger.info("\n--- Stopping video streaming ---")
        stop_result = await analyzer.stop_video_streaming()
        logger.info(f"Stop result: {stop_result}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    finally:
        # Cleanup
        await analyzer.cleanup_all_components()
        logger.info("\nTest completed - all components cleaned up")

async def test_video_motion_detection():
    """Test motion detection in video streaming"""
    
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer, VisionConfig
    
    config = VisionConfig(
        enable_video_streaming=True,
        prefer_video_over_screenshots=True
    )
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    analyzer = ClaudeVisionAnalyzer(api_key, config)
    
    try:
        logger.info("\n=== Testing Motion Detection ===")
        
        # Start video
        await analyzer.start_video_streaming()
        
        # Register motion callback
        motion_events = []
        
        async def on_motion(data):
            motion_events.append(data)
            logger.info(f"Motion detected! Score: {data['motion_score']:.2f}")
        
        video = await analyzer.get_video_streaming()
        if video:
            video.register_callback('motion_detected', on_motion)
        
        logger.info("Move something on screen to trigger motion detection...")
        await asyncio.sleep(10)
        
        logger.info(f"\nMotion events detected: {len(motion_events)}")
        
        await analyzer.stop_video_streaming()
        
    finally:
        await analyzer.cleanup_all_components()

if __name__ == "__main__":
    # Run main test
    asyncio.run(test_video_streaming())
    
    # Optionally run motion test
    # asyncio.run(test_video_motion_detection())