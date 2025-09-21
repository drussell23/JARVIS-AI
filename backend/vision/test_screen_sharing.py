#!/usr/bin/env python3
"""
Test script for screen sharing integration
Tests memory safety and integration with vision system
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_screen_sharing():
    """Test screen sharing integration with vision system"""
    
    # Import the vision analyzer
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer, VisionConfig
    
    # Get API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return
    
    # Create vision analyzer with screen sharing enabled
    config = VisionConfig(
        enable_screen_sharing=True,
        enable_continuous_monitoring=True,
        max_concurrent_requests=5,  # Reduced for testing
        memory_threshold_percent=50,  # More aggressive for testing
        process_memory_limit_mb=1024  # 1GB limit for testing
    )
    
    analyzer = ClaudeVisionAnalyzer(api_key, config)
    
    try:
        logger.info("=== Testing Screen Sharing Integration ===")
        
        # 1. Check initial memory status
        memory_health = await analyzer.check_memory_health()
        logger.info(f"Initial memory health: {memory_health}")
        
        # 2. Start continuous monitoring
        logger.info("\n--- Starting continuous monitoring ---")
        monitoring_started = await analyzer.start_continuous_monitoring()
        logger.info(f"Continuous monitoring started: {monitoring_started}")
        
        # Wait a bit for monitoring to initialize
        await asyncio.sleep(2)
        
        # 3. Test screen capture
        logger.info("\n--- Testing screen capture ---")
        screenshot = await analyzer.capture_screen()
        if screenshot:
            logger.info(f"Screen captured successfully: {screenshot.size}")
        else:
            logger.warning("Screen capture returned None")
        
        # 4. Start screen sharing
        logger.info("\n--- Starting screen sharing ---")
        sharing_result = await analyzer.start_screen_sharing()
        logger.info(f"Screen sharing result: {sharing_result}")
        
        if sharing_result['success']:
            # 5. Get sharing status
            await asyncio.sleep(2)
            status = await analyzer.get_screen_sharing_status()
            logger.info(f"\nScreen sharing status: {status}")
            
            # 6. Let it run for a bit to test memory management
            logger.info("\n--- Running screen sharing for 10 seconds ---")
            for i in range(10):
                await asyncio.sleep(1)
                if i % 3 == 0:
                    metrics = status = await analyzer.get_screen_sharing_status()
                    logger.info(f"Metrics at {i}s: FPS={metrics['metrics']['current_fps']:.1f}, "
                              f"Memory={metrics['metrics']['memory_usage_mb']:.1f}MB")
            
            # 7. Test memory stats
            logger.info("\n--- Memory Statistics ---")
            all_stats = analyzer.get_all_memory_stats()
            for component, stats in all_stats['components'].items():
                logger.info(f"{component}: {stats}")
            
            # 8. Stop screen sharing
            logger.info("\n--- Stopping screen sharing ---")
            stop_result = await analyzer.stop_screen_sharing()
            logger.info(f"Stop result: {stop_result}")
        
        # 9. Final memory check
        logger.info("\n--- Final memory health check ---")
        final_memory_health = await analyzer.check_memory_health()
        logger.info(f"Final memory health: {final_memory_health}")
        
        # 10. Cleanup
        logger.info("\n--- Cleaning up ---")
        await analyzer.cleanup_all_components()
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
    finally:
        # Ensure cleanup
        if 'analyzer' in locals():
            await analyzer.cleanup_all_components()

async def test_memory_pressure():
    """Test screen sharing behavior under memory pressure"""
    
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer, VisionConfig
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return
    
    # Create analyzer with very strict memory limits
    config = VisionConfig(
        enable_screen_sharing=True,
        process_memory_limit_mb=512,  # Very low limit
        memory_warning_threshold_mb=400,
        memory_threshold_percent=40,
        reject_on_memory_pressure=True
    )
    
    analyzer = ClaudeVisionAnalyzer(api_key, config)
    
    try:
        logger.info("\n=== Testing Memory Pressure Handling ===")
        
        # Try to start screen sharing with low memory
        result = await analyzer.start_screen_sharing()
        logger.info(f"Start result under memory pressure: {result}")
        
        if result['success']:
            # Monitor quality adjustments
            logger.info("\n--- Monitoring quality adjustments ---")
            for i in range(5):
                await asyncio.sleep(2)
                status = await analyzer.get_screen_sharing_status()
                metrics = status['metrics']
                logger.info(f"Quality: {metrics['current_quality']}, "
                          f"FPS: {metrics['current_fps']}, "
                          f"Resolution: {metrics['current_resolution']}, "
                          f"Adjustments: {metrics['quality_adjustments']}")
            
            await analyzer.stop_screen_sharing()
        
    except Exception as e:
        logger.error(f"Memory pressure test failed: {e}", exc_info=True)
    finally:
        await analyzer.cleanup_all_components()

if __name__ == "__main__":
    # Run tests
    asyncio.run(test_screen_sharing())
    
    # Uncomment to test memory pressure handling
    # asyncio.run(test_memory_pressure())