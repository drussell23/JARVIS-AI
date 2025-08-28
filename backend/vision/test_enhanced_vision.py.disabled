"""
Test script to demonstrate the enhanced vision system performance
Shows side-by-side comparison of Python vs C++ capture speeds
"""

import asyncio
import time
import numpy as np
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import both implementations
try:
    from enhanced_screen_vision import EnhancedScreenVisionSystem
    from screen_vision import ScreenVisionSystem as OriginalScreenVisionSystem
except ImportError:
    logger.error("Could not import vision systems")
    exit(1)


async def benchmark_single_capture(vision_system, name: str, iterations: int = 10):
    """Benchmark single screen capture"""
    logger.info(f"\n{'='*50}")
    logger.info(f"Benchmarking {name} - Single Screen Capture")
    logger.info(f"{'='*50}")
    
    times = []
    
    # Warmup
    for _ in range(3):
        await vision_system.capture_screen()
    
    # Benchmark
    for i in range(iterations):
        start = time.perf_counter()
        image = await vision_system.capture_screen()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        if i == 0:
            logger.info(f"  Image shape: {image.shape}")
            logger.info(f"  Image size: {image.nbytes / (1024*1024):.2f} MB")
    
    # Calculate statistics
    avg_time = np.mean(times) * 1000  # Convert to ms
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    std_time = np.std(times) * 1000
    fps = 1000 / avg_time
    
    logger.info(f"\nResults for {name}:")
    logger.info(f"  Average: {avg_time:.2f} ms ({fps:.1f} FPS)")
    logger.info(f"  Min: {min_time:.2f} ms")
    logger.info(f"  Max: {max_time:.2f} ms")
    logger.info(f"  Std Dev: {std_time:.2f} ms")
    
    return {
        'name': name,
        'avg_ms': avg_time,
        'min_ms': min_time,
        'max_ms': max_time,
        'fps': fps
    }


async def benchmark_multi_window_capture(vision_system, name: str):
    """Benchmark multi-window capture"""
    logger.info(f"\n{'='*50}")
    logger.info(f"Benchmarking {name} - Multi-Window Capture")
    logger.info(f"{'='*50}")
    
    # Get window list
    windows = await vision_system.get_window_list(visible_only=True)
    logger.info(f"  Found {len(windows)} visible windows")
    
    if not windows:
        logger.warning("  No windows found to capture")
        return None
    
    # Benchmark capturing all visible windows
    start = time.perf_counter()
    
    if hasattr(vision_system, 'capture_multiple_windows'):
        # Use the new parallel capture method
        captures = await vision_system.capture_multiple_windows(visible_only=True)
        elapsed = time.perf_counter() - start
        
        logger.info(f"\nResults for {name}:")
        logger.info(f"  Captured {len(captures)} windows in {elapsed*1000:.2f} ms")
        logger.info(f"  Average per window: {elapsed*1000/len(captures):.2f} ms")
        
        # Show individual capture times if available
        if captures and 'capture_time_ms' in captures[0]:
            for i, capture in enumerate(captures[:5]):  # Show first 5
                logger.info(f"    Window {i+1} ({capture['app_name']}): {capture['capture_time_ms']:.2f} ms")
    else:
        # Fallback to sequential capture
        logger.info(f"  Sequential capture not implemented for {name}")
    
    return elapsed * 1000 if 'elapsed' in locals() else None


async def test_real_world_scenario():
    """Test a real-world scenario: monitoring for updates"""
    logger.info(f"\n{'='*50}")
    logger.info("Real-World Test: Continuous Window Monitoring")
    logger.info(f"{'='*50}")
    
    # Initialize enhanced system
    vision = EnhancedScreenVisionSystem(use_fast_capture=True)
    
    logger.info("\nMonitoring windows for 5 seconds...")
    
    start_time = time.time()
    capture_count = 0
    
    while time.time() - start_time < 5:
        # Capture all visible windows
        windows = await vision.get_window_list(visible_only=True)
        
        # Capture frontmost window
        image = await vision.capture_screen()
        
        capture_count += 1
        
        # Small delay to simulate processing
        await asyncio.sleep(0.1)
    
    elapsed = time.time() - start_time
    avg_captures_per_sec = capture_count / elapsed
    
    logger.info(f"\nMonitoring Results:")
    logger.info(f"  Total captures: {capture_count}")
    logger.info(f"  Average rate: {avg_captures_per_sec:.1f} captures/second")
    
    # Get performance stats
    stats = vision.get_performance_stats()
    logger.info(f"\nPerformance Statistics:")
    logger.info(f"  Average capture time: {stats['avg_capture_time']*1000:.2f} ms")
    logger.info(f"  Min capture time: {stats['min_capture_time']*1000:.2f} ms")
    logger.info(f"  Max capture time: {stats['max_capture_time']*1000:.2f} ms")
    
    if 'cpp_avg_ms' in stats:
        logger.info(f"\nC++ Engine Statistics:")
        logger.info(f"  Average time: {stats['cpp_avg_ms']:.2f} ms")
        logger.info(f"  Total captures: {stats['cpp_total_captures']}")
        logger.info(f"  GPU accelerated: {stats['cpp_gpu_captures']}")


async def main():
    """Main test function"""
    logger.info("\n" + "="*70)
    logger.info("JARVIS Vision System Performance Comparison")
    logger.info("Python (Quartz) vs C++ (Fast Capture)")
    logger.info("="*70)
    
    # Initialize both systems
    enhanced_vision = EnhancedScreenVisionSystem(use_fast_capture=True)
    fallback_vision = EnhancedScreenVisionSystem(use_fast_capture=False)  # Force Python mode
    
    # Run single capture benchmarks
    results = []
    
    # Benchmark C++ implementation
    if enhanced_vision.use_fast_capture:
        cpp_result = await benchmark_single_capture(enhanced_vision, "C++ Fast Capture")
        results.append(cpp_result)
    
    # Benchmark Python implementation
    python_result = await benchmark_single_capture(fallback_vision, "Python (Quartz)")
    results.append(python_result)
    
    # Compare results
    if len(results) == 2:
        speedup = results[1]['avg_ms'] / results[0]['avg_ms']
        logger.info(f"\n{'='*50}")
        logger.info(f"Performance Improvement: {speedup:.1f}x faster!")
        logger.info(f"{'='*50}")
    
    # Test multi-window capture
    if enhanced_vision.use_fast_capture:
        await benchmark_multi_window_capture(enhanced_vision, "C++ Fast Capture")
    
    # Test real-world scenario
    await test_real_world_scenario()
    
    logger.info("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())