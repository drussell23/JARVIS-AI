#!/usr/bin/env python3
"""
Example of using Claude Vision Analyzer with memory safety features
Shows how to configure and use the analyzer to prevent crashes
"""

import asyncio
import os
import numpy as np
from PIL import Image
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Example usage with memory safety"""
    
    # 1. Configure via environment variables (no hardcoding!)
    # These can be set in your .env file or system environment
    
    # Memory safety configuration
    os.environ['VISION_MEMORY_SAFETY'] = 'true'  # Enable memory safety
    os.environ['VISION_PROCESS_LIMIT_MB'] = '2048'  # 2GB limit
    os.environ['VISION_MIN_SYSTEM_RAM_GB'] = '2.0'  # Min 2GB system RAM
    os.environ['VISION_REJECT_ON_MEMORY'] = 'true'  # Reject when memory pressure
    
    # Performance configuration
    os.environ['VISION_MAX_CONCURRENT'] = '10'  # Safe concurrent limit
    os.environ['VISION_CACHE_SIZE_MB'] = '100'  # Limited cache
    os.environ['VISION_CACHE_ENTRIES'] = '50'  # Max cache entries
    os.environ['VISION_MEMORY_THRESHOLD'] = '60'  # More aggressive threshold
    
    # Import analyzer after setting environment
    from claude_vision_analyzer_main import ClaudeVisionAnalyzer
    
    # Initialize with API key
    api_key = os.getenv('ANTHROPIC_API_KEY', 'your-api-key')
    analyzer = ClaudeVisionAnalyzer(api_key)
    
    print("Claude Vision Analyzer initialized with memory safety")
    print(f"Configuration: {analyzer.config.to_dict()}")
    
    # 2. Check memory health before operations
    health = await analyzer.check_memory_health()
    print(f"\nMemory Health Check:")
    print(f"  Healthy: {health['healthy']}")
    print(f"  Process: {health['process_mb']:.1f} MB")
    print(f"  Available: {health['system_available_gb']:.1f} GB")
    
    if health['recommendations']:
        print("  Recommendations:")
        for rec in health['recommendations']:
            print(f"    - {rec}")
    
    # 3. Get dynamic configuration based on current memory
    safe_config = analyzer.get_memory_safe_config()
    print(f"\nRecommended config for current memory:")
    print(f"  Max concurrent: {safe_config['max_concurrent_requests']}")
    print(f"  Cache entries: {safe_config['cache_max_entries']}")
    print(f"  Max dimension: {safe_config['max_image_dimension']}")
    
    # 4. Example: Analyze image with safety checks
    try:
        # Create test image
        test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # This will automatically check memory before processing
        result = await analyzer.smart_analyze(
            test_image,
            "What's in this image?"
        )
        
        print(f"\nAnalysis successful!")
        print(f"Result: {result.get('description', 'No description')[:100]}...")
        
    except MemoryError as e:
        print(f"\n⚠️ Memory Error: {e}")
        print("The system prevented a potential crash by rejecting the request")
        
        # Get memory stats to understand why
        stats = analyzer.get_all_memory_stats()
        safety = stats['memory_safety']
        print(f"\nMemory Safety Status:")
        print(f"  Safe: {safety['is_safe']}")
        print(f"  Process: {safety['process_mb']:.0f}/{safety['process_limit_mb']} MB")
        print(f"  Warnings: {safety['warnings']}")
    
    # 5. Monitor memory during batch operations
    print("\n\nBatch Processing Example:")
    
    # Small images that should be safe
    small_images = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(5)
    ]
    
    results = []
    rejected = 0
    
    for i, img in enumerate(small_images):
        try:
            result = await analyzer.analyze_screenshot(img, f"Image {i}")
            results.append(result)
            print(f"  ✓ Processed image {i}")
        except MemoryError:
            rejected += 1
            print(f"  ✗ Rejected image {i} (memory pressure)")
    
    print(f"\nBatch complete: {len(results)} processed, {rejected} rejected")
    
    # 6. Final memory report
    final_stats = analyzer.get_all_memory_stats()
    print(f"\nFinal Memory Report:")
    print(f"  Process: {final_stats['system']['process_mb']:.1f} MB")
    print(f"  Available: {final_stats['system']['available_mb']:.1f} MB")
    print(f"  Rejected requests: {final_stats['memory_safety']['rejected_requests']}")
    print(f"  Emergency mode: {final_stats['memory_safety']['emergency_mode']}")
    
    # Cleanup
    await analyzer.cleanup_all_components()
    print("\nCleanup complete")


if __name__ == "__main__":
    print("="*60)
    print("Claude Vision Analyzer - Memory Safe Usage Example")
    print("="*60)
    
    asyncio.run(main())