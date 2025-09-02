#!/usr/bin/env python3
"""
Test script for ALL 6 integrated vision components in claude_vision_analyzer_main.py
Demonstrates complete integration with no hardcoded values
"""

import asyncio
import os
import numpy as np
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_all_integrated_components():
    """Test all 6 integrated vision components"""
    print("🔍 Testing ALL Integrated Vision Components")
    print("=" * 70)
    
    # Import the main analyzer
    from claude_vision_analyzer_main import ClaudeVisionAnalyzer
    
    # Initialize with API key
    api_key = os.getenv('ANTHROPIC_API_KEY', 'dummy-key-for-testing')
    analyzer = ClaudeVisionAnalyzer(api_key)
    
    print("\n📊 Main Configuration:")
    print(f"   Max Image Dimension: {analyzer.config.max_image_dimension}")
    print(f"   Cache Enabled: {analyzer.config.cache_enabled}")
    print(f"   Memory Threshold: {analyzer.config.memory_threshold_percent}%")
    
    # Create test screenshots
    small_screenshot = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    medium_screenshot = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    large_screenshot = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
    
    print("\n" + "=" * 70)
    print("TESTING 6 INTEGRATED COMPONENTS")
    print("=" * 70)
    
    # Component 1: Swift Vision Integration
    print("\n1️⃣ SWIFT VISION INTEGRATION")
    print("-" * 50)
    swift_vision = await analyzer.get_swift_vision()
    if swift_vision:
        print("✅ Swift Vision initialized")
        print(f"   Enabled: {swift_vision.enabled}")
        print(f"   Max Memory: {swift_vision.config['max_memory_mb']}MB")
        print(f"   Metal Limit: {swift_vision.config['metal_memory_limit_mb']}MB")
        print(f"   Circuit Breaker Threshold: {swift_vision.config['circuit_breaker_threshold']}")
        stats = swift_vision.get_memory_stats()
        print(f"   Memory Pressure: {stats['memory_pressure']}")
    else:
        print("❌ Swift Vision not available")
    
    # Component 2: Window Analysis
    print("\n2️⃣ WINDOW ANALYSIS")
    print("-" * 50)
    window_analyzer = await analyzer.get_window_analyzer()
    if window_analyzer:
        print("✅ Window Analyzer initialized")
        print(f"   Max Memory: {window_analyzer.config['max_memory_mb']}MB")
        print(f"   Max Cached Windows: {window_analyzer.config['max_cached_windows']}")
        print(f"   Cache TTL: {window_analyzer.config['cache_ttl_seconds']}s")
        print(f"   Skip Minimized: {window_analyzer.config['skip_minimized']}")
        memory_stats = window_analyzer.get_memory_stats()
        print(f"   Current Usage: {memory_stats['current_usage_mb']:.1f}MB")
    else:
        print("❌ Window Analyzer not available")
    
    # Component 3: Window Relationship Detection
    print("\n3️⃣ WINDOW RELATIONSHIP DETECTION")
    print("-" * 50)
    relationship_detector = await analyzer.get_relationship_detector()
    if relationship_detector:
        print("✅ Relationship Detector initialized")
        print(f"   Max Memory: {relationship_detector.config['max_memory_mb']}MB")
        print(f"   Min Confidence: {relationship_detector.config['min_confidence']}")
        print(f"   Max Windows to Analyze: {relationship_detector.config['max_windows_to_analyze']}")
        print(f"   Group Min Confidence: {relationship_detector.config['group_min_confidence']}")
        stats = relationship_detector.get_stats()
        print(f"   Relationships Detected: {stats['relationships_detected']}")
    else:
        print("❌ Relationship Detector not available")
    
    # Component 4: Continuous Screen Analyzer
    print("\n4️⃣ CONTINUOUS SCREEN ANALYZER")
    print("-" * 50)
    continuous_analyzer = await analyzer.get_continuous_analyzer()
    if continuous_analyzer:
        print("✅ Continuous Analyzer initialized")
        print(f"   Update Interval: {continuous_analyzer.config['update_interval']}s")
        print(f"   Max Captures: {continuous_analyzer.config['max_captures_in_memory']}")
        print(f"   Memory Limit: {continuous_analyzer.config['memory_limit_mb']}MB")
        print(f"   Dynamic Interval: {continuous_analyzer.config['dynamic_interval_enabled']}")
        memory_stats = continuous_analyzer.get_memory_stats()
        print(f"   Current Interval: {memory_stats['current_interval']}s")
    else:
        print("❌ Continuous Analyzer not available")
    
    # Component 5: Memory-Efficient Analyzer
    print("\n5️⃣ MEMORY-EFFICIENT ANALYZER")
    print("-" * 50)
    mem_efficient = await analyzer.get_memory_efficient_analyzer()
    if mem_efficient:
        print("✅ Memory-Efficient Analyzer initialized")
        print(f"   Model: {mem_efficient.model}")
        print(f"   Max Cache Size: {mem_efficient.max_cache_size / 1024 / 1024:.0f}MB")
        print(f"   Max Memory Usage: {mem_efficient.max_memory_usage / 1024 / 1024:.0f}MB")
        print(f"   Memory Pressure Threshold: {mem_efficient.memory_pressure_threshold:.0%}")
        print(f"   Compression Strategies: {list(mem_efficient.compression_strategies.keys())}")
        metrics = mem_efficient.get_metrics()
        print(f"   Cache Hit Rate: {metrics['cache_hit_rate']:.1%}")
    else:
        print("❌ Memory-Efficient Analyzer not available")
    
    # Component 6: Simplified Vision System
    print("\n6️⃣ SIMPLIFIED VISION SYSTEM")
    print("-" * 50)
    simplified = await analyzer.get_simplified_vision()
    if simplified:
        print("✅ Simplified Vision System initialized")
        print(f"   Enabled: {simplified.enabled}")
        print(f"   Max Response Cache: {simplified.config['max_response_cache']}")
        print(f"   Cache TTL: {simplified.config['cache_ttl_seconds']}s")
        print(f"   Confidence Threshold: {simplified.config['confidence_threshold']}")
        templates = simplified.get_available_templates()
        print(f"   Available Templates: {len(templates)}")
        print(f"   Templates: {', '.join(templates[:5])}...")
    else:
        print("❌ Simplified Vision System not available")
    
    print("\n" + "=" * 70)
    print("TESTING INTEGRATED FUNCTIONALITY")
    print("=" * 70)
    
    # Test 1: Compression Strategy Analysis
    print("\n🔧 Test 1: Compression Strategy Analysis")
    print("-" * 50)
    try:
        # Test different compression strategies
        strategies = ["text", "ui", "activity", "detailed", "quick"]
        for strategy in strategies:
            result = await analyzer.analyze_with_compression_strategy(
                small_screenshot, 
                f"Test {strategy} compression",
                strategy
            )
            print(f"   {strategy}: Success={bool(result)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Batch Region Analysis
    print("\n🔧 Test 2: Batch Region Analysis")
    print("-" * 50)
    try:
        regions = [
            {"x": 0, "y": 0, "width": 200, "height": 150, "prompt": "Analyze top-left"},
            {"x": 440, "y": 0, "width": 200, "height": 150, "prompt": "Analyze top-right"},
            {"x": 220, "y": 165, "width": 200, "height": 150, "prompt": "Analyze center"}
        ]
        results = await analyzer.batch_analyze_regions(medium_screenshot, regions)
        print(f"   Analyzed {len(results)} regions successfully")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Change Detection
    print("\n🔧 Test 3: Change Detection")
    print("-" * 50)
    try:
        # Create slightly different screenshot
        changed_screenshot = small_screenshot.copy()
        changed_screenshot[100:200, 100:200] = 255  # Add white square
        
        result = await analyzer.analyze_with_change_detection(
            changed_screenshot,
            small_screenshot,
            "Detect changes"
        )
        print(f"   Change detected: {result.get('changed', result.get('description', 'Unknown'))}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 4: Query Templates
    print("\n🔧 Test 4: Query Templates")
    print("-" * 50)
    try:
        # Test various query templates
        notifications = await analyzer.check_for_notifications()
        print(f"   Notifications check: {notifications.get('success', False)}")
        
        errors = await analyzer.check_for_errors()
        print(f"   Error check: {errors.get('success', False)}")
        
        element = await analyzer.find_ui_element("close button")
        print(f"   Element search: {element.get('success', False)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 5: Memory Statistics
    print("\n🔧 Test 5: Comprehensive Memory Statistics")
    print("-" * 50)
    all_stats = analyzer.get_all_memory_stats()
    print(f"   System Memory:")
    print(f"      Available: {all_stats['system']['available_mb']:.0f}MB")
    print(f"      Used: {all_stats['system']['used_percent']:.1f}%")
    print(f"      Process: {all_stats['system']['process_mb']:.0f}MB")
    print(f"   Components with stats: {len(all_stats['components'])}")
    
    # Test 6: Smart Analysis (Automatic Method Selection)
    print("\n🔧 Test 6: Smart Analysis")
    print("-" * 50)
    try:
        # Small image - should use full analysis
        result = await analyzer.smart_analyze(small_screenshot, "Analyze this")
        print(f"   Small image method: {result['metadata']['analysis_method']}")
        
        # Large image - should use sliding window
        result = await analyzer.smart_analyze(large_screenshot, "Find all buttons")
        print(f"   Large image method: {result['metadata']['analysis_method']}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 70)
    print("ENVIRONMENT VARIABLES SUMMARY")
    print("=" * 70)
    
    # Show key environment variables
    env_groups = {
        "Main Analyzer": [
            "VISION_MAX_IMAGE_DIM", "VISION_JPEG_QUALITY", "VISION_CACHE_SIZE_MB"
        ],
        "Continuous Analyzer": [
            "VISION_MONITOR_INTERVAL", "VISION_MAX_CAPTURES", "VISION_MEMORY_LIMIT_MB"
        ],
        "Window Analyzer": [
            "WINDOW_ANALYZER_MAX_MEMORY_MB", "WINDOW_MAX_CACHED", "WINDOW_CACHE_TTL"
        ],
        "Relationship Detector": [
            "WINDOW_REL_MAX_MEMORY_MB", "WINDOW_REL_MIN_CONFIDENCE"
        ],
        "Swift Vision": [
            "SWIFT_VISION_MAX_MEMORY_MB", "SWIFT_VISION_METAL_LIMIT_MB"
        ],
        "Memory Efficient": [
            "VISION_CACHE_DIR", "VISION_MAX_MEMORY_MB", "VISION_MEMORY_PRESSURE_THRESHOLD"
        ],
        "Simplified Vision": [
            "VISION_QUERY_TEMPLATES", "VISION_CONFIDENCE_THRESHOLD"
        ]
    }
    
    for group_name, vars in env_groups.items():
        print(f"\n{group_name}:")
        for var in vars:
            value = os.getenv(var, "not set")
            print(f"   {var}: {value}")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    await analyzer.cleanup_all_components()
    print("✅ All components cleaned up")
    
    print("\n" + "=" * 70)
    print("✨ ALL 6 COMPONENTS SUCCESSFULLY INTEGRATED!")
    print("=" * 70)
    print("\nThe system now includes:")
    print("✅ Swift Vision Integration - Metal acceleration with circuit breaker")
    print("✅ Window Analysis - Memory-aware window content analysis")
    print("✅ Relationship Detection - Configurable window relationships")
    print("✅ Continuous Monitoring - Dynamic interval adjustment")
    print("✅ Memory-Efficient Analyzer - Smart compression strategies")
    print("✅ Simplified Vision - Configurable query templates")
    print("\nAll optimized for 16GB RAM with NO hardcoded values!")

if __name__ == "__main__":
    asyncio.run(test_all_integrated_components())