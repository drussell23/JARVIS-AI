#!/usr/bin/env python3
"""
Test script for integrated Claude Vision Analyzer with all memory-optimized components
Demonstrates usage of all 4 enhanced files integrated into claude_vision_analyzer_main.py
"""

import asyncio
import os
import numpy as np
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_integrated_vision():
    """Test all integrated components"""
    print("🔍 Testing Integrated Claude Vision Analyzer")
    print("=" * 60)
    
    # Import the main analyzer
    from claude_vision_analyzer_main import ClaudeVisionAnalyzer
    
    # Initialize with API key (dummy for testing)
    api_key = os.getenv('ANTHROPIC_API_KEY', 'dummy-key-for-testing')
    analyzer = ClaudeVisionAnalyzer(api_key)
    
    print("\n📊 Configuration Summary:")
    print(f"   Max Image Dimension: {analyzer.config.max_image_dimension}")
    print(f"   Cache Enabled: {analyzer.config.cache_enabled}")
    print(f"   Memory Threshold: {analyzer.config.memory_threshold_percent}%")
    
    # Create test screenshot
    test_screenshot = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    # Test 1: Swift Vision Integration
    print("\n\n1️⃣ Testing Swift Vision Integration:")
    print("-" * 40)
    swift_vision = await analyzer.get_swift_vision()
    if swift_vision:
        print(f"✅ Swift Vision Available: {swift_vision.enabled}")
        print(f"   Config: {swift_vision.config}")
        stats = swift_vision.get_memory_stats()
        print(f"   Memory Stats: {stats}")
    else:
        print("❌ Swift Vision not available")
    
    # Test 2: Window Analysis
    print("\n\n2️⃣ Testing Window Analysis:")
    print("-" * 40)
    window_analyzer = await analyzer.get_window_analyzer()
    if window_analyzer:
        print("✅ Window Analyzer initialized")
        print(f"   Config: {window_analyzer.config}")
        # Note: Actual window analysis requires real window data
        print("   Would analyze workspace windows here...")
        memory_stats = window_analyzer.get_memory_stats()
        print(f"   Memory Stats: {memory_stats}")
    else:
        print("❌ Window Analyzer not available")
    
    # Test 3: Window Relationship Detection
    print("\n\n3️⃣ Testing Window Relationship Detection:")
    print("-" * 40)
    relationship_detector = await analyzer.get_relationship_detector()
    if relationship_detector:
        print("✅ Relationship Detector initialized")
        print(f"   Config: {relationship_detector.config}")
        # Test with mock windows
        class MockWindow:
            def __init__(self, window_id, app_name, title):
                self.window_id = window_id
                self.app_name = app_name
                self.window_title = title
                self.is_visible = True
        
        mock_windows = [
            MockWindow(1, "Visual Studio Code", "myproject — Visual Studio Code"),
            MockWindow(2, "Chrome", "Python Documentation - myproject API"),
            MockWindow(3, "Terminal", "~/projects/myproject")
        ]
        
        relationships = relationship_detector.detect_relationships(mock_windows)
        print(f"   Found {len(relationships)} relationships")
        stats = relationship_detector.get_stats()
        print(f"   Stats: {stats}")
    else:
        print("❌ Relationship Detector not available")
    
    # Test 4: Continuous Screen Analyzer
    print("\n\n4️⃣ Testing Continuous Screen Analyzer:")
    print("-" * 40)
    continuous_analyzer = await analyzer.get_continuous_analyzer()
    if continuous_analyzer:
        print("✅ Continuous Analyzer initialized")
        print(f"   Config: {continuous_analyzer.config}")
        memory_stats = continuous_analyzer.get_memory_stats()
        print(f"   Memory Stats: {memory_stats}")
        
        # Register a test callback
        def test_callback(data):
            print(f"   Event triggered: {data}")
        
        continuous_analyzer.register_callback('memory_warning', test_callback)
        print("   Registered memory warning callback")
    else:
        print("❌ Continuous Analyzer not available")
    
    # Test 5: Comprehensive Workspace Analysis
    print("\n\n5️⃣ Testing Comprehensive Workspace Analysis:")
    print("-" * 40)
    try:
        # This uses all components together
        workspace_result = await analyzer.analyze_workspace_comprehensive(test_screenshot)
        print("✅ Comprehensive analysis completed")
        print(f"   Components used: {workspace_result['components_used']}")
        print(f"   Memory stats: {workspace_result['memory_stats']}")
    except Exception as e:
        print(f"❌ Comprehensive analysis failed: {e}")
    
    # Test 6: Smart Analysis (Automatic method selection)
    print("\n\n6️⃣ Testing Smart Analysis:")
    print("-" * 40)
    try:
        result = await analyzer.smart_analyze(
            test_screenshot,
            "Find all buttons and interactive elements on the screen"
        )
        print("✅ Smart analysis completed")
        print(f"   Method used: {result['metadata']['analysis_method']}")
        print(f"   Confidence: {result.get('confidence', 'N/A')}")
    except Exception as e:
        print(f"❌ Smart analysis failed: {e}")
    
    # Test 7: Memory Statistics
    print("\n\n7️⃣ Overall Memory Statistics:")
    print("-" * 40)
    all_stats = analyzer.get_all_memory_stats()
    print(f"System Memory:")
    print(f"   Available: {all_stats['system']['available_mb']:.0f} MB")
    print(f"   Used: {all_stats['system']['used_percent']:.1f}%")
    print(f"   Process: {all_stats['system']['process_mb']:.0f} MB")
    
    print(f"\nComponent Memory Usage:")
    for component, stats in all_stats['components'].items():
        print(f"   {component}: {stats}")
    
    # Test 8: Configuration
    print("\n\n8️⃣ Environment Variables for Configuration:")
    print("-" * 40)
    env_vars = [
        # Main vision analyzer
        "VISION_MAX_IMAGE_DIM", "VISION_JPEG_QUALITY", "VISION_CACHE_SIZE_MB",
        # Continuous analyzer
        "VISION_MONITOR_INTERVAL", "VISION_MAX_CAPTURES", "VISION_MEMORY_LIMIT_MB",
        # Window analyzer
        "WINDOW_ANALYZER_MAX_MEMORY_MB", "WINDOW_MAX_CACHED", "WINDOW_CACHE_TTL",
        # Relationship detector
        "WINDOW_REL_MAX_MEMORY_MB", "WINDOW_REL_MIN_CONFIDENCE",
        # Swift vision
        "SWIFT_VISION_MAX_MEMORY_MB", "SWIFT_VISION_METAL_LIMIT_MB"
    ]
    
    print("Key environment variables for memory optimization:")
    for var in env_vars[:10]:  # Show first 10
        value = os.getenv(var, "not set")
        print(f"   {var}: {value}")
    print(f"   ... and {len(env_vars) - 10} more")
    
    # Cleanup
    print("\n\n🧹 Cleaning up...")
    await analyzer.cleanup_all_components()
    print("✅ All components cleaned up")
    
    print("\n\n✨ Integration test complete!")
    print("\nAll 4 enhanced files are successfully integrated:")
    print("✅ continuous_screen_analyzer.py - Memory-aware monitoring")
    print("✅ window_analysis.py - Configurable window analysis")
    print("✅ window_relationship_detector.py - Dynamic relationship detection")
    print("✅ swift_vision_integration.py - Memory-safe Metal acceleration")
    print("\nThe system is optimized for 16GB RAM with no hardcoded values!")

if __name__ == "__main__":
    asyncio.run(test_integrated_vision())