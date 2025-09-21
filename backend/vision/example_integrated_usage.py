#!/usr/bin/env python3
"""
Example usage of the integrated Claude Vision Analyzer
Shows practical examples of using all 4 memory-optimized components together
"""

import asyncio
import os
import numpy as np
from PIL import Image

async def example_basic_usage():
    """Basic usage example"""
    from claude_vision_analyzer_main import ClaudeVisionAnalyzer
    
    # Initialize analyzer
    api_key = os.getenv('ANTHROPIC_API_KEY', 'your-api-key')
    analyzer = ClaudeVisionAnalyzer(api_key)
    
    # Example 1: Simple screenshot analysis
    screenshot = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    result = await analyzer.smart_analyze(screenshot, "What's on the screen?")
    print(f"Analysis: {result['summary']}")
    
    # Example 2: Start continuous monitoring
    await analyzer.start_continuous_monitoring({
        'app_changed': lambda data: print(f"App changed: {data}"),
        'memory_warning': lambda data: print(f"Memory warning: {data}")
    })
    
    # Example 3: Comprehensive workspace analysis
    workspace = await analyzer.analyze_workspace_comprehensive(screenshot)
    print(f"Workspace analysis used: {workspace['components_used']}")
    
    # Example 4: Get memory stats
    stats = analyzer.get_all_memory_stats()
    print(f"Available memory: {stats['system']['available_mb']:.0f} MB")
    
    # Cleanup
    await analyzer.cleanup_all_components()

async def example_memory_aware_processing():
    """Example showing memory-aware processing"""
    from claude_vision_analyzer_main import ClaudeVisionAnalyzer
    
    analyzer = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY', 'your-api-key'))
    
    # Configure for low memory usage
    analyzer.update_config(
        max_image_dimension=1024,  # Smaller images
        jpeg_quality=70,           # Lower quality
        cache_size_mb=50          # Smaller cache
    )
    
    # Large screenshot that will trigger sliding window
    large_screenshot = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
    
    # Smart analyze will automatically use sliding window for large images
    result = await analyzer.smart_analyze(
        large_screenshot,
        "Find all error messages and warnings on screen"
    )
    
    print(f"Method used: {result['metadata']['analysis_method']}")
    if result['metadata']['analysis_method'] == 'sliding_window':
        print(f"Windows analyzed: {result['metadata']['windows_analyzed']}")

async def example_window_relationship_analysis():
    """Example of analyzing window relationships"""
    from claude_vision_analyzer_main import ClaudeVisionAnalyzer
    
    analyzer = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY', 'your-api-key'))
    
    # Get window analyzer
    window_analyzer = await analyzer.get_window_analyzer()
    if window_analyzer:
        # Analyze current workspace
        workspace = await window_analyzer.analyze_workspace()
        
        # Get relationship detector
        detector = await analyzer.get_relationship_detector()
        if detector and 'windows' in workspace:
            # Detect relationships
            relationships = detector.detect_relationships(workspace['windows'])
            
            # Group related windows
            groups = detector.group_windows(workspace['windows'], relationships)
            
            for group in groups:
                print(f"Found {group.group_type} group with {len(group.windows)} windows")
                print(f"  Common elements: {', '.join(group.common_elements)}")

async def example_continuous_monitoring():
    """Example of continuous screen monitoring"""
    from claude_vision_analyzer_main import ClaudeVisionAnalyzer
    
    analyzer = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY', 'your-api-key'))
    
    # Define event handlers
    async def on_weather_visible(data):
        print(f"Weather detected: {data['weather_info']}")
    
    async def on_error_detected(data):
        print(f"Error on screen: {data['error_context']}")
    
    async def on_memory_warning(data):
        level = data['level']
        if level == 'critical':
            print("⚠️ Critical memory - stopping non-essential monitoring")
            await analyzer.stop_continuous_monitoring()
    
    # Start monitoring with callbacks
    await analyzer.start_continuous_monitoring({
        'weather_visible': on_weather_visible,
        'error_detected': on_error_detected,
        'memory_warning': on_memory_warning
    })
    
    # Monitor for 30 seconds
    await asyncio.sleep(30)
    
    # Get current context
    context = await analyzer.get_current_screen_context()
    print(f"Current app: {context.get('current_app')}")
    
    # Stop monitoring
    await analyzer.stop_continuous_monitoring()

async def example_swift_acceleration():
    """Example using Swift/Metal acceleration"""
    from claude_vision_analyzer_main import ClaudeVisionAnalyzer
    
    analyzer = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY', 'your-api-key'))
    
    # Get Swift vision integration
    swift_vision = await analyzer.get_swift_vision()
    if swift_vision and swift_vision.enabled:
        print("✅ Swift/Metal acceleration available")
        
        # Process image with Swift
        test_image = Image.new('RGB', (1920, 1080), color='white')
        result = await swift_vision.process_screenshot(test_image)
        
        print(f"Processing method: {result.method}")
        print(f"Processing time: {result.processing_time:.3f}s")
        print(f"Memory pressure: {result.memory_pressure_level}")
        
        # Extract text regions
        text_regions = await swift_vision.extract_text_regions(test_image, max_regions=5)
        print(f"Found {len(text_regions)} text regions")
    else:
        print("❌ Swift acceleration not available, using Python fallback")

async def example_configuration():
    """Example of runtime configuration"""
    from claude_vision_analyzer_main import ClaudeVisionAnalyzer
    
    # Set environment variables before initialization
    os.environ['VISION_MAX_IMAGE_DIM'] = '1024'
    os.environ['VISION_CACHE_SIZE_MB'] = '50'
    os.environ['VISION_MONITOR_INTERVAL'] = '5.0'
    os.environ['WINDOW_REL_MIN_CONFIDENCE'] = '0.7'
    os.environ['SWIFT_VISION_MAX_MEMORY_MB'] = '200'
    
    analyzer = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY', 'your-api-key'))
    
    print("Configuration loaded from environment:")
    print(f"  Max image dimension: {analyzer.config.max_image_dimension}")
    print(f"  Cache size: {analyzer.config.cache_size_mb} MB")
    
    # Get component configs
    continuous = await analyzer.get_continuous_analyzer()
    if continuous:
        print(f"  Monitor interval: {continuous.config['update_interval']}s")
    
    detector = await analyzer.get_relationship_detector()
    if detector:
        print(f"  Min confidence: {detector.config['min_confidence']}")

if __name__ == "__main__":
    print("Integrated Vision System Examples")
    print("=" * 50)
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Memory-Aware Processing", example_memory_aware_processing),
        ("Window Relationships", example_window_relationship_analysis),
        ("Continuous Monitoring", example_continuous_monitoring),
        ("Swift Acceleration", example_swift_acceleration),
        ("Configuration", example_configuration)
    ]
    
    async def run_all():
        for name, func in examples:
            print(f"\n\n{'='*50}")
            print(f"Example: {name}")
            print(f"{'='*50}")
            try:
                await func()
            except Exception as e:
                print(f"Error in {name}: {e}")
    
    asyncio.run(run_all())