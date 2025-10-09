#!/usr/bin/env python3
"""Test efficient Claude Vision API usage with Quadtree optimization"""

import asyncio
import os
import sys
from PIL import Image, ImageDraw
import numpy as np

# Add backend to path
backend_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_path)

async def test_efficient_vision():
    """Test the efficient vision API usage"""
    print("🔬 Testing Efficient Claude Vision API Usage")
    print("=" * 80)
    
    # Create a test image with distinct regions
    print("\n1. Creating test image with important regions...")
    test_image = create_test_image()
    print(f"   Created test image: {test_image.size}")
    
    # Initialize the vision analyzer
    print("\n2. Initializing Vision Analyzer...")
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY", "test-key")
    
    # Create analyzer instance
    analyzer = ClaudeVisionAnalyzer(api_key)
    print("   ✅ Vision analyzer initialized")
    
    # Test 1: Normal analysis (should use quadtree)
    print("\n3. Testing normal analysis with Quadtree optimization:")
    try:
        result, metrics = await analyzer.analyze_screenshot(
            test_image,
            "Describe what you see in this screen, focusing on the important areas",
            priority="normal"
        )
        
        print(f"\n   📊 Analysis Metrics:")
        print(f"      Processing strategy: {metrics.processing_strategy}")
        print(f"      Regions extracted: {metrics.regions_extracted}")
        print(f"      Coverage ratio: {metrics.coverage_ratio:.1%}")
        print(f"      Original size: {metrics.image_size_original:,} bytes")
        print(f"      Compressed size: {metrics.image_size_compressed:,} bytes")
        print(f"      Compression ratio: {metrics.compression_ratio:.1%}")
        print(f"      Total time: {metrics.total_time:.2f}s")
        
        if metrics.processing_strategy == "region_composite":
            print(f"\n   🎯 Optimization Applied!")
            print(f"      Sent only {metrics.regions_extracted} regions instead of full image")
            print(f"      API payload reduced by {metrics.compression_ratio:.1%}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Test with different system modes
    print("\n\n4. Testing with different system modes:")
    
    # Simulate pressure mode
    print("\n   Testing in PRESSURE mode:")
    try:
        # Modify config temporarily
        custom_config = {'memory_threshold_percent': 30}  # Simulate memory pressure
        
        result, metrics = await analyzer.analyze_screenshot(
            test_image,
            "What's on screen?",
            priority="normal",
            custom_config=custom_config
        )
        
        print(f"      Strategy: {metrics.processing_strategy}")
        print(f"      Regions: {metrics.regions_extracted}")
        print(f"      System mode: {metrics.system_mode}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Compare with and without quadtree
    print("\n\n5. Comparing efficiency:")
    
    # Without quadtree (disable it)
    print("\n   Without Quadtree optimization:")
    os.environ['QUADTREE_SPATIAL_ENABLED'] = 'false'
    try:
        result1, metrics1 = await analyzer.analyze_screenshot(
            test_image,
            "Describe the screen",
            priority="normal"
        )
        print(f"      Strategy: {metrics1.processing_strategy}")
        print(f"      Compressed size: {metrics1.image_size_compressed:,} bytes")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # With quadtree (re-enable)
    print("\n   With Quadtree optimization:")
    os.environ['QUADTREE_SPATIAL_ENABLED'] = 'true'
    try:
        result2, metrics2 = await analyzer.analyze_screenshot(
            test_image,
            "Describe the screen",
            priority="normal"
        )
        print(f"      Strategy: {metrics2.processing_strategy}")
        print(f"      Compressed size: {metrics2.image_size_compressed:,} bytes")
        
        if metrics1.image_size_compressed > 0 and metrics2.image_size_compressed > 0:
            savings = 1 - (metrics2.image_size_compressed / metrics1.image_size_compressed)
            print(f"\n   💰 API Payload Savings: {savings:.1%}")
            print(f"      Reduced from {metrics1.image_size_compressed:,} to {metrics2.image_size_compressed:,} bytes")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Test completed!")

def create_test_image():
    """Create a test image with distinct important regions"""
    # Create a 1920x1080 image
    width, height = 1920, 1080
    image = Image.new('RGB', (width, height), color=(50, 50, 50))
    draw = ImageDraw.Draw(image)
    
    # Add some important regions
    # Region 1: Top-left - Error dialog
    draw.rectangle([50, 50, 350, 200], fill=(255, 100, 100), outline=(255, 0, 0), width=3)
    draw.text((60, 60), "ERROR: Critical Issue!", fill=(255, 255, 255))
    
    # Region 2: Center - Main content
    draw.rectangle([600, 300, 1300, 700], fill=(100, 150, 255), outline=(0, 0, 255), width=3)
    draw.text((610, 310), "Main Application Window", fill=(255, 255, 255))
    
    # Region 3: Bottom-right - Notification
    draw.rectangle([1500, 900, 1850, 1030], fill=(100, 255, 100), outline=(0, 255, 0), width=3)
    draw.text((1510, 910), "New Notification", fill=(0, 0, 0))
    
    # Add some less important background elements
    for i in range(20):
        x = np.random.randint(0, width-100)
        y = np.random.randint(0, height-50)
        draw.rectangle([x, y, x+80, y+40], fill=(80, 80, 80), outline=(100, 100, 100))
    
    return image

if __name__ == "__main__":
    asyncio.run(test_efficient_vision())