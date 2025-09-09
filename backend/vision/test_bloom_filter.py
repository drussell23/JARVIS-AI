#!/usr/bin/env python3
"""
Test script for Bloom Filter Network (4.4) in JARVIS Vision System
Demonstrates hierarchical bloom filter capabilities and memory efficiency
"""

import asyncio
import hashlib
import time
import random
import json
from datetime import datetime
import numpy as np
from PIL import Image
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.vision.bloom_filter_network import (
    get_bloom_filter_network, BloomFilterLevel, VisionBloomFilterIntegration
)


async def test_basic_bloom_filter():
    """Test basic bloom filter operations"""
    print("\n=== Testing Basic Bloom Filter Operations ===")
    
    # Get the network
    network = get_bloom_filter_network()
    integration = VisionBloomFilterIntegration(network)
    
    # Test duplicate detection
    print("\n1. Testing duplicate detection:")
    
    # Create test image hashes
    test_images = []
    for i in range(10):
        image_data = f"test_image_{i}"
        image_hash = hashlib.sha256(image_data.encode()).hexdigest()
        test_images.append((image_data, image_hash))
    
    # First pass - all should be new
    print("   First pass (all new):")
    for i, (data, hash_val) in enumerate(test_images):
        is_duplicate = integration.is_image_duplicate(
            hash_val,
            {'index': i, 'timestamp': time.time()}
        )
        print(f"   - Image {i}: {'DUPLICATE' if is_duplicate else 'NEW'}")
    
    # Second pass - all should be duplicates
    print("\n   Second pass (all duplicates):")
    for i, (data, hash_val) in enumerate(test_images):
        is_duplicate = integration.is_image_duplicate(
            hash_val,
            {'index': i, 'timestamp': time.time()}
        )
        print(f"   - Image {i}: {'DUPLICATE' if is_duplicate else 'NEW'}")
    
    # Print stats
    stats = network.get_network_stats()
    print(f"\n   Network stats:")
    print(f"   - Total checks: {stats['network_metrics']['total_checks']}")
    print(f"   - Global hits: {stats['network_metrics']['global_hits']}")
    print(f"   - Hit rate: {stats['efficiency_stats']['hit_rate']:.2%}")
    print(f"   - Hierarchical efficiency: {stats['efficiency_stats']['hierarchical_efficiency']:.2%}")


async def test_quadrant_based_filtering():
    """Test regional bloom filters with quadrants"""
    print("\n=== Testing Quadrant-Based Regional Filtering ===")
    
    network = get_bloom_filter_network()
    integration = VisionBloomFilterIntegration(network)
    
    # Simulate UI elements in different quadrants
    quadrants = [
        {'name': 'Top-Left', 'x': 100, 'y': 100},
        {'name': 'Top-Right', 'x': 1820, 'y': 100},
        {'name': 'Bottom-Left', 'x': 100, 'y': 980},
        {'name': 'Bottom-Right', 'x': 1820, 'y': 980}
    ]
    
    print("\n1. Testing regional filter isolation:")
    
    # Add elements to each quadrant
    for q_idx, quadrant in enumerate(quadrants):
        print(f"\n   Quadrant {q_idx} ({quadrant['name']}):")
        
        # Create 5 elements per quadrant
        for i in range(5):
            element_data = {
                'type': 'button',
                'text': f"Button_{q_idx}_{i}",
                'x': quadrant['x'] + i * 10,
                'y': quadrant['y'] + i * 10
            }
            
            context = {
                'quadrant': q_idx,
                'screen_width': 1920,
                'screen_height': 1080,
                'x': element_data['x'],
                'y': element_data['y']
            }
            
            # First check (should be new)
            is_duplicate = integration.is_window_region_duplicate(
                element_data,
                context
            )
            print(f"     - Element {i}: {'DUPLICATE' if is_duplicate else 'NEW'}")
    
    # Test cross-quadrant checks
    print("\n2. Testing cross-quadrant duplicate detection:")
    
    # Try to add same element to different quadrant (should still be detected)
    test_element = {
        'type': 'button',
        'text': 'Button_0_0',  # Same as first button in quadrant 0
        'x': 1820,  # But in different location
        'y': 980
    }
    
    context = {
        'quadrant': 3,  # Different quadrant
        'screen_width': 1920,
        'screen_height': 1080,
        'x': test_element['x'],
        'y': test_element['y']
    }
    
    is_duplicate = integration.is_window_region_duplicate(test_element, context)
    print(f"   Same element in different quadrant: {'DUPLICATE' if is_duplicate else 'NEW'}")
    
    # Get regional stats
    stats = network.get_network_stats()
    print(f"\n   Regional filter stats:")
    print(f"   - Regional hits: {stats['network_metrics']['regional_hits']}")
    print(f"   - Total memory: {stats['total_memory_mb']}MB")


async def test_hierarchical_promotion():
    """Test element promotion through hierarchy"""
    print("\n=== Testing Hierarchical Promotion ===")
    
    network = get_bloom_filter_network()
    
    # Test element that gets promoted
    test_data = b"frequently_accessed_element"
    
    print("\n1. Initial element check (Element level):")
    is_duplicate, level = network.check_and_add(
        test_data,
        BloomFilterLevel.ELEMENT
    )
    print(f"   - Is duplicate: {is_duplicate}, Level: {level.name}")
    
    # Access same element multiple times
    print("\n2. Accessing element multiple times:")
    for i in range(3):
        is_duplicate, level = network.check_and_add(
            test_data,
            BloomFilterLevel.ELEMENT
        )
        print(f"   - Access {i+1}: Is duplicate: {is_duplicate}, Found at level: {level.name}")
    
    # Now check if it's been promoted to higher levels
    print("\n3. Checking promotion to higher levels:")
    
    # Should now be found at Global level due to hierarchical promotion
    is_duplicate, level = network.check_and_add(
        test_data,
        BloomFilterLevel.GLOBAL
    )
    print(f"   - Checking at Global level: Is duplicate: {is_duplicate}, Level: {level.name}")
    
    stats = network.get_network_stats()
    print(f"\n   Hierarchical shortcuts: {stats['network_metrics']['hierarchical_shortcuts']}")


async def test_memory_efficiency():
    """Test memory usage and efficiency"""
    print("\n=== Testing Memory Efficiency ===")
    
    network = get_bloom_filter_network()
    integration = VisionBloomFilterIntegration(network)
    
    print("\n1. Memory allocation compliance:")
    stats = network.get_network_stats()
    
    print(f"   - Global filter: {stats['global_filter']['allocated_mb']}MB (spec: 4MB)")
    print(f"   - Regional filters total: {sum(rf['allocated_mb'] for rf in stats['regional_filters'])}MB (spec: 4MB)")
    print(f"   - Element filter: {stats['element_filter']['allocated_mb']}MB (spec: 2MB)")
    print(f"   - Total: {stats['total_memory_mb']}MB (spec: 10MB)")
    
    # Test saturation handling
    print("\n2. Testing saturation and automatic reset:")
    
    # Add many elements to trigger saturation
    print("   Adding many elements...")
    for i in range(10000):
        data = f"element_{i}_{random.random()}"
        hash_val = hashlib.sha256(data.encode()).hexdigest()
        integration.is_ui_element_duplicate(
            {'id': data},
            {'index': i}
        )
    
    # Check saturation levels
    stats = network.get_network_stats()
    print(f"\n   Saturation levels after heavy use:")
    print(f"   - Global: {stats['global_filter']['saturation_level']:.2%}")
    for idx, rf in enumerate(stats['regional_filters']):
        print(f"   - Regional {idx}: {rf['saturation_level']:.2%}")
    print(f"   - Element: {stats['element_filter']['saturation_level']:.2%}")
    
    # Optimize network
    print("\n3. Running network optimization...")
    network.optimize_network()
    
    # Check saturation after optimization
    stats = network.get_network_stats()
    print("\n   Saturation levels after optimization:")
    print(f"   - Global: {stats['global_filter']['saturation_level']:.2%}")
    for idx, rf in enumerate(stats['regional_filters']):
        print(f"   - Regional {idx}: {rf['saturation_level']:.2%}")
    print(f"   - Element: {stats['element_filter']['saturation_level']:.2%}")


async def test_false_positive_rates():
    """Test false positive rates at different levels"""
    print("\n=== Testing False Positive Rates ===")
    
    network = get_bloom_filter_network()
    
    # Track false positives
    false_positives = {'global': 0, 'regional': 0, 'element': 0}
    total_checks = 1000
    known_elements = set()
    
    print(f"\n1. Adding {total_checks//2} known elements...")
    for i in range(total_checks // 2):
        data = f"known_element_{i}".encode()
        known_elements.add(data)
        network.check_and_add(data, BloomFilterLevel.GLOBAL)
    
    print(f"\n2. Checking {total_checks} random elements for false positives...")
    for i in range(total_checks):
        # Generate random element that wasn't added
        data = f"random_unknown_{i}_{random.random()}".encode()
        
        if data not in known_elements:
            # Check at each level
            for level in [BloomFilterLevel.GLOBAL, BloomFilterLevel.REGIONAL, BloomFilterLevel.ELEMENT]:
                is_duplicate, found_level = network.check_and_add(data, level)
                if is_duplicate:
                    false_positives[level.name.lower()] += 1
    
    print("\n   False positive rates:")
    for level, count in false_positives.items():
        rate = count / total_checks * 100
        print(f"   - {level.capitalize()}: {rate:.2f}% ({count}/{total_checks})")


async def test_performance_benchmark():
    """Benchmark bloom filter performance"""
    print("\n=== Performance Benchmark ===")
    
    network = get_bloom_filter_network()
    
    # Benchmark insertion speed
    print("\n1. Insertion performance:")
    num_insertions = 10000
    
    start_time = time.time()
    for i in range(num_insertions):
        data = f"benchmark_element_{i}".encode()
        network.check_and_add(data, BloomFilterLevel.GLOBAL)
    
    insert_time = time.time() - start_time
    insert_rate = num_insertions / insert_time
    print(f"   - Inserted {num_insertions} elements in {insert_time:.2f}s")
    print(f"   - Rate: {insert_rate:.0f} insertions/second")
    
    # Benchmark query speed
    print("\n2. Query performance:")
    num_queries = 50000
    
    start_time = time.time()
    hits = 0
    for i in range(num_queries):
        data = f"benchmark_element_{i % num_insertions}".encode()
        is_duplicate, _ = network.check_and_add(data, BloomFilterLevel.GLOBAL)
        if is_duplicate:
            hits += 1
    
    query_time = time.time() - start_time
    query_rate = num_queries / query_time
    print(f"   - Queried {num_queries} elements in {query_time:.2f}s")
    print(f"   - Rate: {query_rate:.0f} queries/second")
    print(f"   - Hit rate: {hits/num_queries:.2%}")


async def test_integration_with_vision():
    """Test integration with vision analyzer"""
    print("\n=== Testing Integration with Vision Analyzer ===")
    
    try:
        from backend.vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        
        # Create analyzer with bloom filter enabled
        analyzer = ClaudeVisionAnalyzer()
        
        # Create test image
        test_image = Image.new('RGB', (800, 600), color='white')
        
        # Analyze same image multiple times
        print("\n1. Analyzing same image multiple times:")
        
        for i in range(3):
            print(f"\n   Analysis {i+1}:")
            result, metrics = await analyzer.analyze_screenshot(
                test_image,
                "What do you see in this image?"
            )
            
            # Check bloom filter stats in result
            if 'bloom_filter_stats' in result:
                stats = result['bloom_filter_stats']
                print(f"   - Total checks: {stats['total_checks']}")
                print(f"   - Hit rate: {stats['hit_rate']:.2%}")
                print(f"   - Memory usage: {stats['memory_usage_mb']}MB")
            else:
                print("   - No bloom filter stats in result")
        
        print("\n✅ Vision analyzer integration successful!")
        
    except Exception as e:
        print(f"\n❌ Vision analyzer integration failed: {e}")
        print("   This is expected if running outside the full JARVIS environment")


async def main():
    """Run all bloom filter tests"""
    print("=" * 60)
    print("JARVIS Vision - Bloom Filter Network Test Suite")
    print("Testing 4MB/1MB×4/2MB hierarchical configuration")
    print("=" * 60)
    
    # Run all tests
    await test_basic_bloom_filter()
    await test_quadrant_based_filtering()
    await test_hierarchical_promotion()
    await test_memory_efficiency()
    await test_false_positive_rates()
    await test_performance_benchmark()
    await test_integration_with_vision()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())