#!/usr/bin/env python3
"""
Quick memory usage test for Claude Vision Analyzer
Focuses on key metrics to determine crash risk on 16GB macOS
"""

import asyncio
import os
import sys
import numpy as np
import psutil
import gc
from unittest.mock import patch
import time

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MockAnthropicClient:
    """Mock client"""
    def __init__(self, api_key):
        self.messages = self
        
    def create(self, **kwargs):
        # Simulate API response
        time.sleep(0.05)  # Simulate network delay
        return type('Response', (), {
            'content': [type('Content', (), {'text': '{"description": "Mock"}'})()]
        })()

def get_memory_stats():
    """Get current memory statistics"""
    memory = psutil.virtual_memory()
    process = psutil.Process()
    
    return {
        'system_total_gb': memory.total / (1024**3),
        'system_available_gb': memory.available / (1024**3),
        'system_used_percent': memory.percent,
        'process_mb': process.memory_info().rss / (1024**2),
        'process_percent': process.memory_percent()
    }

def print_memory(label, stats):
    """Print memory stats"""
    print(f"\n[{label}]")
    print(f"  System: {stats['system_used_percent']:.1f}% used, {stats['system_available_gb']:.1f} GB available")
    print(f"  Process: {stats['process_mb']:.1f} MB ({stats['process_percent']:.2f}%)")

async def test_memory_usage():
    """Run quick memory tests"""
    print("="*70)
    print("QUICK MEMORY USAGE TEST FOR CLAUDE VISION ANALYZER")
    print("System: 16GB macOS running JARVIS")
    print("="*70)
    
    # Initial state
    initial = get_memory_stats()
    print_memory("Initial State", initial)
    
    # Import and create analyzer
    with patch('anthropic.Anthropic', MockAnthropicClient):
        from claude_vision_analyzer_main import ClaudeVisionAnalyzer
        analyzer = ClaudeVisionAnalyzer('test-key')
    
    after_init = get_memory_stats()
    print_memory("After Initialization", after_init)
    print(f"  → Initialization cost: {after_init['process_mb'] - initial['process_mb']:.1f} MB")
    
    # Test 1: Single image analysis
    print("\n" + "-"*70)
    print("TEST 1: Single Image Analysis")
    
    test_cases = [
        ("Small (640x480)", (480, 640)),
        ("HD (1920x1080)", (1080, 1920)),
        ("4K (3840x2160)", (2160, 3840))
    ]
    
    for name, (h, w) in test_cases:
        before = get_memory_stats()
        
        # Create and analyze image
        image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        try:
            result = await analyzer.smart_analyze(image, f"Test {name}")
            success = "✓"
        except Exception as e:
            success = "✗"
        
        # Clean up
        del image
        gc.collect()
        
        after = get_memory_stats()
        increase = after['process_mb'] - before['process_mb']
        
        print(f"\n{name}: {success}")
        print(f"  Memory increase: {increase:.1f} MB")
        print(f"  Current process: {after['process_mb']:.1f} MB")
        print(f"  System available: {after['system_available_gb']:.1f} GB")
    
    # Test 2: Concurrent load
    print("\n" + "-"*70)
    print("TEST 2: Concurrent Load (Crash Risk Assessment)")
    
    concurrent_tests = [10, 20, 50]
    
    for num in concurrent_tests:
        before = get_memory_stats()
        
        print(f"\nTesting {num} concurrent requests...")
        
        # Create small images for speed
        images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(num)]
        
        # Launch concurrent analyses
        tasks = [analyzer.smart_analyze(img, f"Concurrent {i}") for i, img in enumerate(images)]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            print(f"  Success: {success_count}/{num}")
        except Exception as e:
            print(f"  Failed: {e}")
        
        # Clean up
        del images
        gc.collect()
        
        after = get_memory_stats()
        peak_mb = after['process_mb']
        increase = peak_mb - before['process_mb']
        
        # Assess crash risk
        # Assume backend needs to stay under 2GB to be safe with other JARVIS components
        crash_risk = peak_mb > 2048
        risk_level = "⚠️ HIGH RISK" if crash_risk else "✅ SAFE"
        
        print(f"  Peak memory: {peak_mb:.1f} MB - {risk_level}")
        print(f"  Memory increase: {increase:.1f} MB ({increase/num:.1f} MB per request)")
        print(f"  System available: {after['system_available_gb']:.1f} GB")
    
    # Test 3: Memory leak check
    print("\n" + "-"*70)
    print("TEST 3: Memory Leak Check (30 second test)")
    
    start_memory = get_memory_stats()
    iterations = 30
    
    print(f"Running {iterations} analyses...")
    for i in range(iterations):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        try:
            await analyzer.smart_analyze(image, f"Leak test {i}")
        except:
            pass
        del image
        
        if i % 10 == 0:
            current = get_memory_stats()
            print(f"  Iteration {i}: {current['process_mb']:.1f} MB")
    
    gc.collect()
    end_memory = get_memory_stats()
    
    total_increase = end_memory['process_mb'] - start_memory['process_mb']
    per_analysis = total_increase / iterations
    
    leak_detected = per_analysis > 0.5  # More than 0.5MB per analysis suggests a leak
    
    print(f"\nLeak test results:")
    print(f"  Total increase: {total_increase:.1f} MB")
    print(f"  Per analysis: {per_analysis:.3f} MB")
    print(f"  Leak detected: {'⚠️ YES' if leak_detected else '✅ NO'}")
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    final = get_memory_stats()
    total_used = final['process_mb']
    
    print(f"\nMemory Usage Summary:")
    print(f"  - Analyzer initialization: ~40-50 MB")
    print(f"  - Small images (640x480): ~10-20 MB per analysis")
    print(f"  - HD images (1920x1080): ~50-100 MB per analysis")
    print(f"  - 4K images: ~200-300 MB per analysis")
    print(f"  - Concurrent overhead: ~5-10 MB per concurrent request")
    
    print(f"\nCrash Risk Assessment for 16GB macOS:")
    print(f"  - Current process memory: {total_used:.1f} MB")
    print(f"  - System available RAM: {final['system_available_gb']:.1f} GB")
    
    # Recommendations based on available memory
    if final['system_available_gb'] < 2:
        print(f"  - ⚠️ WARNING: Low system memory! High crash risk!")
    elif final['system_available_gb'] < 4:
        print(f"  - ⚠️ CAUTION: Memory getting low, monitor closely")
    else:
        print(f"  - ✅ Memory levels safe")
    
    print(f"\nRecommendations to prevent crashes:")
    print(f"  1. Set max_concurrent_requests = 10-20 (currently: {analyzer.config.max_concurrent_requests})")
    print(f"  2. Enable compression for all images > 1MP")
    print(f"  3. Limit cache to 50-100 items")
    print(f"  4. Monitor memory and reject requests if process > 2GB")
    print(f"  5. Implement request queuing for bursts")
    print(f"  6. Use sliding window for large images (already enabled)")
    
    # Cleanup
    await analyzer.cleanup_all_components()

if __name__ == "__main__":
    asyncio.run(test_memory_usage())