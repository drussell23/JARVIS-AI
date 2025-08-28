#!/usr/bin/env python3
"""
Test script to measure vision response performance improvements
Compares original vs optimized implementations
"""

import asyncio
import time
import os
from PIL import Image
import numpy as np

# Set up environment
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', '')

async def test_performance():
    """Test and compare vision response times"""
    
    print("🔍 Testing JARVIS Vision Response Performance\n")
    print("=" * 60)
    
    # Test query
    test_query = "Hey JARVIS, can you see my screen?"
    
    # Create a test screenshot (or capture real one)
    try:
        from vision.async_screen_capture import capture_screen_optimized
        print("📸 Capturing screen...")
        screenshot = await capture_screen_optimized()
        if not screenshot:
            # Create dummy screenshot for testing
            screenshot = Image.new('RGB', (1920, 1080), color=(50, 50, 50))
            print("⚠️  Using dummy screenshot (no screen capture permission)")
    except:
        screenshot = Image.new('RGB', (1920, 1080), color=(50, 50, 50))
        print("⚠️  Using dummy screenshot")
    
    print(f"✓ Screenshot ready: {screenshot.size}\n")
    
    # Test 1: Original Implementation (if available)
    print("1️⃣  Testing Original Implementation...")
    try:
        from vision.natural_responses import DynamicResponseGenerator
        
        generator = DynamicResponseGenerator()
        if not generator.client:
            print("❌ No API key configured - skipping original test\n")
            original_time = None
        else:
            start = time.time()
            
            # Disable optimizer for original test
            if hasattr(generator, 'optimizer'):
                generator.optimizer = None
            
            result = await generator.analyze_screen_with_full_context(
                screenshot=screenshot,
                user_query=test_query,
                analysis_depth="comprehensive"
            )
            
            original_time = time.time() - start
            print(f"✓ Original time: {original_time:.2f} seconds")
            print(f"  Response preview: {str(result.raw_description)[:100]}...")
    except Exception as e:
        print(f"❌ Original test failed: {e}")
        original_time = None
    
    print()
    
    # Test 2: Optimized Implementation
    print("2️⃣  Testing Optimized Implementation...")
    try:
        from vision.natural_responses import analyze_and_respond
        
        start = time.time()
        
        result = await analyze_and_respond(
            screenshot=screenshot,
            user_query=test_query,
            response_type="confirmation",
            analysis_depth="fast"
        )
        
        optimized_time = time.time() - start
        print(f"✓ Optimized time: {optimized_time:.2f} seconds")
        
        if 'response' in result:
            print(f"  Response: {result['response'][:100]}...")
        if 'performance_ms' in result:
            print(f"  Reported performance: {result['performance_ms']}ms")
        if 'cached' in result:
            print(f"  Cache hit: {result.get('cached', False)}")
        if 'complexity' in result:
            print(f"  Request complexity: {result['complexity']}")
            
    except Exception as e:
        print(f"❌ Optimized test failed: {e}")
        optimized_time = None
    
    print()
    
    # Test 3: NEW Fast Vision Confirmation
    print("3️⃣  Testing NEW Fast Vision Confirmation...")
    try:
        from vision.natural_responses import confirm_vision_capability
        
        # Test multiple times to check consistency
        times = []
        for i in range(5):
            start = time.time()
            result = await confirm_vision_capability()
            elapsed = time.time() - start
            times.append(elapsed)
            
            if i == 0:  # Show first result details
                print(f"✓ First run time: {elapsed:.3f} seconds ({elapsed*1000:.0f}ms)")
                print(f"  Response: {result['response']}")
                print(f"  Screen info: {result.get('screen_info', 'N/A')}")
                print(f"  Cached: {result.get('cached', False)}")
        
        avg_time = sum(times) / len(times)
        print(f"\n  Average time (5 runs): {avg_time:.3f}s ({avg_time*1000:.0f}ms)")
        print(f"  Min time: {min(times):.3f}s ({min(times)*1000:.0f}ms)")
        print(f"  Max time: {max(times):.3f}s ({max(times)*1000:.0f}ms)")
        
        if avg_time < 0.1:  # Less than 100ms
            print("  ✅ EXCELLENT: Sub-100ms response achieved!")
        elif avg_time < 1.0:
            print("  ✅ GOOD: Sub-second response achieved!")
        else:
            print("  ⚠️  Response time still above 1 second")
            
    except Exception as e:
        print(f"❌ New function test failed: {e}")
    
    print()
    
    # Test 4: Performance Breakdown
    print("4️⃣  Performance Breakdown...")
    try:
        from vision.performance_optimizer import get_performance_optimizer
        
        optimizer = get_performance_optimizer()
        stats = optimizer.get_performance_stats()
        
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
        print(f"  Circuit breaker: {'OPEN' if stats['circuit_breaker_open'] else 'CLOSED'}")
        print(f"  Fast model: {stats['config']['fast_model']}")
        print(f"  Caching: {'Enabled' if stats['config']['caching_enabled'] else 'Disabled'}")
        print(f"  Parallel: {'Enabled' if stats['config']['parallel_enabled'] else 'Disabled'}")
        
    except Exception as e:
        print(f"  Stats unavailable: {e}")
    
    print("\n" + "=" * 60)
    print("📊 Performance Summary:\n")
    
    if original_time and optimized_time:
        improvement = (original_time - optimized_time) / original_time * 100
        speedup = original_time / optimized_time
        
        print(f"  Original:  {original_time:.2f}s")
        print(f"  Optimized: {optimized_time:.2f}s")
        print(f"  Improvement: {improvement:.1f}%")
        print(f"  Speedup: {speedup:.1f}x faster")
        
        if optimized_time < 2.0:
            print(f"\n✅ Target met: Response under 2 seconds!")
        else:
            print(f"\n⚠️  Response still over 2 seconds")
    else:
        print("  Unable to compare - some tests failed")
    
    # Test cache performance
    print("\n5️⃣  Testing Cache Performance...")
    print("  Running same query again...")
    
    start = time.time()
    result2 = await analyze_and_respond(
        screenshot=screenshot,
        user_query=test_query,
        response_type="confirmation",
        analysis_depth="fast"
    )
    cached_time = time.time() - start
    
    print(f"  Second run time: {cached_time:.2f}s")
    if result2.get('cached'):
        print(f"  ✅ Cache hit! {original_time/cached_time if original_time else 'N/A'}x faster")
    else:
        print(f"  ❌ Cache miss")
    
    print("\n✨ Performance optimization complete!")


async def test_model_performance():
    """Test different Claude models for speed"""
    print("\n\n🏃 Testing Model Performance...")
    print("=" * 60)
    
    models = [
        ("claude-3-haiku-20240307", "Haiku (Fastest)"),
        ("claude-3-sonnet-20240229", "Sonnet (Balanced)"),
        ("claude-3-opus-20240229", "Opus (Most Capable)")
    ]
    
    test_prompt = "Describe this screen briefly"
    
    try:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic()
        
        for model_id, model_name in models:
            print(f"\nTesting {model_name}...")
            
            try:
                start = time.time()
                message = await client.messages.create(
                    model=model_id,
                    max_tokens=100,
                    messages=[{
                        "role": "user",
                        "content": test_prompt
                    }]
                )
                elapsed = time.time() - start
                print(f"  ✓ Response time: {elapsed:.2f}s")
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                
    except Exception as e:
        print(f"Model testing unavailable: {e}")


if __name__ == "__main__":
    print("🤖 JARVIS Vision Performance Test\n")
    
    # Check for API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("⚠️  Warning: No ANTHROPIC_API_KEY found")
        print("  Set it with: export ANTHROPIC_API_KEY='your-key'")
        print("  Testing will use mock responses\n")
    
    # Run tests
    asyncio.run(test_performance())
    
    # Optional: Test model speeds
    if os.getenv('ANTHROPIC_API_KEY'):
        asyncio.run(test_model_performance())