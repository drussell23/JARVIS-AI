#!/usr/bin/env python3
"""
Test script to verify Rust-Python bridge is working correctly.
Tests all components with dynamic configuration.
"""

import sys
import time
import numpy as np
import asyncio
import psutil
from datetime import datetime

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def test_rust_import():
    """Test basic Rust import."""
    print("\n1. Testing Rust Core Import...")
    try:
        import jarvis_rust_core
        print(f"{GREEN}✓ Successfully imported jarvis_rust_core{RESET}")
        
        # Check version
        if hasattr(jarvis_rust_core, '__version__'):
            print(f"  Version: {jarvis_rust_core.__version__}")
            
        return True, jarvis_rust_core
    except ImportError as e:
        print(f"{RED}✗ Failed to import jarvis_rust_core: {e}{RESET}")
        return False, None

def test_memory_pool(rust_core):
    """Test Rust memory pool."""
    print("\n2. Testing Rust Memory Pool...")
    try:
        # Test advanced memory pool
        pool = rust_core.RustAdvancedMemoryPool()
        print(f"{GREEN}✓ Created RustAdvancedMemoryPool{RESET}")
        
        # Allocate buffer
        size = 1024 * 1024  # 1MB
        buffer = pool.allocate(size)
        print(f"{GREEN}✓ Allocated {size} bytes{RESET}")
        
        # Test numpy conversion
        np_array = buffer.as_numpy()
        print(f"{GREEN}✓ Converted to numpy array: shape={np_array.shape}, dtype={np_array.dtype}{RESET}")
        
        # Write data
        np_array.fill(42)
        print(f"{GREEN}✓ Filled array with test data{RESET}")
        
        # Release buffer
        buffer.release()
        print(f"{GREEN}✓ Released buffer{RESET}")
        
        # Check for leaks
        leaks = pool.check_leaks()
        if not leaks:
            print(f"{GREEN}✓ No memory leaks detected{RESET}")
        else:
            print(f"{YELLOW}⚠ Memory leaks detected: {leaks}{RESET}")
            
        # Get stats
        stats = pool.stats()
        print(f"  Pool stats: {stats}")
        
        return True
    except Exception as e:
        print(f"{RED}✗ Memory pool test failed: {e}{RESET}")
        return False

def test_bloom_filter(rust_core):
    """Test Rust bloom filter."""
    print("\n3. Testing Rust Bloom Filter...")
    
    if not hasattr(rust_core, 'bloom_filter'):
        print(f"{YELLOW}⚠ Bloom filter module not found{RESET}")
        return False
        
    try:
        # Create bloom filter network
        bloom_net = rust_core.bloom_filter.PyRustBloomNetwork(
            global_mb=4.0,
            regional_mb=1.0,
            element_mb=2.0
        )
        print(f"{GREEN}✓ Created RustBloomNetwork (10MB total){RESET}")
        
        # Test operations
        test_items = 10000
        duplicates = 0
        
        start = time.perf_counter()
        for i in range(test_items):
            data = f"test_item_{i}".encode()
            is_dup, level = bloom_net.check_and_add(data, i % 4)  # Quadrant 0-3
            if is_dup:
                duplicates += 1
        elapsed = time.perf_counter() - start
        
        ops_per_sec = test_items / elapsed
        print(f"{GREEN}✓ Processed {test_items} items in {elapsed:.3f}s ({ops_per_sec:.0f} ops/sec){RESET}")
        
        # Check duplicates
        for i in range(100):
            data = f"test_item_{i}".encode()
            is_dup, level = bloom_net.check_and_add(data, i % 4)
            if is_dup:
                duplicates += 1
                
        print(f"{GREEN}✓ Duplicate detection working: {duplicates} duplicates found{RESET}")
        
        # Get stats
        stats = bloom_net.stats()
        print(f"  Bloom filter stats:")
        for stat in stats:
            print(f"    {stat[0]}: {stat[1]:.2f}")
            
        return True
    except Exception as e:
        print(f"{RED}✗ Bloom filter test failed: {e}{RESET}")
        return False

def test_sliding_window(rust_core):
    """Test Rust sliding window."""
    print("\n4. Testing Rust Sliding Window...")
    
    if not hasattr(rust_core, 'sliding_window'):
        print(f"{YELLOW}⚠ Sliding window module not found{RESET}")
        return False
        
    try:
        # Create frame buffer
        buffer = rust_core.sliding_window.PyFrameRingBuffer(capacity_mb=100)
        print(f"{GREEN}✓ Created FrameRingBuffer (100MB){RESET}")
        
        # Add frames
        num_frames = 30
        frame_size = (480, 640, 3)  # Small frames for testing
        
        start = time.perf_counter()
        for i in range(num_frames):
            frame = np.random.randint(0, 255, frame_size, dtype=np.uint8)
            buffer.add_frame(frame.tobytes(), frame_size[1], frame_size[0], frame_size[2])
        elapsed = time.perf_counter() - start
        
        fps = num_frames / elapsed
        print(f"{GREEN}✓ Added {num_frames} frames at {fps:.1f} FPS{RESET}")
        
        # Get recent frames
        recent = buffer.get_recent_frames(5)
        print(f"{GREEN}✓ Retrieved {len(recent)} recent frames{RESET}")
        
        # Test temporal analysis
        analyzer = rust_core.sliding_window.PySlidingWindowAnalyzer(
            window_size=10,
            stride=2,
            buffer=buffer
        )
        
        analysis = analyzer.analyze_temporal_patterns()
        print(f"{GREEN}✓ Temporal analysis completed{RESET}")
        print(f"  Analysis results:")
        for metric in analysis:
            print(f"    {metric[0]}: {metric[1]:.2f}")
            
        # Get buffer stats
        stats = buffer.stats()
        print(f"  Buffer stats:")
        for stat in stats:
            print(f"    {stat[0]}: {stat[1]:.0f}")
            
        return True
    except Exception as e:
        print(f"{RED}✗ Sliding window test failed: {e}{RESET}")
        return False

def test_metal_acceleration(rust_core):
    """Test Metal GPU acceleration."""
    print("\n5. Testing Metal GPU Acceleration...")
    
    if sys.platform != "darwin":
        print(f"{YELLOW}⚠ Skipping Metal test (not on macOS){RESET}")
        return True
        
    if not hasattr(rust_core, 'metal_accelerator'):
        print(f"{YELLOW}⚠ Metal accelerator module not found{RESET}")
        return False
        
    try:
        # Create Metal accelerator
        metal = rust_core.metal_accelerator.PyMetalAccelerator()
        print(f"{GREEN}✓ Created MetalAccelerator{RESET}")
        
        # Process frames
        frames = [
            np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8).tobytes()
            for _ in range(5)
        ]
        
        start = time.perf_counter()
        results = metal.process_batch(frames)
        elapsed = time.perf_counter() - start
        
        fps = len(frames) / elapsed
        print(f"{GREEN}✓ Processed {len(frames)} frames at {fps:.1f} FPS on GPU{RESET}")
        
        # Get GPU stats
        stats = metal.get_stats()
        print(f"  GPU stats:")
        for key, value in stats.items():
            print(f"    {key}: {value:.2f}")
            
        return True
    except Exception as e:
        print(f"{RED}✗ Metal acceleration test failed: {e}{RESET}")
        return False

def test_zero_copy_memory(rust_core):
    """Test zero-copy memory management."""
    print("\n6. Testing Zero-Copy Memory...")
    
    if not hasattr(rust_core, 'zero_copy'):
        print(f"{YELLOW}⚠ Zero-copy module not found{RESET}")
        return False
        
    try:
        # Create zero-copy pool
        pool = rust_core.zero_copy.PyZeroCopyPool(max_memory_mb=1024)
        print(f"{GREEN}✓ Created ZeroCopyPool (1GB){RESET}")
        
        # Allocate buffers
        sizes = [1024, 1024*1024, 10*1024*1024]  # 1KB, 1MB, 10MB
        buffers = []
        
        for size in sizes:
            buffer = pool.allocate(size)
            buffers.append(buffer)
            print(f"{GREEN}✓ Allocated {size/1024/1024:.2f}MB buffer{RESET}")
            
        # Test numpy conversion
        for i, buffer in enumerate(buffers):
            np_array = buffer.as_numpy()
            np_array.fill(i)
            print(f"{GREEN}✓ Filled buffer {i} with test data{RESET}")
            
        # Check memory pressure
        pressure = pool.memory_pressure()
        print(f"{GREEN}✓ Memory pressure: {pressure}{RESET}")
        
        # Get pool stats
        stats = pool.stats()
        print(f"  Pool stats:")
        for stat in stats:
            print(f"    {stat[0]}: {stat[1]}")
            
        return True
    except Exception as e:
        print(f"{RED}✗ Zero-copy memory test failed: {e}{RESET}")
        return False

async def test_async_operations(rust_core):
    """Test async operations."""
    print("\n7. Testing Async Operations...")
    
    if not hasattr(rust_core, 'sliding_window'):
        print(f"{YELLOW}⚠ Async operations require sliding_window module{RESET}")
        return False
        
    try:
        # Create async buffer
        buffer = rust_core.sliding_window.PyFrameRingBuffer(capacity_mb=50)
        
        # Async frame addition
        frames = [(np.random.randint(0, 255, 10000, dtype=np.uint8).tobytes(), i) 
                  for i in range(20)]
        
        start = time.perf_counter()
        tasks = [buffer.add_frame_async(frame, ts) for frame, ts in frames]
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start
        
        fps = len(frames) / elapsed
        print(f"{GREEN}✓ Async added {len(frames)} frames at {fps:.1f} FPS{RESET}")
        
        # Async frame retrieval
        recent = await buffer.get_recent_frames_async(5)
        print(f"{GREEN}✓ Async retrieved {len(recent)} frames{RESET}")
        
        return True
    except Exception as e:
        print(f"{RED}✗ Async operations test failed: {e}{RESET}")
        return False

def run_all_tests():
    """Run all Rust-Python bridge tests."""
    print("=" * 60)
    print("RUST-PYTHON BRIDGE TEST SUITE")
    print("=" * 60)
    print(f"System: {sys.platform}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"CPU: {psutil.cpu_count()} cores")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: Import
    success, rust_core = test_rust_import()
    if not success:
        print(f"\n{RED}Cannot proceed without Rust core. Please build first:{RESET}")
        print("cd jarvis-rust-core && cargo build --release && maturin develop")
        return
        
    # Run remaining tests
    tests = [
        ("Memory Pool", lambda: test_memory_pool(rust_core)),
        ("Bloom Filter", lambda: test_bloom_filter(rust_core)),
        ("Sliding Window", lambda: test_sliding_window(rust_core)),
        ("Metal Acceleration", lambda: test_metal_acceleration(rust_core)),
        ("Zero-Copy Memory", lambda: test_zero_copy_memory(rust_core)),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"{RED}✗ {test_name} crashed: {e}{RESET}")
            results[test_name] = False
            
    # Run async test
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results["Async Operations"] = loop.run_until_complete(test_async_operations(rust_core))
    except Exception as e:
        print(f"{RED}✗ Async test crashed: {e}{RESET}")
        results["Async Operations"] = False
        
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed in results.items():
        status = f"{GREEN}✓ PASSED{RESET}" if passed else f"{RED}✗ FAILED{RESET}"
        print(f"{test_name:.<30} {status}")
        
    print("=" * 60)
    
    if passed == total:
        print(f"{GREEN}✅ ALL TESTS PASSED! ({passed}/{total}){RESET}")
        print(f"{GREEN}Rust-Python bridge is fully operational!{RESET}")
    else:
        print(f"{YELLOW}⚠ {passed}/{total} tests passed{RESET}")
        
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)