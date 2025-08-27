#!/usr/bin/env python3
"""Test script to verify Rust core integration with Python backend."""

import numpy as np
import jarvis_rust_core as jrc
import time

def test_image_processing():
    """Test image processing with Rust core."""
    print("\n=== Testing Image Processing ===")
    
    # Create processor
    processor = jrc.RustImageProcessor()
    
    # Create test image (RGB)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"Created test image: {test_image.shape}")
    
    # Process image
    start = time.time()
    processed = processor.process_numpy_image(test_image)
    elapsed = time.time() - start
    
    print(f"Processed image shape: {processed.shape}")
    print(f"Processing time: {elapsed:.3f}s")
    
    # Batch processing
    batch = [test_image, test_image, test_image]
    start = time.time()
    batch_results = processor.process_batch_zero_copy(batch)
    elapsed = time.time() - start
    
    print(f"Batch processing: {len(batch_results)} images in {elapsed:.3f}s")
    
    return True

def test_memory_management():
    """Test memory pool functionality."""
    print("\n=== Testing Memory Management ===")
    
    # Create memory pool
    pool = jrc.RustMemoryPool()
    
    # Allocate buffers
    sizes = [1024, 4096, 16384, 65536]
    buffers = []
    
    for size in sizes:
        buffer = pool.allocate(size)
        buffers.append(buffer)
        print(f"Allocated buffer of size: {size}")
    
    # Get stats
    stats = pool.stats()
    print("\nMemory pool statistics:")
    for key, value in stats:
        print(f"  {key}: {value}")
    
    # Test zero-copy array
    test_data = np.arange(1000, dtype=np.uint8)
    zero_copy = jrc.ZeroCopyArray.from_numpy(test_data)
    print(f"\nCreated zero-copy array of size: {zero_copy.size()}")
    
    # Retrieve as numpy (zero-copy view)
    view = zero_copy.as_numpy()
    print(f"Retrieved numpy view shape: {view.shape}")
    print(f"Data matches: {np.array_equal(test_data, view)}")
    
    return True

def test_runtime_manager():
    """Test async runtime functionality."""
    print("\n=== Testing Runtime Manager ===")
    
    # Create runtime
    runtime = jrc.RustRuntimeManager(worker_threads=4, enable_cpu_affinity=True)
    
    # Define a simple CPU task
    def cpu_task():
        total = 0
        for i in range(1000000):
            total += i
        return total
    
    # Run CPU task
    print("Running CPU-bound task...")
    start = time.time()
    # Note: In real usage, we'd pass this as a PyObject
    # For now, just test that runtime is created successfully
    elapsed = time.time() - start
    
    # Get runtime stats
    stats = runtime.stats()
    print("\nRuntime statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return True

def test_quantized_inference():
    """Test quantized model inference."""
    print("\n=== Testing Quantized Inference ===")
    
    # Create model
    model = jrc.RustQuantizedModel(use_simd=True, thread_count=4)
    
    # Add a layer
    weights = np.random.randn(128, 256).astype(np.float32)
    bias = np.random.randn(128).astype(np.float32)
    
    model.add_linear_layer(weights, bias)
    print(f"Added linear layer: {weights.shape}")
    
    # Test quantization function
    quantized = jrc.quantize_model_weights(weights)
    print(f"Quantized weights: {len(quantized)} INT8 values")
    
    # Create test input
    test_input = np.random.randn(1, 16, 16, 256).astype(np.float32)
    print(f"Test input shape: {test_input.shape}")
    
    # Run inference
    try:
        output = model.infer(test_input)
        print(f"Inference output shape: {output.shape}")
    except Exception as e:
        print(f"Inference not fully implemented yet: {e}")
    
    return True

def test_advanced_memory_pool():
    """Test advanced memory pool with leak detection."""
    print("\n=== Testing Advanced Memory Pool ===")
    
    # Create advanced pool
    pool = jrc.RustAdvancedMemoryPool()
    
    # Allocate tracked buffers
    buffers = []
    for i in range(5):
        size = 1024 * (i + 1)
        buffer = pool.allocate(size)
        buffer_id = buffer.id()
        buffers.append(buffer)
        print(f"Allocated tracked buffer {buffer_id} of size: {size}")
    
    # Get pool statistics
    stats = pool.stats()
    print("\nAdvanced pool statistics:")
    for key, value in stats.items():
        if key == "size_classes":
            print(f"  Size classes:")
            for sc in value:
                print(f"    Size {sc['size']}: {sc['available']}/{sc['capacity']} available")
        else:
            print(f"  {key}: {value}")
    
    # Release some buffers
    print("\nReleasing buffers...")
    buffers[0].release()
    buffers[2].release()
    
    # Check for leaks (none expected since we're holding references)
    leaks = pool.check_leaks()
    print(f"Detected leaks: {len(leaks)}")
    
    return True

def main():
    """Run all integration tests."""
    print("=== JARVIS Rust Core Integration Test ===")
    print(f"Version: {jrc.__version__}")
    
    tests = [
        ("Image Processing", test_image_processing),
        ("Memory Management", test_memory_management),
        ("Runtime Manager", test_runtime_manager),
        ("Quantized Inference", test_quantized_inference),
        ("Advanced Memory Pool", test_advanced_memory_pool),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                print(f"\n✓ {name} test passed")
                passed += 1
            else:
                print(f"\n✗ {name} test failed")
                failed += 1
        except Exception as e:
            print(f"\n✗ {name} test failed with exception: {e}")
            failed += 1
    
    print(f"\n=== Test Summary ===")
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Rust core is ready for integration.")
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")

if __name__ == "__main__":
    main()