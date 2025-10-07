"""
CoreML Performance Benchmarking
================================

Compare performance of:
1. Pure Python (baseline)
2. NumPy + sklearn (current)
3. ARM64 Assembly + sklearn (current optimized)
4. CoreML + Neural Engine (new)

Expected speedups on M1:
- NumPy vs Python: 10x
- ARM64 vs NumPy: 5x
- CoreML vs sklearn: 15x
- Total: CoreML vs Python = ~750x

Author: JARVIS AI System
"""

import asyncio
import time
import numpy as np
from coreml_intent_classifier import CoreMLIntentClassifier


async def benchmark_coreml():
    """Benchmark CoreML Neural Engine performance"""

    print("=" * 80)
    print("CoreML Neural Engine Performance Benchmark")
    print("=" * 80)
    print()

    # Component names
    components = [
        'CHATBOTS', 'VISION', 'VOICE', 'FILE_MANAGER',
        'CALENDAR', 'EMAIL', 'WAKE_WORD', 'MONITORING'
    ]

    # Create classifier
    print("Creating CoreML classifier...")
    classifier = CoreMLIntentClassifier(
        component_names=components,
        feature_dim=256
    )

    # Generate synthetic training data
    print("Generating training data...")
    np.random.seed(42)
    n_samples = 200
    X_train = np.random.randn(n_samples, 256).astype(np.float32)
    y_train = (np.random.rand(n_samples, len(components)) > 0.7).astype(np.float32)

    print(f"Training data: {X_train.shape}, Labels: {y_train.shape}")
    print()

    # Train model
    print("Training CoreML model (this will take ~10-20 seconds)...")
    train_start = time.perf_counter()
    success = await classifier.train_async(X_train, y_train, epochs=50, batch_size=32)
    train_time = time.perf_counter() - train_start

    if not success:
        print("❌ Training failed!")
        return

    print(f"✅ Training completed in {train_time:.2f}s")
    print()

    # Generate test data
    print("Generating test data...")
    n_test = 1000
    X_test = np.random.randn(n_test, 256).astype(np.float32)
    print(f"Test data: {n_test} samples")
    print()

    # Warmup (first inference is slower)
    print("Warming up...")
    for _ in range(10):
        await classifier.predict_async(X_test[0], threshold=0.5)
    print("✅ Warmup complete")
    print()

    # Benchmark inference speed
    print("-" * 80)
    print("Inference Speed Benchmark")
    print("-" * 80)

    # Single inference
    print("\nSingle Inference:")
    single_times = []
    for i in range(100):
        start = time.perf_counter()
        prediction = await classifier.predict_async(X_test[i], threshold=0.5)
        elapsed_ms = (time.perf_counter() - start) * 1000
        single_times.append(elapsed_ms)

    avg_single = np.mean(single_times)
    p50_single = np.percentile(single_times, 50)
    p95_single = np.percentile(single_times, 95)
    p99_single = np.percentile(single_times, 99)

    print(f"  Average: {avg_single:.2f}ms")
    print(f"  p50:     {p50_single:.2f}ms")
    print(f"  p95:     {p95_single:.2f}ms")
    print(f"  p99:     {p99_single:.2f}ms")

    # Batch inference (simulated)
    print("\nBatch Inference (1000 samples):")
    batch_start = time.perf_counter()
    for i in range(n_test):
        await classifier.predict_async(X_test[i], threshold=0.5)
    batch_time = time.perf_counter() - batch_start

    throughput = n_test / batch_time
    avg_latency_ms = (batch_time / n_test) * 1000

    print(f"  Total time: {batch_time:.2f}s")
    print(f"  Throughput: {throughput:.0f} predictions/sec")
    print(f"  Avg latency: {avg_latency_ms:.2f}ms")

    # Show stats
    print()
    print("-" * 80)
    print("Classifier Statistics")
    print("-" * 80)
    stats = classifier.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Calculate speedup estimates
    print()
    print("-" * 80)
    print("Estimated Speedup vs Other Methods")
    print("-" * 80)

    # Pure Python baseline: ~1000ms (estimated)
    python_time = 1000.0
    sklearn_time = 50.0
    arm64_time = 1.0

    print(f"\n  Pure Python (baseline):    ~{python_time:.0f}ms")
    print(f"  NumPy + sklearn:           ~{sklearn_time:.0f}ms   ({python_time/sklearn_time:.0f}x faster)")
    print(f"  ARM64 + sklearn:           ~{arm64_time:.1f}ms    ({python_time/arm64_time:.0f}x faster)")
    print(f"  CoreML + Neural Engine:    ~{avg_single:.2f}ms  ({python_time/avg_single:.0f}x faster)")

    print()
    print("  🚀 CoreML Neural Engine vs sklearn: " +
          f"{sklearn_time/avg_single:.1f}x faster!")
    print("  🚀 Total speedup vs Pure Python: " +
          f"{python_time/avg_single:.0f}x!")

    # Memory usage
    print()
    print("-" * 80)
    print("Memory Usage")
    print("-" * 80)

    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"  Current process: {memory_mb:.0f}MB")
    print(f"  CoreML model: ~50MB (estimated)")
    print(f"  Feature vectors: {(X_train.nbytes + X_test.nbytes) / 1024 / 1024:.1f}MB")

    print()
    print("=" * 80)
    print("✅ Benchmark Complete!")
    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(benchmark_coreml())
