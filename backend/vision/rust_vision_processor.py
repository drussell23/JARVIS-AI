#!/usr/bin/env python3
"""
Rust-powered Vision Processor for JARVIS
High-performance vision data processing using Rust extension
"""

import sys
import os
import time
import logging
from typing import List, Optional, Dict, Any
import numpy as np

# Add the native extensions path
sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "native_extensions", "rust_processor")
)

try:
    import rust_processor

    RUST_AVAILABLE = True
    logging.info("✅ Rust processor loaded successfully")
except ImportError as e:
    RUST_AVAILABLE = False
    logging.warning(f"⚠️  Rust processor not available: {e}")

logger = logging.getLogger(__name__)


class RustVisionProcessor:
    """High-performance vision processing using Rust extension"""

    def __init__(self):
        self.rust_available = RUST_AVAILABLE
        if self.rust_available:
            logger.info("🚀 Rust vision processor initialized")
        else:
            logger.warning("⚠️  Falling back to Python implementation")

    def process_vision_data(self, data: np.ndarray) -> np.ndarray:
        """Process vision data with Rust performance"""
        if not self.rust_available:
            return self._python_fallback(data)

        try:
            # Convert numpy array to list for Rust
            data_list = data.flatten().tolist()

            # Process with Rust
            start_time = time.time()
            processed_list = rust_processor.process_vision_data(data_list)
            processing_time = (time.time() - start_time) * 1000

            logger.debug(f"Rust processing time: {processing_time:.2f}ms")

            # Convert back to numpy array
            return np.array(processed_list).reshape(data.shape)

        except Exception as e:
            logger.error(f"Rust processing failed: {e}, falling back to Python")
            return self._python_fallback(data)

    def process_audio_data(self, data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Process audio data with Rust performance"""
        if not self.rust_available:
            return self._python_fallback(data)

        try:
            # Convert numpy array to list for Rust
            data_list = data.flatten().tolist()

            # Process with Rust
            start_time = time.time()
            processed_list = rust_processor.process_audio_data(data_list, sample_rate)
            processing_time = (time.time() - start_time) * 1000

            logger.debug(f"Rust audio processing time: {processing_time:.2f}ms")

            # Convert back to numpy array
            return np.array(processed_list).reshape(data.shape)

        except Exception as e:
            logger.error(f"Rust audio processing failed: {e}, falling back to Python")
            return self._python_fallback(data)

    def compress_data(
        self, data: np.ndarray, compression_factor: float = 2.0
    ) -> np.ndarray:
        """Compress data efficiently using Rust"""
        if not self.rust_available:
            return self._python_compression_fallback(data, compression_factor)

        try:
            # Convert numpy array to list for Rust
            data_list = data.flatten().tolist()

            # Compress with Rust
            start_time = time.time()
            compressed_list = rust_processor.compress_data(
                data_list, compression_factor
            )
            compression_time = (time.time() - start_time) * 1000

            logger.debug(f"Rust compression time: {compression_time:.2f}ms")

            # Convert back to numpy array
            return np.array(compressed_list)

        except Exception as e:
            logger.error(f"Rust compression failed: {e}, falling back to Python")
            return self._python_compression_fallback(data, compression_factor)

    def quantized_inference(
        self, input_data: np.ndarray, model_weights: np.ndarray
    ) -> np.ndarray:
        """Run quantized inference using Rust"""
        if not self.rust_available:
            return self._python_inference_fallback(input_data, model_weights)

        try:
            # Convert numpy arrays to lists for Rust
            input_list = input_data.flatten().tolist()
            weights_list = model_weights.flatten().tolist()

            # Run inference with Rust
            start_time = time.time()
            output_list = rust_processor.quantized_inference(input_list, weights_list)
            inference_time = (time.time() - start_time) * 1000

            logger.debug(f"Rust inference time: {inference_time:.2f}ms")

            # Convert back to numpy array
            return np.array(output_list)

        except Exception as e:
            logger.error(f"Rust inference failed: {e}, falling back to Python")
            return self._python_inference_fallback(input_data, model_weights)

    def _python_fallback(self, data: np.ndarray) -> np.ndarray:
        """Python fallback for vision processing"""
        # Simple normalization
        normalized = (data - 0.5) * 2.0
        return np.clip(normalized, -1.0, 1.0)

    def _python_compression_fallback(
        self, data: np.ndarray, compression_factor: float
    ) -> np.ndarray:
        """Python fallback for data compression"""
        step = max(1, int(compression_factor))
        return data[::step]

    def _python_inference_fallback(
        self, input_data: np.ndarray, model_weights: np.ndarray
    ) -> np.ndarray:
        """Python fallback for inference"""
        # Simple weighted sum
        output_size = int(len(input_data) * 0.5)
        output = np.zeros(output_size)

        for i in range(output_size):
            idx = i * 2
            if idx < len(input_data):
                weight = model_weights[idx] if idx < len(model_weights) else 1.0
                output[i] = input_data[idx] * weight

        return np.clip(output, -1.0, 1.0)

    def benchmark_performance(self, data_size: int = 1000000) -> Dict[str, float]:
        """Benchmark Rust vs Python performance"""
        logger.info(f"🔍 Benchmarking performance with {data_size:,} data points...")

        # Generate test data
        test_data = np.random.random(data_size).astype(np.float32)

        # Benchmark Python
        start_time = time.time()
        python_result = self._python_fallback(test_data)
        python_time = (time.time() - start_time) * 1000

        # Benchmark Rust
        rust_time = float("inf")
        if self.rust_available:
            try:
                start_time = time.time()
                rust_result = self.process_vision_data(test_data)
                rust_time = (time.time() - start_time) * 1000
            except Exception as e:
                logger.error(f"Rust benchmark failed: {e}")

        # Calculate speedup
        speedup = python_time / rust_time if rust_time < float("inf") else 0

        results = {
            "python_time_ms": python_time,
            "rust_time_ms": rust_time,
            "speedup": speedup,
            "data_size": data_size,
        }

        logger.info(f"📊 Benchmark Results:")
        logger.info(f"  Python: {python_time:.2f}ms")
        logger.info(f"  Rust: {rust_time:.2f}ms")
        logger.info(f"  Speedup: {speedup:.2f}x")

        return results


# Test the processor
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    processor = RustVisionProcessor()

    # Run benchmark
    results = processor.benchmark_performance()

    # Test individual functions
    test_data = np.random.random(1000).astype(np.float32)

    print("\n🧪 Testing individual functions...")

    # Vision processing
    processed = processor.process_vision_data(test_data)
    print(f"✅ Vision processing: {len(processed)} samples")

    # Audio processing
    audio_processed = processor.process_audio_data(test_data, 44100)
    print(f"✅ Audio processing: {len(audio_processed)} samples")

    # Data compression
    compressed = processor.compress_data(test_data, 2.0)
    print(f"✅ Data compression: {len(compressed)} samples (from {len(test_data)})")

    print(f"\n🎉 Rust processor test completed successfully!")
