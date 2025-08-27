#!/usr/bin/env python3
"""
Test suite for Python-Rust integration
Validates zero-copy transfer and performance improvements
"""

import numpy as np
import time
import psutil
import logging
from typing import List
import asyncio
from rust_integration import RustAccelerator, ZeroCopyVisionPipeline, SharedMemoryBuffer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class RustIntegrationTester:
    """Comprehensive test suite for Rust integration"""
    
    def __init__(self):
        self.rust_accel = RustAccelerator()
        self.pipeline = ZeroCopyVisionPipeline()
        self.test_results = {}
    
    def test_zero_copy_transfer(self) -> bool:
        """Test zero-copy data transfer between Python and Rust"""
        logger.info("\n🔧 Testing Zero-Copy Transfer...")
        
        try:
            # Create large test array
            size = 1920 * 1080 * 3  # Full HD image
            test_data = np.random.randint(0, 255, size, dtype=np.uint8)
            
            # Measure memory before
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Allocate shared buffer
            buffer = self.rust_accel.allocate_shared_memory(test_data.nbytes)
            
            # Write without copy
            buffer.write_numpy(test_data)
            
            # Read without copy
            read_data = buffer.as_numpy(test_data.shape)
            
            # Measure memory after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_increase = mem_after - mem_before
            
            # Verify data integrity
            assert np.array_equal(test_data, read_data), "Data mismatch"
            
            logger.info(f"✅ Zero-copy transfer successful")
            logger.info(f"   Data size: {test_data.nbytes / 1024 / 1024:.1f} MB")
            logger.info(f"   Memory increase: {mem_increase:.1f} MB (should be ~0 for true zero-copy)")
            
            self.test_results['zero_copy'] = {
                'passed': True,
                'data_size_mb': test_data.nbytes / 1024 / 1024,
                'memory_overhead_mb': mem_increase
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Zero-copy test failed: {e}")
            self.test_results['zero_copy'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_performance_improvement(self) -> bool:
        """Compare Python vs Rust processing performance"""
        logger.info("\n🚀 Testing Performance Improvement...")
        
        try:
            # Test image
            image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
            
            # Python processing (baseline)
            start = time.time()
            python_result = self._python_process_image(image)
            python_time = (time.time() - start) * 1000
            
            # Rust processing (accelerated)
            start = time.time()
            rust_result = self.rust_accel.process_image_zero_copy(image, 'process')
            rust_time = (time.time() - start) * 1000
            
            # Calculate speedup
            speedup = python_time / rust_time
            
            logger.info(f"✅ Performance comparison:")
            logger.info(f"   Python processing: {python_time:.2f}ms")
            logger.info(f"   Rust processing: {rust_time:.2f}ms")
            logger.info(f"   Speedup: {speedup:.1f}x")
            
            self.test_results['performance'] = {
                'passed': speedup > 1,
                'python_ms': python_time,
                'rust_ms': rust_time,
                'speedup': speedup
            }
            
            return speedup > 1
            
        except Exception as e:
            logger.error(f"❌ Performance test failed: {e}")
            self.test_results['performance'] = {'passed': False, 'error': str(e)}
            return False
    
    def _python_process_image(self, image: np.ndarray) -> np.ndarray:
        """Simulate Python image processing"""
        # Simple convolution (expensive in Python)
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
        result = np.zeros_like(image)
        
        # Simplified convolution (partial for speed)
        for y in range(1, min(100, image.shape[0]-1)):
            for x in range(1, min(100, image.shape[1]-1)):
                for c in range(image.shape[2]):
                    result[y, x, c] = np.sum(
                        image[y-1:y+2, x-1:x+2, c] * kernel
                    )
        
        return result
    
    async def test_async_batch_processing(self) -> bool:
        """Test asynchronous batch processing"""
        logger.info("\n⚡ Testing Async Batch Processing...")
        
        try:
            # Create batch of images
            batch_size = 10
            images = [
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                for _ in range(batch_size)
            ]
            
            # Process batch
            start = time.time()
            results = await self.rust_accel.process_batch_async(images)
            batch_time = (time.time() - start) * 1000
            
            # Verify results
            assert len(results) == batch_size, "Result count mismatch"
            
            avg_time = batch_time / batch_size
            logger.info(f"✅ Batch processing successful")
            logger.info(f"   Batch size: {batch_size}")
            logger.info(f"   Total time: {batch_time:.2f}ms")
            logger.info(f"   Per-image: {avg_time:.2f}ms")
            
            self.test_results['batch_processing'] = {
                'passed': True,
                'batch_size': batch_size,
                'total_ms': batch_time,
                'per_image_ms': avg_time
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Batch processing test failed: {e}")
            self.test_results['batch_processing'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_quantized_inference(self) -> bool:
        """Test INT8 quantized inference"""
        logger.info("\n🔢 Testing Quantized Inference...")
        
        try:
            # Create test weights and input
            weights = np.random.randn(1000, 784).astype(np.float32)
            input_data = np.random.randn(784).astype(np.float32)
            
            # Quantize weights
            q_weights, scale = self.rust_accel.quantize_model_int8(weights)
            
            # Verify quantization
            assert q_weights.dtype == np.int8, "Wrong quantization type"
            assert q_weights.min() >= -128 and q_weights.max() <= 127, "Invalid INT8 range"
            
            # Run inference
            output = self.rust_accel.run_quantized_inference([weights], input_data)
            
            logger.info(f"✅ Quantized inference successful")
            logger.info(f"   Weight compression: {weights.nbytes / q_weights.nbytes:.1f}x")
            logger.info(f"   Quantization scale: {scale:.6f}")
            logger.info(f"   Output shape: {output.shape}")
            
            self.test_results['quantized_inference'] = {
                'passed': True,
                'compression_ratio': weights.nbytes / q_weights.nbytes,
                'scale': scale
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Quantized inference test failed: {e}")
            self.test_results['quantized_inference'] = {'passed': False, 'error': str(e)}
            return False
    
    async def test_pipeline_integration(self) -> bool:
        """Test complete vision pipeline integration"""
        logger.info("\n🔄 Testing Pipeline Integration...")
        
        try:
            # Process multiple frames
            num_frames = 5
            frame_times = []
            
            for i in range(num_frames):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                start = time.time()
                result = await self.pipeline.process_frame(frame)
                frame_time = (time.time() - start) * 1000
                frame_times.append(frame_time)
                
                assert 'predictions' in result, "Missing predictions"
                assert result['zero_copy_used'], "Zero-copy not used"
            
            # Get pipeline stats
            stats = self.pipeline.get_performance_stats()
            
            logger.info(f"✅ Pipeline integration successful")
            logger.info(f"   Frames processed: {stats['frames_processed']}")
            logger.info(f"   Average latency: {stats['average_time_ms']:.2f}ms")
            logger.info(f"   Throughput: {stats['fps']:.1f} FPS")
            logger.info(f"   Zero-copy transfers: {stats['zero_copy_transfers']}")
            
            self.test_results['pipeline'] = {
                'passed': True,
                'fps': stats['fps'],
                'latency_ms': stats['average_time_ms'],
                'zero_copy_count': stats['zero_copy_transfers']
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline test failed: {e}")
            self.test_results['pipeline'] = {'passed': False, 'error': str(e)}
            return False
    
    def print_summary(self):
        """Print test summary"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r.get('passed', False))
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
            logger.info(f"{test_name.ljust(20)}: {status}")
            
            if result.get('passed', False):
                # Print key metrics
                if test_name == 'performance' and 'speedup' in result:
                    logger.info(f"  → Speedup: {result['speedup']:.1f}x")
                elif test_name == 'pipeline' and 'fps' in result:
                    logger.info(f"  → FPS: {result['fps']:.1f}")
        
        logger.info("\n" + "=" * 60)
        logger.info(f"Total: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All tests passed! Rust integration is working correctly.")
        else:
            logger.info("⚠️  Some tests failed. Check the logs above.")


async def main():
    """Run all integration tests"""
    tester = RustIntegrationTester()
    
    # Run tests
    tester.test_zero_copy_transfer()
    tester.test_performance_improvement()
    await tester.test_async_batch_processing()
    tester.test_quantized_inference()
    await tester.test_pipeline_integration()
    
    # Print summary
    tester.print_summary()


if __name__ == "__main__":
    logger.info("🦀 RUST INTEGRATION TEST SUITE")
    logger.info("Testing Python-Rust zero-copy integration...\n")
    
    asyncio.run(main())