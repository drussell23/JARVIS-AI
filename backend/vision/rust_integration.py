"""
Advanced Rust integration for JARVIS with memory safety and performance optimizations.

This module provides Python bindings to the high-performance Rust core with:
- Zero-copy memory management with leak detection
- Advanced async runtime with CPU affinity
- Quantized ML inference for reduced memory usage
- Hardware-accelerated image processing
"""

import os
import sys
import logging
import asyncio
import numpy as np
from typing import Optional, List, Dict, Any, Callable, Union
from contextlib import contextmanager
import psutil

# Try to import the Rust core
try:
    import jarvis_rust_core
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    jarvis_rust_core = None  # Define as None to prevent NameError
    logging.warning("Rust core not available. Run 'maturin develop' in backend/vision/jarvis-rust-core")

logger = logging.getLogger(__name__)


class SharedMemoryBuffer:
    """Shared memory buffer for zero-copy data transfer."""
    
    def __init__(self, size: int):
        """Initialize shared memory buffer."""
        self.size = size
        self.data = bytearray(size)
        self.position = 0
        
    def write(self, data: bytes) -> int:
        """Write data to buffer."""
        data_len = len(data)
        if self.position + data_len > self.size:
            raise ValueError("Buffer overflow")
        
        self.data[self.position:self.position + data_len] = data
        self.position += data_len
        return data_len
    
    def read(self, size: int) -> bytes:
        """Read data from buffer."""
        if size > self.position:
            size = self.position
        
        data = bytes(self.data[:size])
        # Shift remaining data
        self.data[:self.position - size] = self.data[size:self.position]
        self.position -= size
        return data
    
    def clear(self):
        """Clear the buffer."""
        self.position = 0
    
    def __len__(self):
        """Get current data size."""
        return self.position


class RustAccelerator:
    """Main interface for Rust acceleration in JARVIS."""
    
    def __init__(self, 
                 enable_memory_pool: bool = True,
                 enable_runtime_manager: bool = True,
                 worker_threads: Optional[int] = None,
                 enable_cpu_affinity: bool = True):
        """
        Initialize Rust accelerator.
        
        Args:
            enable_memory_pool: Use advanced memory pooling with leak detection
            enable_runtime_manager: Use advanced async runtime
            worker_threads: Number of worker threads (defaults to CPU count)
            enable_cpu_affinity: Pin threads to CPU cores for better performance
        """
        if not RUST_AVAILABLE:
            raise ImportError("Rust core not available. Please build the Rust extension.")
        
        # Initialize Rust core
        jarvis_rust_core.initialize()
        
        self.memory_pool = None
        self.runtime_manager = None
        
        if enable_memory_pool:
            self.memory_pool = jarvis_rust_core.RustAdvancedMemoryPool()
            logger.info("Advanced memory pool initialized")
        
        if enable_runtime_manager:
            self.runtime_manager = jarvis_rust_core.RustRuntimeManager(
                worker_threads=worker_threads,
                enable_cpu_affinity=enable_cpu_affinity
            )
            logger.info("Runtime manager initialized with %s workers", 
                       worker_threads or psutil.cpu_count())
    
    @contextmanager
    def allocate_buffer(self, size: int):
        """
        Allocate a tracked buffer that automatically returns to pool.
        
        Args:
            size: Buffer size in bytes
            
        Yields:
            Tracked buffer as numpy array
        """
        if not self.memory_pool:
            # Fallback to numpy
            yield np.zeros(size, dtype=np.uint8)
            return
        
        buffer = self.memory_pool.allocate(size)
        try:
            yield buffer.as_numpy()
        finally:
            buffer.release()
    
    def run_cpu_task(self, func: Callable[[], Any]) -> Any:
        """
        Run CPU-bound task with optimal thread placement.
        
        Args:
            func: Function to execute
            
        Returns:
            Function result
        """
        if not self.runtime_manager:
            return func()
        
        return self.runtime_manager.run_cpu_task(func)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        if not self.memory_pool:
            return {}
        
        stats = self.memory_pool.stats()
        
        # Check for leaks
        leaks = self.memory_pool.check_leaks()
        if leaks:
            logger.warning("Memory leaks detected: %s", leaks)
        
        return {
            'pool_stats': stats,
            'leaks': leaks,
            'system_memory': {
                'used_gb': psutil.virtual_memory().used / 1e9,
                'available_gb': psutil.virtual_memory().available / 1e9,
                'percent': psutil.virtual_memory().percent
            }
        }
    
    def get_runtime_stats(self) -> Dict[str, Any]:
        """Get runtime statistics."""
        if not self.runtime_manager:
            return {}
        
        return self.runtime_manager.stats()

class RustImageProcessor:
    """Hardware-accelerated image processing using Rust."""
    
    def __init__(self):
        """Initialize image processor."""
        if not RUST_AVAILABLE:
            raise ImportError("Rust core not available")
        
        self.processor = jarvis_rust_core.RustImageProcessor()
        self.accelerator = RustAccelerator()
    
    def process_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process batch of images with zero-copy optimization.
        
        Args:
            images: List of numpy arrays (H, W, C)
            
        Returns:
            Processed images
        """
        return self.processor.process_batch_zero_copy(images)
    
    def resize_batch(self, images: List[np.ndarray], 
                     target_size: tuple) -> List[np.ndarray]:
        """
        Resize batch of images efficiently.
        
        Args:
            images: Input images
            target_size: (width, height)
            
        Returns:
            Resized images
        """
        # This would need to be implemented in Rust
        # For now, use the process_batch as example
        return self.process_batch(images)

class RustQuantizedModel:
    """Quantized ML model for memory-efficient inference."""
    
    def __init__(self, use_simd: bool = True, thread_count: Optional[int] = None):
        """
        Initialize quantized model.
        
        Args:
            use_simd: Enable SIMD optimizations
            thread_count: Number of threads for inference
        """
        if not RUST_AVAILABLE:
            raise ImportError("Rust core not available")
        
        self.model = jarvis_rust_core.RustQuantizedModel(
            use_simd=use_simd,
            thread_count=thread_count or psutil.cpu_count()
        )
        self.accelerator = RustAccelerator()
    
    def add_linear_layer(self, weights: np.ndarray, bias: Optional[np.ndarray] = None):
        """
        Add quantized linear layer.
        
        Args:
            weights: Weight matrix (will be quantized to INT8)
            bias: Optional bias vector
        """
        self.model.add_linear_layer(weights, bias)
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference with quantized model.
        
        Args:
            input_data: Input tensor
            
        Returns:
            Output predictions
        """
        return self.model.infer(input_data)
    
    @staticmethod
    def quantize_weights(weights: np.ndarray) -> List[int]:
        """
        Quantize float32 weights to INT8.
        
        Args:
            weights: Float32 weight matrix
            
        Returns:
            Quantized weights as INT8
        """
        return jarvis_rust_core.quantize_model_weights(weights)

class ZeroCopyVisionPipeline:
    """Zero-copy vision pipeline for maximum performance."""
    
    def __init__(self, enable_quantization: bool = True):
        """Initialize zero-copy vision pipeline."""
        if not RUST_AVAILABLE:
            raise ImportError("Rust core not available for zero-copy pipeline")
        
        self.enable_quantization = enable_quantization
        self.processor = RustImageProcessor()
        self.quantized_model = None
        
        logger.info("Zero-copy vision pipeline initialized")
    
    async def process_image(self, image_data: Union[bytes, np.ndarray], 
                          model_name: Optional[str] = None) -> Dict[str, Any]:
        """Process image with zero-copy optimization."""
        try:
            # Convert to numpy array if needed
            if isinstance(image_data, bytes):
                import cv2
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                image = image_data
                
            # Process with hardware acceleration
            result = await self.processor.process_image_async(image)
            
            return {
                "success": True,
                "features": result.get("features", []),
                "objects": result.get("objects", []),
                "processing_time_ms": result.get("processing_time_ms", 0)
            }
            
        except Exception as e:
            logger.error(f"Zero-copy pipeline error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'processor'):
            self.processor = None
        if hasattr(self, 'quantized_model'):
            self.quantized_model = None


class RustMemoryMonitor:
    """Monitor and prevent memory leaks."""
    
    def __init__(self, check_interval: float = 10.0):
        """
        Initialize memory monitor.
        
        Args:
            check_interval: Leak check interval in seconds
        """
        self.accelerator = RustAccelerator(enable_memory_pool=True)
        self.check_interval = check_interval
        self._monitoring = False
        self._monitor_task = None
    
    async def start_monitoring(self):
        """Start background memory monitoring."""
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Memory monitoring started")
    
    async def stop_monitoring(self):
        """Stop memory monitoring."""
        self._monitoring = False
        if self._monitor_task:
            await self._monitor_task
        logger.info("Memory monitoring stopped")
    
    async def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            stats = self.accelerator.get_memory_stats()
            
            # Log statistics
            if stats.get('pool_stats'):
                pool_stats = stats['pool_stats']
                logger.debug(
                    "Memory pool: %d active, %d MB allocated, pressure: %s",
                    pool_stats.get('total_active', 0),
                    pool_stats.get('total_allocated_bytes', 0) / 1e6,
                    pool_stats.get('memory_pressure', 'unknown')
                )
            
            # Check for leaks
            leaks = stats.get('leaks', [])
            if leaks:
                logger.warning("Detected %d memory leaks", len(leaks))
                for leak in leaks:
                    logger.warning("Leak: %s", leak)
            
            await asyncio.sleep(self.check_interval)

def benchmark_rust_performance():
    """Benchmark Rust acceleration performance."""
    import time
    
    if not RUST_AVAILABLE:
        print("Rust core not available")
        return
    
    print("Benchmarking Rust acceleration...")
    
    # Initialize
    accelerator = RustAccelerator()
    
    # Memory allocation benchmark
    sizes = [1024, 1024*1024, 10*1024*1024]  # 1KB, 1MB, 10MB
    
    for size in sizes:
        # Rust allocation
        start = time.perf_counter()
        with accelerator.allocate_buffer(size) as buf:
            buf.fill(42)  # Touch memory
        rust_time = time.perf_counter() - start
        
        # NumPy allocation
        start = time.perf_counter()
        buf = np.zeros(size, dtype=np.uint8)
        buf.fill(42)
        numpy_time = time.perf_counter() - start
        
        print(f"Buffer size {size/1024:.0f}KB: Rust={rust_time*1000:.2f}ms, "
              f"NumPy={numpy_time*1000:.2f}ms, "
              f"Speedup={numpy_time/rust_time:.2f}x")
    
    # CPU task benchmark
    def cpu_task():
        """Compute-intensive task."""
        total = 0
        for i in range(1000000):
            total += i * i
        return total
    
    # Rust execution
    start = time.perf_counter()
    result1 = accelerator.run_cpu_task(cpu_task)
    rust_time = time.perf_counter() - start
    
    # Direct execution
    start = time.perf_counter()
    result2 = cpu_task()
    direct_time = time.perf_counter() - start
    
    print(f"\nCPU task: Rust={rust_time*1000:.2f}ms, "
          f"Direct={direct_time*1000:.2f}ms")
    
    # Print statistics
    print("\nMemory statistics:")
    print(accelerator.get_memory_stats())
    
    print("\nRuntime statistics:")
    print(accelerator.get_runtime_stats())

# Global accelerator instance
_global_accelerator: Optional[RustAccelerator] = None

def initialize_rust_acceleration(**kwargs) -> RustAccelerator:
    """
    Initialize global Rust acceleration.
    
    Returns:
        Configured RustAccelerator instance
    """
    global _global_accelerator
    
    if _global_accelerator is None:
        _global_accelerator = RustAccelerator(**kwargs)
        logger.info("Global Rust acceleration initialized")
    
    return _global_accelerator

def get_rust_accelerator() -> Optional[RustAccelerator]:
    """Get global Rust accelerator instance."""
    return _global_accelerator

if __name__ == "__main__":
    # Run benchmark
    benchmark_rust_performance()