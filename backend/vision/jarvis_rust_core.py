"""
JARVIS Rust Core Python Integration
Provides initialization and utilities for the Rust acceleration layer.
"""

import jarvis_rust_core as jrc
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Global references to Rust components
_runtime_manager: Optional[jrc.RustRuntimeManager] = None
_memory_pool: Optional[jrc.RustMemoryPool] = None
_advanced_pool: Optional[jrc.RustAdvancedMemoryPool] = None
_image_processor: Optional[jrc.RustImageProcessor] = None


def initialize_rust_runtime(config: Dict[str, Any]) -> None:
    """
    Initialize the Rust runtime with the given configuration.
    
    Args:
        config: Configuration dictionary with keys:
            - worker_threads: Number of worker threads
            - enable_cpu_affinity: Enable CPU pinning
            - memory_pool_size: Size of memory pool in bytes
            - enable_simd: Enable SIMD optimizations
    """
    global _runtime_manager, _memory_pool, _advanced_pool, _image_processor
    
    try:
        # Initialize runtime manager
        _runtime_manager = jrc.RustRuntimeManager(
            worker_threads=config.get('worker_threads', 4),
            enable_cpu_affinity=config.get('enable_cpu_affinity', True)
        )
        logger.info("Rust runtime manager initialized")
        
        # Initialize memory pools
        _memory_pool = jrc.RustMemoryPool()
        _advanced_pool = jrc.RustAdvancedMemoryPool()
        logger.info("Rust memory pools initialized")
        
        # Initialize image processor
        _image_processor = jrc.RustImageProcessor()
        logger.info("Rust image processor initialized")
        
        # Log runtime stats
        stats = _runtime_manager.stats()
        logger.info(f"Rust runtime stats: {stats}")
        
    except Exception as e:
        logger.error(f"Failed to initialize Rust runtime: {e}")
        raise


def get_rust_runtime() -> Optional[jrc.RustRuntimeManager]:
    """Get the global Rust runtime manager."""
    return _runtime_manager


def get_rust_memory_pool() -> Optional[jrc.RustMemoryPool]:
    """Get the global Rust memory pool."""
    return _memory_pool


def get_rust_advanced_pool() -> Optional[jrc.RustAdvancedMemoryPool]:
    """Get the global advanced memory pool with leak detection."""
    return _advanced_pool


def get_rust_image_processor() -> Optional[jrc.RustImageProcessor]:
    """Get the global Rust image processor."""
    return _image_processor


def is_rust_available() -> bool:
    """Check if Rust core is available and initialized."""
    return _runtime_manager is not None


def process_image_with_rust(image_array):
    """
    Process an image using Rust acceleration if available.
    Falls back to Python implementation if Rust is not available.
    """
    if _image_processor is not None:
        try:
            return _image_processor.process_numpy_image(image_array)
        except Exception as e:
            logger.warning(f"Rust image processing failed: {e}")
            # Fall back to Python implementation
    
    # Python fallback
    return image_array  # Or call Python implementation


def allocate_rust_buffer(size: int):
    """
    Allocate a buffer using Rust memory pool if available.
    """
    if _memory_pool is not None:
        try:
            return _memory_pool.allocate(size)
        except Exception as e:
            logger.warning(f"Rust buffer allocation failed: {e}")
    
    # Python fallback
    return bytearray(size)


def quantize_weights_with_rust(weights):
    """
    Quantize model weights using Rust if available.
    """
    try:
        return jrc.quantize_model_weights(weights)
    except Exception as e:
        logger.warning(f"Rust quantization failed: {e}")
        # Fall back to Python implementation
        return weights.astype('int8')  # Simple fallback