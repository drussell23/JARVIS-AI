#!/usr/bin/env python3
"""
Memory safety configuration for Claude Vision Analyzer
Prevents crashes on 16GB macOS systems running JARVIS
"""

import psutil
import asyncio
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MemorySafetyConfig:
    """Configuration to prevent memory-related crashes"""
    
    # Memory limits (in MB)
    PROCESS_MEMORY_LIMIT_MB = 2048  # 2GB max for vision analyzer
    MEMORY_WARNING_THRESHOLD_MB = 1536  # 1.5GB warning threshold
    MIN_SYSTEM_AVAILABLE_GB = 2.0  # Minimum 2GB system RAM available
    
    # Per-operation memory estimates (in MB)
    MEMORY_ESTIMATES = {
        'small_image': 20,      # 640x480 or smaller
        'medium_image': 100,    # 1920x1080 (HD)
        'large_image': 300,     # 4K or larger
        'concurrent_overhead': 10  # Per concurrent request
    }
    
    # Safe operating parameters
    SAFE_CONCURRENT_LIMIT = 10  # Reduced from default
    SAFE_CACHE_SIZE = 50  # Limited cache items
    
    @staticmethod
    def get_safe_config() -> Dict[str, Any]:
        """Get memory-safe configuration for ClaudeVisionAnalyzer"""
        return {
            # Core settings
            'max_concurrent_requests': MemorySafetyConfig.SAFE_CONCURRENT_LIMIT,
            'cache_enabled': True,
            'cache_size_mb': 100,  # 100MB cache limit
            'max_cache_items': MemorySafetyConfig.SAFE_CACHE_SIZE,
            
            # Memory thresholds
            'memory_threshold_percent': 60.0,  # More aggressive than default 70%
            
            # Image processing
            'max_image_dimension': 2048,  # Limit very large images
            'compression_enabled': True,
            'jpeg_quality': 85,
            
            # Sliding window for large images
            'sliding_window_threshold_pixels': 2_000_000,  # 2MP
            
            # Enable all memory optimizations
            'enable_memory_efficient_mode': True,
            'enable_dynamic_compression': True,
            'enable_aggressive_gc': True
        }
    
    @staticmethod
    def check_memory_safety() -> Dict[str, Any]:
        """Check current memory status and safety"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        process_mb = process.memory_info().rss / (1024**2)
        system_available_gb = memory.available / (1024**3)
        
        is_safe = (
            process_mb < MemorySafetyConfig.PROCESS_MEMORY_LIMIT_MB and
            system_available_gb > MemorySafetyConfig.MIN_SYSTEM_AVAILABLE_GB
        )
        
        return {
            'is_safe': is_safe,
            'process_mb': process_mb,
            'process_limit_mb': MemorySafetyConfig.PROCESS_MEMORY_LIMIT_MB,
            'system_available_gb': system_available_gb,
            'system_min_gb': MemorySafetyConfig.MIN_SYSTEM_AVAILABLE_GB,
            'warnings': []
        }
    
    @staticmethod
    def estimate_memory_usage(image_width: int, image_height: int, 
                            concurrent_requests: int = 1) -> Dict[str, Any]:
        """Estimate memory usage for an operation"""
        pixels = image_width * image_height
        
        # Determine image size category
        if pixels <= 640 * 480:
            base_memory = MemorySafetyConfig.MEMORY_ESTIMATES['small_image']
            size_category = 'small'
        elif pixels <= 1920 * 1080:
            base_memory = MemorySafetyConfig.MEMORY_ESTIMATES['medium_image']
            size_category = 'medium'
        else:
            base_memory = MemorySafetyConfig.MEMORY_ESTIMATES['large_image']
            size_category = 'large'
        
        # Add concurrent overhead
        concurrent_memory = (concurrent_requests - 1) * MemorySafetyConfig.MEMORY_ESTIMATES['concurrent_overhead']
        total_estimate = base_memory + concurrent_memory
        
        # Check if safe
        current = psutil.Process().memory_info().rss / (1024**2)
        projected = current + total_estimate
        is_safe = projected < MemorySafetyConfig.PROCESS_MEMORY_LIMIT_MB
        
        return {
            'size_category': size_category,
            'base_memory_mb': base_memory,
            'concurrent_memory_mb': concurrent_memory,
            'total_estimate_mb': total_estimate,
            'current_process_mb': current,
            'projected_process_mb': projected,
            'is_safe': is_safe
        }


class MemorySafetyMonitor:
    """Monitor memory usage and prevent crashes"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.rejected_requests = 0
        self.warning_logged = False
    
    async def check_before_analysis(self, image_width: int, image_height: int) -> bool:
        """Check if it's safe to process an image"""
        # Check current memory status
        status = MemorySafetyConfig.check_memory_safety()
        
        if not status['is_safe']:
            self.rejected_requests += 1
            logger.error(f"Memory safety check failed! Process: {status['process_mb']:.0f}MB, "
                        f"System available: {status['system_available_gb']:.1f}GB")
            return False
        
        # Estimate memory for this operation
        estimate = MemorySafetyConfig.estimate_memory_usage(
            image_width, image_height,
            self.analyzer._semaphore._value if hasattr(self.analyzer, '_semaphore') else 1
        )
        
        if not estimate['is_safe']:
            self.rejected_requests += 1
            logger.warning(f"Rejecting {estimate['size_category']} image analysis. "
                          f"Would exceed memory limit: {estimate['projected_process_mb']:.0f}MB")
            return False
        
        # Log warning if approaching limit
        if estimate['projected_process_mb'] > MemorySafetyConfig.MEMORY_WARNING_THRESHOLD_MB:
            if not self.warning_logged:
                logger.warning(f"Approaching memory limit: {estimate['projected_process_mb']:.0f}MB")
                self.warning_logged = True
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current memory safety status"""
        status = MemorySafetyConfig.check_memory_safety()
        status['rejected_requests'] = self.rejected_requests
        return status


def apply_memory_safety_config(analyzer):
    """Apply memory safety configuration to analyzer"""
    # Update configuration
    safe_config = MemorySafetyConfig.get_safe_config()
    for key, value in safe_config.items():
        if hasattr(analyzer.config, key):
            setattr(analyzer.config, key, value)
    
    # Add memory safety monitor
    analyzer.memory_monitor = MemorySafetyMonitor(analyzer)
    
    # Wrap the analyze methods with safety checks
    original_analyze = analyzer.analyze_screenshot
    
    async def safe_analyze_screenshot(image, prompt, **kwargs):
        # Check memory safety
        if hasattr(image, 'shape'):
            height, width = image.shape[:2]
        elif hasattr(image, 'size'):
            width, height = image.size
        else:
            # Default to medium size estimate
            width, height = 1920, 1080
        
        if not await analyzer.memory_monitor.check_before_analysis(width, height):
            raise MemoryError("Analysis rejected due to memory constraints. "
                            "System memory too low or process memory too high.")
        
        # Proceed with original analysis
        return await original_analyze(image, prompt, **kwargs)
    
    analyzer.analyze_screenshot = safe_analyze_screenshot
    
    logger.info("Memory safety configuration applied")
    logger.info(f"  Max concurrent: {analyzer.config.max_concurrent_requests}")
    logger.info(f"  Process limit: {MemorySafetyConfig.PROCESS_MEMORY_LIMIT_MB}MB")
    logger.info(f"  Min system RAM: {MemorySafetyConfig.MIN_SYSTEM_AVAILABLE_GB}GB")
    
    return analyzer


# Example usage
if __name__ == "__main__":
    print("Memory Safety Configuration for Claude Vision Analyzer")
    print("="*60)
    
    # Show safe configuration
    config = MemorySafetyConfig.get_safe_config()
    print("\nRecommended Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Check current memory status
    status = MemorySafetyConfig.check_memory_safety()
    print(f"\nCurrent Memory Status:")
    print(f"  Process: {status['process_mb']:.1f}/{status['process_limit_mb']} MB")
    print(f"  System Available: {status['system_available_gb']:.1f}/{status['system_min_gb']} GB")
    print(f"  Safe to operate: {'✅ YES' if status['is_safe'] else '⚠️ NO'}")
    
    # Example memory estimates
    print(f"\nMemory Usage Estimates:")
    examples = [
        ("Small image", 640, 480),
        ("HD image", 1920, 1080),
        ("4K image", 3840, 2160)
    ]
    
    for name, w, h in examples:
        estimate = MemorySafetyConfig.estimate_memory_usage(w, h)
        print(f"  {name} ({w}x{h}): ~{estimate['total_estimate_mb']}MB")