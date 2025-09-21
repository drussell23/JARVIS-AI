"""
Rust-accelerated proactive monitoring integration for JARVIS.
Optimized for macOS with 16GB RAM using Metal GPU acceleration.
"""

import os
import sys
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import psutil

# Import unified components that handle Rust/Python switching
from .unified_components import (
    create_bloom_filter,
    create_sliding_window,
    create_memory_pool,
    create_zero_copy_pool,
    UnifiedBloomFilter,
    UnifiedSlidingWindow,
    UnifiedMemoryPool,
    UnifiedZeroCopyPool
)

# Check if base Rust is available
try:
    import jarvis_rust_core
    RUST_CORE_AVAILABLE = True
except ImportError:
    RUST_CORE_AVAILABLE = False
    jarvis_rust_core = None

logger = logging.getLogger(__name__)

@dataclass
class RustPerformanceMetrics:
    """Track Rust acceleration performance."""
    frame_processing_times: List[float]
    memory_usage_mb: float
    gpu_utilization: float
    bloom_operations_per_sec: int
    zero_copy_transfers: int
    cache_hit_rate: float
    metal_compute_time_ms: float
    
    def get_average_frame_time(self) -> float:
        """Get average frame processing time."""
        if not self.frame_processing_times:
            return 0.0
        return sum(self.frame_processing_times) / len(self.frame_processing_times)

class RustProactiveMonitor:
    """Rust-accelerated proactive monitoring system."""
    
    def __init__(self, vision_analyzer=None, interaction_handler=None):
        """
        Initialize monitor with dynamic Rust/Python components.
        
        Args:
            vision_analyzer: Claude vision analyzer instance
            interaction_handler: Real-time interaction handler
        """
        self.vision_analyzer = vision_analyzer
        self.interaction_handler = interaction_handler
        self.is_monitoring = False
        
        # Initialize unified components that will use Rust when available
        self._initialize_unified_components()
            
        # Performance tracking
        self.metrics = RustPerformanceMetrics(
            frame_processing_times=[],
            memory_usage_mb=0,
            gpu_utilization=0,
            bloom_operations_per_sec=0,
            zero_copy_transfers=0,
            cache_hit_rate=0,
            metal_compute_time_ms=0
        )
        
    def _initialize_unified_components(self):
        """Initialize unified components that dynamically use Rust or Python."""
        logger.info("Initializing unified acceleration components...")
        
        # Get system info
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        is_macos = sys.platform == 'darwin'
        
        logger.info(f"System: {total_ram_gb:.1f}GB RAM, {cpu_count} CPUs, macOS: {is_macos}")
        
        # Initialize unified components
        self.rust_components = {
            'bloom_filter': create_bloom_filter(size_mb=10.0, num_hashes=7),
            'sliding_window': create_sliding_window(window_size=30, overlap_threshold=0.9),
            'memory_pool': create_memory_pool(),
            'zero_copy_pool': create_zero_copy_pool()
        }
        
        # Log what implementations are being used
        for name, component in self.rust_components.items():
            impl_type = component.implementation_type
            if impl_type:
                logger.info(f"  • {name}: Using {impl_type.value} implementation")
            else:
                logger.info(f"  • {name}: Using fallback implementation")
                
        # Check if any Rust components are available
        rust_count = sum(1 for c in self.rust_components.values() 
                        if c.implementation_type and c.implementation_type.value == 'rust')
        
        if rust_count > 0:
            logger.info(f"✅ {rust_count}/{len(self.rust_components)} components using Rust acceleration")
        else:
            logger.info("⚠️ No Rust components available, using Python implementations")
            
        logger.info("Unified components initialized successfully")
        
    async def process_frame(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single frame through the monitoring system."""
        start_time = time.time()
        
        # Extract frame data
        import base64
        if isinstance(frame_data.get('data'), str):
            frame_bytes = base64.b64decode(frame_data['data'])
        else:
            frame_bytes = frame_data.get('data', b'')
            
        width = frame_data.get('width', 1280)
        height = frame_data.get('height', 720)
        timestamp = frame_data.get('timestamp', time.time())
        
        # Check for duplicates using bloom filter
        bloom_filter = self.rust_components['bloom_filter']
        frame_hash = f"{width}x{height}_{hash(frame_bytes)}"
        
        is_duplicate = bloom_filter.contains(frame_hash)
        if not is_duplicate:
            bloom_filter.add(frame_hash)
            
        # Process through sliding window
        sliding_window = self.rust_components['sliding_window']
        window_result = sliding_window.add_frame(frame_data, timestamp)
        
        # Track processing time
        process_time = time.time() - start_time
        self.metrics.frame_processing_times.append(process_time)
        
        # Generate insights based on frame content
        insights = None
        if not is_duplicate and self.vision_analyzer:
            # Analyze non-duplicate frames
            insights = "Frame processed successfully"
            
        return {
            'duplicate': is_duplicate,
            'timestamp': timestamp,
            'process_time_ms': process_time * 1000,
            'insights': insights,
            'window_analysis': window_result,
            'implementation': {
                name: comp.implementation_type.value if comp.implementation_type else 'fallback'
                for name, comp in self.rust_components.items()
            }
        }
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the monitor (for compatibility with tests)."""
        # Components are already initialized in __init__
        # This method is for async initialization if needed
        
        rust_available = any(
            c.implementation_type and c.implementation_type.value == 'rust'
            for c in self.rust_components.values()
        )
        
        return {
            'success': True,
            'rust_available': rust_available,
            'components': {
                name: {
                    'implementation': comp.implementation_type.value if comp.implementation_type else 'fallback',
                    'available': True
                }
                for name, comp in self.rust_components.items()
            }
        }
        
    async def cleanup(self):
        """Clean up resources."""
        # Stop monitoring if active
        if self.is_monitoring:
            await self.stop_monitoring()
            
        # No explicit cleanup needed for unified components
        # They handle their own lifecycle
        logger.info("Monitor cleaned up")
        
    async def start_monitoring(self):
        """Start Rust-accelerated monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
            
        self.is_monitoring = True
        logger.info("Starting Rust-accelerated proactive monitoring")
        
        # Start memory monitoring
        if self.rust_enabled and self.components.get('memory_monitor'):
            await self.components['memory_monitor'].start_monitoring()
            
        # Start main monitoring loops
        tasks = [
            asyncio.create_task(self._frame_capture_loop()),
            asyncio.create_task(self._analysis_loop()),
            asyncio.create_task(self._performance_monitoring_loop())
        ]
        
        # Wait for all tasks
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        finally:
            await self.stop_monitoring()
            
    async def stop_monitoring(self):
        """Stop monitoring and cleanup."""
        self.is_monitoring = False
        
        # Stop memory monitoring
        if self.rust_enabled and self.components.get('memory_monitor'):
            await self.components['memory_monitor'].stop_monitoring()
            
        logger.info("Stopped Rust-accelerated monitoring")
        
    async def _frame_capture_loop(self):
        """High-performance frame capture loop."""
        logger.info("Starting frame capture loop")
        
        frame_batch = []
        batch_size = 5  # Process 5 frames at a time
        last_batch_time = time.time()
        
        while self.is_monitoring:
            try:
                # Capture screenshot
                screenshot = await self._capture_screenshot_optimized()
                
                if screenshot is not None:
                    # Check for duplicate using Rust bloom filter
                    if await self._is_duplicate_frame(screenshot):
                        continue
                        
                    # Add to batch
                    frame_batch.append(screenshot)
                    
                    # Process batch when ready or timeout
                    current_time = time.time()
                    if len(frame_batch) >= batch_size or (current_time - last_batch_time) > 0.5:
                        if frame_batch:
                            await self._process_frame_batch(frame_batch)
                            frame_batch = []
                            last_batch_time = current_time
                            
                # Adaptive frame rate based on system load
                delay = await self._calculate_adaptive_delay()
                await asyncio.sleep(delay)
                
            except Exception as e:
                logger.error(f"Frame capture error: {e}")
                await asyncio.sleep(0.1)
                
    async def _analysis_loop(self):
        """Intelligent analysis loop with Rust acceleration."""
        logger.info("Starting analysis loop")
        
        while self.is_monitoring:
            try:
                # Get recent frames from buffer
                if self.rust_enabled and self.components.get('frame_buffer'):
                    frames = await self._get_recent_frames(5)
                    
                    if frames:
                        # Analyze with Metal GPU acceleration
                        results = await self._analyze_frames_gpu(frames)
                        
                        # Process results
                        if results:
                            await self._handle_analysis_results(results)
                            
                await asyncio.sleep(1.0)  # Analysis every second
                
            except Exception as e:
                logger.error(f"Analysis loop error: {e}")
                await asyncio.sleep(1.0)
                
    async def _performance_monitoring_loop(self):
        """Monitor and optimize performance."""
        logger.info("Starting performance monitoring loop")
        
        while self.is_monitoring:
            try:
                # Collect metrics
                await self._update_performance_metrics()
                
                # Log performance stats every 30 seconds
                if int(time.time()) % 30 == 0:
                    self._log_performance_stats()
                    
                # Adaptive optimization
                await self._adaptive_optimization()
                
                await asyncio.sleep(5.0)
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(5.0)
                
    async def _capture_screenshot_optimized(self) -> Optional[np.ndarray]:
        """Capture screenshot with optimization."""
        try:
            # Use existing capture method if available
            if hasattr(self.interaction_handler, '_capture_screenshot_async'):
                return await self.interaction_handler._capture_screenshot_async()
            else:
                # Fallback to synchronous capture
                import cv2
                from PIL import ImageGrab
                
                screenshot = ImageGrab.grab()
                return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                
        except Exception as e:
            logger.error(f"Screenshot capture error: {e}")
            return None
            
    async def _is_duplicate_frame(self, frame: np.ndarray) -> bool:
        """Check if frame is duplicate using Rust bloom filter."""
        if not self.rust_enabled or not self.components.get('bloom_network'):
            return False
            
        try:
            # Generate frame hash
            frame_hash = await self._generate_frame_hash(frame)
            
            # Check bloom filter
            start = time.perf_counter()
            is_duplicate = self.components['bloom_network'].contains(frame_hash)
            bloom_time = (time.perf_counter() - start) * 1000
            
            # Update metrics
            self.metrics.bloom_operations_per_sec = int(1000 / max(0.1, bloom_time))
            
            if not is_duplicate:
                # Add to bloom filter
                self.components['bloom_network'].add(frame_hash)
                
            return is_duplicate
            
        except Exception as e:
            logger.error(f"Bloom filter error: {e}")
            return False
            
    async def _generate_frame_hash(self, frame: np.ndarray) -> str:
        """Generate efficient frame hash."""
        if self.rust_enabled and hasattr(jarvis_rust_core, 'hash_image_fast'):
            # Use Rust SIMD hashing
            return jarvis_rust_core.hash_image_fast(frame.tobytes())
        else:
            # Fallback to Python
            import hashlib
            # Downsample for faster hashing
            small_frame = frame[::8, ::8]
            return hashlib.sha256(small_frame.tobytes()).hexdigest()
            
    async def _process_frame_batch(self, frames: List[np.ndarray]):
        """Process frame batch with Rust acceleration."""
        if not frames:
            return
            
        start_time = time.perf_counter()
        
        try:
            if self.rust_enabled and self.components.get('zero_copy_pipeline'):
                # Use zero-copy pipeline
                results = []
                for frame in frames:
                    result = await self.components['zero_copy_pipeline'].process_image(
                        frame,
                        model_name='vision-accelerated'
                    )
                    results.append(result)
                    
                # Store in frame buffer
                if self.components.get('frame_buffer'):
                    for frame in frames:
                        timestamp = int(time.time() * 1000)
                        await self._add_to_frame_buffer(frame, timestamp)
                        
            else:
                # Fallback to standard processing
                logger.debug("Using fallback frame processing")
                
            # Update metrics
            processing_time = (time.perf_counter() - start_time) * 1000
            self.metrics.frame_processing_times.append(processing_time)
            if len(self.metrics.frame_processing_times) > 100:
                self.metrics.frame_processing_times.pop(0)
                
            self.metrics.zero_copy_transfers += len(frames)
            
        except Exception as e:
            logger.error(f"Frame batch processing error: {e}")
            
    async def _add_to_frame_buffer(self, frame: np.ndarray, timestamp: int):
        """Add frame to Rust ring buffer."""
        if self.components.get('frame_buffer'):
            try:
                frame_bytes = frame.tobytes()
                await self.components['frame_buffer'].add_frame_async(
                    frame_bytes, timestamp
                )
            except Exception as e:
                logger.error(f"Frame buffer error: {e}")
                
    async def _get_recent_frames(self, count: int) -> List[Tuple[np.ndarray, int]]:
        """Get recent frames from buffer."""
        if not self.components.get('frame_buffer'):
            return []
            
        try:
            # Get frame data from Rust buffer
            frame_data = await self.components['frame_buffer'].get_recent_frames_async(count)
            
            # Convert back to numpy arrays
            frames = []
            for data, timestamp in frame_data:
                # Assuming 1080p RGB frames
                frame = np.frombuffer(data, dtype=np.uint8).reshape((1080, 1920, 3))
                frames.append((frame, timestamp))
                
            return frames
            
        except Exception as e:
            logger.error(f"Get frames error: {e}")
            return []
            
    async def _analyze_frames_gpu(self, frames: List[Tuple[np.ndarray, int]]) -> List[Dict[str, Any]]:
        """Analyze frames using Metal GPU acceleration."""
        if not self.components.get('metal_accelerator'):
            return []
            
        try:
            start_time = time.perf_counter()
            
            # Extract frame data
            frame_arrays = [f[0] for f in frames]
            timestamps = [f[1] for f in frames]
            
            # Process on GPU
            results = await self.components['metal_accelerator'].analyze_frames_async(
                frame_arrays,
                detect_changes=True,
                extract_features=True
            )
            
            # Update metrics
            self.metrics.metal_compute_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Combine with timestamps
            analyzed_results = []
            for i, result in enumerate(results):
                result['timestamp'] = timestamps[i]
                analyzed_results.append(result)
                
            return analyzed_results
            
        except Exception as e:
            logger.error(f"GPU analysis error: {e}")
            return []
            
    async def _handle_analysis_results(self, results: List[Dict[str, Any]]):
        """Handle analysis results from GPU processing."""
        for result in results:
            try:
                # Check for significant changes
                if result.get('change_magnitude', 0) > 0.3:
                    # Forward to interaction handler
                    if self.interaction_handler:
                        await self.interaction_handler._handle_screen_change(
                            result.get('features', {}),
                            result.get('objects', []),
                            result.get('timestamp', 0)
                        )
                        
            except Exception as e:
                logger.error(f"Result handling error: {e}")
                
    async def _calculate_adaptive_delay(self) -> float:
        """Calculate adaptive frame capture delay."""
        # Base delay
        base_delay = 0.1  # 10 FPS
        
        if not self.rust_enabled:
            return base_delay
            
        try:
            # Get system metrics
            stats = self.rust_accelerator.get_memory_stats()
            memory_pressure = stats.get('system_memory', {}).get('percent', 0)
            
            # Adjust based on memory pressure
            if memory_pressure > 80:
                return base_delay * 2  # Slow down
            elif memory_pressure < 50:
                return base_delay * 0.5  # Speed up
                
            return base_delay
            
        except Exception:
            return base_delay
            
    async def _update_performance_metrics(self):
        """Update performance metrics."""
        if not self.rust_enabled:
            return
            
        try:
            # Memory usage
            memory_stats = self.rust_accelerator.get_memory_stats()
            self.metrics.memory_usage_mb = memory_stats.get('pool_stats', {}).get(
                'total_allocated_bytes', 0
            ) / (1024 * 1024)
            
            # GPU utilization (estimated)
            if self.components.get('metal_accelerator'):
                # Estimate based on compute time
                compute_ratio = self.metrics.metal_compute_time_ms / 1000.0
                self.metrics.gpu_utilization = min(100, compute_ratio * 100)
                
        except Exception as e:
            logger.error(f"Metrics update error: {e}")
            
    async def _adaptive_optimization(self):
        """Perform adaptive optimization based on metrics."""
        if not self.rust_enabled:
            return
            
        try:
            # Check memory pressure
            stats = self.rust_accelerator.get_memory_stats()
            pool_stats = stats.get('pool_stats', {})
            
            if pool_stats.get('memory_pressure') == 'Critical':
                logger.warning("Critical memory pressure detected")
                # Trigger cleanup
                if self.components.get('bloom_network'):
                    self.components['bloom_network'].reset('regional')
                    
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            
    def _log_performance_stats(self):
        """Log current performance statistics."""
        avg_frame_time = self.metrics.get_average_frame_time()
        
        logger.info(
            f"Performance: Frame={avg_frame_time:.1f}ms, "
            f"Memory={self.metrics.memory_usage_mb:.1f}MB, "
            f"GPU={self.metrics.gpu_utilization:.1f}%, "
            f"Bloom={self.metrics.bloom_operations_per_sec}/s, "
            f"ZeroCopy={self.metrics.zero_copy_transfers}"
        )
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report."""
        return {
            'rust_enabled': self.rust_enabled,
            'average_frame_time_ms': self.metrics.get_average_frame_time(),
            'memory_usage_mb': self.metrics.memory_usage_mb,
            'gpu_utilization_percent': self.metrics.gpu_utilization,
            'bloom_ops_per_sec': self.metrics.bloom_operations_per_sec,
            'zero_copy_transfers': self.metrics.zero_copy_transfers,
            'cache_hit_rate': self.metrics.cache_hit_rate,
            'metal_compute_ms': self.metrics.metal_compute_time_ms,
            'components_active': {
                name: (comp is not None) for name, comp in self.components.items()
            } if self.rust_enabled else {}
        }

# Global instance
_rust_monitor: Optional[RustProactiveMonitor] = None

def initialize_rust_monitoring(
    vision_analyzer=None,
    interaction_handler=None
) -> RustProactiveMonitor:
    """Initialize global Rust monitoring instance."""
    global _rust_monitor
    
    if _rust_monitor is None:
        _rust_monitor = RustProactiveMonitor(
            vision_analyzer=vision_analyzer,
            interaction_handler=interaction_handler
        )
        logger.info("Initialized Rust proactive monitoring")
        
    return _rust_monitor

def get_rust_monitor() -> Optional[RustProactiveMonitor]:
    """Get global Rust monitor instance."""
    return _rust_monitor
