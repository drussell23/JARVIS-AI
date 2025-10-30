"""
Python wrapper for the Fast Capture C++ extension.

This module provides a clean, Pythonic interface to the high-performance capture engine
implemented in C++. It handles screen capture operations for macOS applications with
support for both synchronous and asynchronous operations, comprehensive configuration
options, and performance monitoring.

The module includes:
- FastCaptureEngine: Main wrapper class for the C++ capture engine
- CaptureConfig: Configuration dataclass for capture parameters
- Utility functions for creating filters and convenience operations
- Performance benchmarking and metrics collection
- Support for multiple output formats (JPEG, PNG, raw numpy arrays)

Example:
    >>> engine = FastCaptureEngine()
    >>> windows = engine.get_visible_windows()
    >>> result = engine.capture_window(windows[0]['window_id'])
    >>> if result['success']:
    ...     print(f"Captured {result['width']}x{result['height']} image")
"""

import os
import sys
import time
import asyncio
from typing import List, Dict, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import io

# Import the C++ extension
try:
    from . import fast_capture
except ImportError:
    # Try direct import if running as script
    import fast_capture

@dataclass
class CaptureConfig:
    """Configuration parameters for screen capture operations.
    
    This dataclass provides a Python-friendly interface to configure capture
    behavior, including output format, quality settings, filtering options,
    and performance parameters.
    
    Attributes:
        capture_cursor: Whether to include the mouse cursor in captures
        capture_shadow: Whether to include window shadows in captures
        capture_only_visible: Only capture windows that are currently visible
        output_format: Output format - "auto", "jpeg", "png", or "raw"
        jpeg_quality: JPEG compression quality (1-100)
        use_gpu_acceleration: Enable GPU acceleration if available
        parallel_capture: Enable parallel processing for multi-window captures
        max_threads: Maximum number of threads (0 = auto-detect)
        max_width: Maximum output width (0 = no limit)
        max_height: Maximum output height (0 = no limit)
        maintain_aspect_ratio: Preserve aspect ratio when resizing
        include_apps: List of app names to include (empty = all)
        exclude_apps: List of app names to exclude
        capture_metadata: Include window metadata in results
        custom_filter: Custom filter function for window selection
    """
    capture_cursor: bool = False
    capture_shadow: bool = False
    capture_only_visible: bool = True
    output_format: str = "auto"  # "auto", "jpeg", "png", "raw"
    jpeg_quality: int = 85
    use_gpu_acceleration: bool = True
    parallel_capture: bool = True
    max_threads: int = 0  # 0 = auto
    max_width: int = 0  # 0 = no limit
    max_height: int = 0  # 0 = no limit
    maintain_aspect_ratio: bool = True
    include_apps: List[str] = field(default_factory=list)
    exclude_apps: List[str] = field(default_factory=list)
    capture_metadata: bool = True
    custom_filter: Optional[Callable] = None
    
    def to_cpp_config(self) -> 'fast_capture.CaptureConfig':
        """Convert Python configuration to C++ configuration object.
        
        Returns:
            fast_capture.CaptureConfig: C++ configuration object with all
                parameters transferred from this Python configuration.
                
        Example:
            >>> config = CaptureConfig(jpeg_quality=90)
            >>> cpp_config = config.to_cpp_config()
        """
        cpp_config = fast_capture.CaptureConfig()
        cpp_config.capture_cursor = self.capture_cursor
        cpp_config.capture_shadow = self.capture_shadow
        cpp_config.capture_only_visible = self.capture_only_visible
        cpp_config.output_format = self.output_format
        cpp_config.jpeg_quality = self.jpeg_quality
        cpp_config.use_gpu_acceleration = self.use_gpu_acceleration
        cpp_config.parallel_capture = self.parallel_capture
        cpp_config.max_threads = self.max_threads
        cpp_config.max_width = self.max_width
        cpp_config.max_height = self.max_height
        cpp_config.maintain_aspect_ratio = self.maintain_aspect_ratio
        cpp_config.include_apps = self.include_apps
        cpp_config.exclude_apps = self.exclude_apps
        cpp_config.capture_metadata = self.capture_metadata
        
        if self.custom_filter:
            cpp_config.set_custom_filter(self.custom_filter)
            
        return cpp_config

class FastCaptureEngine:
    """Python wrapper for the C++ Fast Capture Engine.
    
    This class provides a high-level, Pythonic interface to the high-performance
    screen capture engine implemented in C++. It supports both single and multi-window
    captures, asynchronous operations, performance monitoring, and flexible
    configuration options.
    
    The engine handles window discovery, capture operations, format conversion,
    and provides comprehensive error handling and performance metrics.
    
    Attributes:
        _engine: The underlying C++ capture engine instance
        _executor: Thread pool executor for async operations
        _capture_callback: Optional callback for capture events
        _error_callback: Optional callback for error events
        
    Example:
        >>> engine = FastCaptureEngine()
        >>> result = engine.capture_frontmost_window()
        >>> if result['success']:
        ...     print(f"Captured {result['width']}x{result['height']} image")
    """
    
    def __init__(self, default_config: Optional[CaptureConfig] = None):
        """Initialize the capture engine.
        
        Args:
            default_config: Optional default configuration to use for all
                capture operations. If None, uses engine defaults.
                
        Example:
            >>> config = CaptureConfig(jpeg_quality=95, capture_cursor=True)
            >>> engine = FastCaptureEngine(default_config=config)
        """
        self._engine = fast_capture.FastCaptureEngine()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        if default_config:
            self.set_default_config(default_config)
            
        # Callbacks
        self._capture_callback = None
        self._error_callback = None
        
    def __del__(self):
        """Cleanup resources when the engine is destroyed."""
        self._executor.shutdown(wait=False)
    
    # ===== Single Window Capture =====
    
    def capture_window(self, window_id: int, 
                      config: Optional[CaptureConfig] = None) -> Dict[str, Any]:
        """Capture a single window by its unique ID.
        
        Args:
            window_id: Unique identifier for the target window
            config: Optional capture configuration. If None, uses default config.
            
        Returns:
            Dict containing capture results with keys:
                - success: bool indicating if capture succeeded
                - width: int width of captured image
                - height: int height of captured image
                - format: str output format used
                - capture_time_ms: float time taken for capture
                - image: numpy array (if raw format) or None
                - image_data: bytes (if compressed format) or None
                - error: str error message if success is False
                - metadata: dict with window information if enabled
                
        Example:
            >>> result = engine.capture_window(12345)
            >>> if result['success']:
            ...     print(f"Captured {result['format']} image: {result['width']}x{result['height']}")
        """
        cpp_config = (config or CaptureConfig()).to_cpp_config()
        result = self._engine.capture_window(window_id, cpp_config)
        return self._process_result(result)
    
    def capture_window_by_name(self, app_name: str, 
                              window_title: str = "",
                              config: Optional[CaptureConfig] = None) -> Dict[str, Any]:
        """Capture a window by application name and optional window title.
        
        Args:
            app_name: Name of the application (e.g., "Safari", "Chrome")
            window_title: Optional window title to match. If empty, captures
                the first window found for the app.
            config: Optional capture configuration
            
        Returns:
            Dict containing capture results (same format as capture_window)
            
        Raises:
            RuntimeError: If the specified window cannot be found
            
        Example:
            >>> result = engine.capture_window_by_name("Safari", "Google")
            >>> if result['success']:
            ...     print("Captured Safari window with 'Google' in title")
        """
        cpp_config = (config or CaptureConfig()).to_cpp_config()
        result = self._engine.capture_window_by_name(app_name, window_title, cpp_config)
        return self._process_result(result)
    
    def capture_frontmost_window(self, config: Optional[CaptureConfig] = None) -> Dict[str, Any]:
        """Capture the currently frontmost (active) window.
        
        Args:
            config: Optional capture configuration
            
        Returns:
            Dict containing capture results (same format as capture_window)
            
        Example:
            >>> result = engine.capture_frontmost_window()
            >>> if result['success']:
            ...     print(f"Captured active window: {result['metadata']['app_name']}")
        """
        cpp_config = (config or CaptureConfig()).to_cpp_config()
        result = self._engine.capture_frontmost_window(cpp_config)
        return self._process_result(result)
    
    # ===== Multi-Window Capture =====
    
    def capture_all_windows(self, config: Optional[CaptureConfig] = None) -> List[Dict[str, Any]]:
        """Capture all windows on the system.
        
        Args:
            config: Optional capture configuration applied to all windows
            
        Returns:
            List of capture result dictionaries, one per window
            
        Note:
            This operation can be resource-intensive for systems with many windows.
            Consider using capture_visible_windows() for better performance.
            
        Example:
            >>> results = engine.capture_all_windows()
            >>> successful = [r for r in results if r['success']]
            >>> print(f"Captured {len(successful)} out of {len(results)} windows")
        """
        cpp_config = (config or CaptureConfig()).to_cpp_config()
        results = self._engine.capture_all_windows(cpp_config)
        return [self._process_result(r) for r in results]
    
    def capture_visible_windows(self, config: Optional[CaptureConfig] = None) -> List[Dict[str, Any]]:
        """Capture only currently visible windows.
        
        Args:
            config: Optional capture configuration applied to all windows
            
        Returns:
            List of capture result dictionaries for visible windows only
            
        Example:
            >>> results = engine.capture_visible_windows()
            >>> for result in results:
            ...     if result['success']:
            ...         print(f"Captured {result['metadata']['app_name']}")
        """
        cpp_config = (config or CaptureConfig()).to_cpp_config()
        results = self._engine.capture_visible_windows(cpp_config)
        return [self._process_result(r) for r in results]
    
    def capture_windows_by_app(self, app_name: str,
                              config: Optional[CaptureConfig] = None) -> List[Dict[str, Any]]:
        """Capture all windows belonging to a specific application.
        
        Args:
            app_name: Name of the target application
            config: Optional capture configuration applied to all windows
            
        Returns:
            List of capture result dictionaries for the app's windows
            
        Example:
            >>> results = engine.capture_windows_by_app("Chrome")
            >>> print(f"Captured {len(results)} Chrome windows")
        """
        cpp_config = (config or CaptureConfig()).to_cpp_config()
        results = self._engine.capture_windows_by_app(app_name, cpp_config)
        return [self._process_result(r) for r in results]
    
    # ===== Async Capture Methods =====
    
    async def capture_window_async(self, window_id: int,
                                  config: Optional[CaptureConfig] = None) -> Dict[str, Any]:
        """Asynchronously capture a single window by ID.
        
        Args:
            window_id: Unique identifier for the target window
            config: Optional capture configuration
            
        Returns:
            Dict containing capture results (same format as capture_window)
            
        Example:
            >>> result = await engine.capture_window_async(12345)
            >>> if result['success']:
            ...     print("Async capture completed")
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self.capture_window, window_id, config
        )
    
    async def capture_all_windows_async(self, 
                                       config: Optional[CaptureConfig] = None) -> List[Dict[str, Any]]:
        """Asynchronously capture all windows.
        
        Args:
            config: Optional capture configuration applied to all windows
            
        Returns:
            List of capture result dictionaries
            
        Example:
            >>> results = await engine.capture_all_windows_async()
            >>> successful = [r for r in results if r['success']]
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self.capture_all_windows, config
        )
    
    # ===== Window Discovery =====
    
    def get_all_windows(self) -> List[Dict[str, Any]]:
        """Get information about all windows on the system.
        
        Returns:
            List of dictionaries containing window information with keys:
                - window_id: int unique window identifier
                - app_name: str application name
                - window_title: str window title
                - bundle_identifier: str app bundle ID (macOS)
                - x, y: int window position
                - width, height: int window dimensions
                - is_visible: bool visibility status
                - is_minimized: bool minimization status
                - is_fullscreen: bool fullscreen status
                - layer: int window layer/z-order
                - alpha: float window transparency (0.0-1.0)
                - metadata: dict additional window properties
                
        Example:
            >>> windows = engine.get_all_windows()
            >>> for window in windows:
            ...     print(f"{window['app_name']}: {window['window_title']}")
        """
        windows = self._engine.get_all_windows()
        return [self._window_info_to_dict(w) for w in windows]
    
    def get_visible_windows(self) -> List[Dict[str, Any]]:
        """Get information about currently visible windows only.
        
        Returns:
            List of window information dictionaries (same format as get_all_windows)
            
        Example:
            >>> visible = engine.get_visible_windows()
            >>> print(f"Found {len(visible)} visible windows")
        """
        windows = self._engine.get_visible_windows()
        return [self._window_info_to_dict(w) for w in windows]
    
    def get_windows_by_app(self, app_name: str) -> List[Dict[str, Any]]:
        """Get all windows belonging to a specific application.
        
        Args:
            app_name: Name of the target application
            
        Returns:
            List of window information dictionaries for the app's windows
            
        Example:
            >>> chrome_windows = engine.get_windows_by_app("Chrome")
            >>> for window in chrome_windows:
            ...     print(f"Chrome tab: {window['window_title']}")
        """
        windows = self._engine.get_windows_by_app(app_name)
        return [self._window_info_to_dict(w) for w in windows]
    
    def find_window(self, app_name: str, window_title: str = "") -> Optional[Dict[str, Any]]:
        """Find a specific window by app name and optional title.
        
        Args:
            app_name: Name of the target application
            window_title: Optional window title to match. If empty, returns
                the first window found for the app.
                
        Returns:
            Window information dictionary if found, None otherwise
            
        Example:
            >>> window = engine.find_window("Safari", "Google")
            >>> if window:
            ...     print(f"Found window: {window['window_id']}")
        """
        window = self._engine.find_window(app_name, window_title)
        if window is not None:
            return self._window_info_to_dict(window)
        return None
    
    def get_frontmost_window(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently frontmost (active) window.
        
        Returns:
            Window information dictionary if found, None if no active window
            
        Example:
            >>> active = engine.get_frontmost_window()
            >>> if active:
            ...     print(f"Active: {active['app_name']} - {active['window_title']}")
        """
        window = self._engine.get_frontmost_window()
        if window is not None:
            return self._window_info_to_dict(window)
        return None
    
    # ===== Performance Metrics =====
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for capture operations.
        
        Returns:
            Dictionary containing performance metrics:
                - avg_capture_time_ms: float average capture time
                - min_capture_time_ms: float minimum capture time
                - max_capture_time_ms: float maximum capture time
                - p95_capture_time_ms: float 95th percentile capture time
                - p99_capture_time_ms: float 99th percentile capture time
                - total_captures: int total number of capture attempts
                - successful_captures: int number of successful captures
                - failed_captures: int number of failed captures
                - bytes_processed: int total bytes of image data processed
                - peak_memory_usage: int peak memory usage in bytes
                - captures_per_app: dict capture counts by application
                - avg_time_per_app: dict average capture times by application
                
        Example:
            >>> metrics = engine.get_metrics()
            >>> print(f"Average capture time: {metrics['avg_capture_time_ms']:.2f}ms")
            >>> print(f"Success rate: {metrics['successful_captures']/metrics['total_captures']*100:.1f}%")
        """
        metrics = self._engine.get_metrics()
        return {
            'avg_capture_time_ms': metrics.avg_capture_time_ms,
            'min_capture_time_ms': metrics.min_capture_time_ms,
            'max_capture_time_ms': metrics.max_capture_time_ms,
            'p95_capture_time_ms': metrics.p95_capture_time_ms,
            'p99_capture_time_ms': metrics.p99_capture_time_ms,
            'total_captures': metrics.total_captures,
            'successful_captures': metrics.successful_captures,
            'failed_captures': metrics.failed_captures,
            'bytes_processed': metrics.bytes_processed,
            'peak_memory_usage': metrics.peak_memory_usage,
            'captures_per_app': dict(metrics.captures_per_app),
            'avg_time_per_app': dict(metrics.avg_time_per_app)
        }
    
    def reset_metrics(self):
        """Reset all performance metrics to zero.
        
        This is useful for benchmarking specific operations or clearing
        historical data.
        
        Example:
            >>> engine.reset_metrics()
            >>> # Perform operations to benchmark
            >>> metrics = engine.get_metrics()
        """
        self._engine.reset_metrics()
    
    def enable_metrics(self, enable: bool):
        """Enable or disable metrics collection.
        
        Args:
            enable: True to enable metrics collection, False to disable
            
        Note:
            Disabling metrics can provide a small performance improvement
            for high-frequency capture operations.
            
        Example:
            >>> engine.enable_metrics(False)  # Disable for max performance
            >>> # Perform high-frequency captures
            >>> engine.enable_metrics(True)   # Re-enable for monitoring
        """
        self._engine.enable_metrics(enable)
    
    # ===== Configuration =====
    
    def set_default_config(self, config: CaptureConfig):
        """Set the default capture configuration for all operations.
        
        Args:
            config: CaptureConfig object with desired default settings
            
        Example:
            >>> config = CaptureConfig(jpeg_quality=95, capture_cursor=True)
            >>> engine.set_default_config(config)
        """
        self._engine.set_default_config(config.to_cpp_config())
    
    def get_default_config(self) -> CaptureConfig:
        """Get the current default capture configuration.
        
        Returns:
            CaptureConfig object with current default settings
            
        Example:
            >>> config = engine.get_default_config()
            >>> print(f"Default JPEG quality: {config.jpeg_quality}")
        """
        cpp_config = self._engine.get_default_config()
        config = CaptureConfig()
        config.capture_cursor = cpp_config.capture_cursor
        config.capture_shadow = cpp_config.capture_shadow
        config.output_format = cpp_config.output_format
        config.jpeg_quality = cpp_config.jpeg_quality
        # ... copy other fields
        return config
    
    # ===== Callbacks =====
    
    def set_capture_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set a callback function to be called after each capture operation.
        
        Args:
            callback: Function that takes a capture result dictionary as argument
            
        Example:
            >>> def on_capture(result):
            ...     if result['success']:
            ...         print(f"Captured {result['width']}x{result['height']} image")
            >>> engine.set_capture_callback(on_capture)
        """
        self._capture_callback = callback
        self._engine.set_capture_callback(lambda result: callback(self._process_result(result)))
    
    def set_error_callback(self, callback: Callable[[str], None]):
        """Set a callback function to be called when capture errors occur.
        
        Args:
            callback: Function that takes an error message string as argument
            
        Example:
            >>> def on_error(error_msg):
            ...     print(f"Capture error: {error_msg}")
            >>> engine.set_error_callback(on_error)
        """
        self._error_callback = callback
        self._engine.set_error_callback(callback)
    
    # ===== Utility Methods =====
    
    def capture_to_pil(self, window_id: int, 
                      config: Optional[CaptureConfig] = None) -> Optional[Image.Image]:
        """Capture a window and return the result as a PIL Image object.
        
        Args:
            window_id: Unique identifier for the target window
            config: Optional capture configuration
            
        Returns:
            PIL Image object if capture successful, None otherwise
            
        Example:
            >>> image = engine.capture_to_pil(12345)
            >>> if image:
            ...     image.save("screenshot.png")
        """
        result = self.capture_window(window_id, config)
        if result['success']:
            if 'image' in result:  # Raw numpy array
                return Image.fromarray(result['image'])
            elif 'image_data' in result:  # Compressed data
                return Image.open(io.BytesIO(result['image_data']))
        return None
    
    def capture_to_numpy(self, window_id: int,
                        config: Optional[CaptureConfig] = None) -> Optional[np.ndarray]:
        """Capture a window and return the result as a numpy array.
        
        Args:
            window_id: Unique identifier for the target window
            config: Optional capture configuration (output_format will be set to "raw")
            
        Returns:
            numpy.ndarray with shape (height, width, channels) if successful, None otherwise
            
        Example:
            >>> array = engine.capture_to_numpy(12345)
            >>> if array is not None:
            ...     print(f"Captured array shape: {array.shape}")
        """
        # Force raw format for numpy
        if config is None:
            config = CaptureConfig()
        config.output_format = "raw"
        
        result = self.capture_window(window_id, config)
        if result['success'] and 'image' in result:
            return result['image']
        return None
    
    def benchmark(self, window_id: int, iterations: int = 100) -> Dict[str, float]:
        """Benchmark capture performance for a specific window.
        
        Args:
            window_id: Unique identifier for the target window
            iterations: Number of capture operations to perform for benchmarking
            
        Returns:
            Dictionary containing benchmark results:
                - avg_ms: float average capture time in milliseconds
                - min_ms: float minimum capture time in milliseconds
                - max_ms: float maximum capture time in milliseconds
                - std_ms: float standard deviation in milliseconds
                - p95_ms: float 95th percentile time in milliseconds
                - p99_ms: float 99th percentile time in milliseconds
                - fps: float potential frames per second based on average time
                
        Example:
            >>> bench = engine.benchmark(12345, iterations=50)
            >>> print(f"Average: {bench['avg_ms']:.2f}ms ({bench['fps']:.1f} FPS)")
            >>> print(f"P95: {bench['p95_ms']:.2f}ms")
        """
        times = []
        config = CaptureConfig(output_format="jpeg", jpeg_quality=85)
        
        # Warmup
        for _ in range(5):
            self.capture_window(window_id, config)
        
        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            result = self.capture_window(window_id, config)
            if result['success']:
                times.append(time.perf_counter() - start)
        
        if times:
            return {
                'avg_ms': np.mean(times) * 1000,
                'min_ms': np.min(times) * 1000,
                'max_ms': np.max(times) * 1000,
                'std_ms': np.std(times) * 1000,
                'p95_ms': np.percentile(times, 95) * 1000,
                'p99_ms': np.percentile(times, 99) * 1000,
                'fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0
            }
        return {}
    
    # ===== Private Methods =====
    
    def _process_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process and enhance capture result from C++ engine.
        
        Args:
            result: Raw result dictionary from C++ engine
            
        Returns:
            Enhanced result dictionary with additional convenience fields
        """
        # Convert image data if needed
        if 'image' in result and isinstance(result['image'], np.ndarray):
            # Already a numpy array from C++
            pass
        elif 'image_data' in result:
            # Convert bytes to more useful format if needed
            result['image_data'] = bytes(result['image_data'])
        
        # Add convenience fields
        if result['success']:
            result['capture_fps'] = 1000.0 / result['capture_time_ms'] if result['capture_time_ms'] > 0 else 0
        
        return result
    
    def _window_info_to_dict(self, window: 'fast_capture.WindowInfo') -> Dict[str, Any]:
        """Convert C++ WindowInfo object to Python dictionary.
        
        Args:
            window: C++ WindowInfo object
            
        Returns:
            Dictionary containing all window information fields
        """
        return {
            'window_id': window.window_id,
            'app_name': window.app_name,
            'window_title': window.window_title,
            'bundle_identifier': window.bundle_identifier,
            'x': window.x,
            'y': window.y,
            'width': window.width,
            'height': window.height,
            'is_visible': window.is_visible,
            'is_minimized': window.is_minimized,
            'is_fullscreen': window.is_fullscreen,
            'layer': window.layer,
            'alpha': window.alpha,
            'metadata': dict(window.metadata) if window.metadata else {}
        }

# ===== Convenience Functions =====

def create_size_filter(min_width: int, min_height: int) -> Callable:
    """Create a filter function for minimum window size requirements.
    
    Args:
        min_width: Minimum required window width in pixels
        min_height: Minimum required window height in pixels
        
    Returns:
        Callable filter function that accepts a window_info object and
        returns True if the window meets the size requirements
        
    Example:
        >>> size_filter = create_size_filter(800, 600)
        >>> config = CaptureConfig(custom_filter=size_filter)
        >>> results = engine.capture_all_windows(config)
    """
    def filter_func(window_info):
        return window_info.width >= min_width and window_info.height >= min_height
    return filter_func

def create_app_filter(apps: List[str]) -> Callable:
    """Create a filter function for specific applications.
    
    Args:
        apps: List of application names or bundle identifiers to include
        
    Returns:
        Callable filter function that accepts a window_info object and
        returns True if the window belongs to one of the specified apps
        
    Example:
        >>> app_filter = create_app_filter(["Safari", "Chrome", "Firefox"])
        >>> config = CaptureConfig(custom_filter=app_filter)
        >>> results = engine.capture_visible_windows(config)
    """
    def filter_func(window_info):
        for app in apps:
            if app in window_info.app_name or app in window_info.bundle_identifier:
                return True
        return False
    return filter_func

# ===== Example Usage =====

if __name__ == "__main__":
    """
    Example usage demonstrating the main features of the FastCaptureEngine.
    
    This script shows how to:
    - Initialize the engine
    - Discover windows
    - Capture windows
    - Benchmark performance
    """
    # Example usage
    engine = FastCaptureEngine()
    
    # List all windows
    print("All windows:")
    for window in engine.get_all_windows():
        print(f"  {window['app_name']} - {window['window_title']}")
    
    # Capture frontmost window
    result = engine.capture_frontmost_window()
    if result['success']:
        print(f"\nCaptured frontmost window in {result['capture_