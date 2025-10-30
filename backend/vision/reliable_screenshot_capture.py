#!/usr/bin/env python3
"""
Reliable Screenshot Capture System

This module implements a multi-method screenshot capture system with intelligent
fallback mechanisms for macOS. It provides robust screenshot capture across
different desktop spaces using various capture methods including Quartz, AppKit,
and command-line tools.

The system automatically falls back through different capture methods if one fails,
ensuring reliable screenshot capture even in challenging scenarios like permission
issues, off-screen windows, or system resource constraints.

Example:
    >>> capture = ReliableScreenshotCapture()
    >>> result = capture.capture_space(1)
    >>> if result.success:
    ...     result.image.save('screenshot.png')
"""

import os
import io
import time
import subprocess
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from PIL import Image
import numpy as np
import Quartz
from Quartz import (
    CGWindowListCopyWindowInfo,
    CGWindowListCreateImage,
    CGRectNull,
    CGRectMake,
    kCGWindowListOptionOnScreenOnly,
    kCGWindowImageDefault,
    kCGWindowImageBoundsIgnoreFraming,
    kCGWindowImageNominalResolution,
    kCGNullWindowID
)
import AppKit
from AppKit import NSScreen, NSBitmapImageRep, NSImage

logger = logging.getLogger(__name__)

# Import window capture manager for robust edge case handling
try:
    import sys
    from pathlib import Path as PathLib
    backend_path = PathLib(__file__).parent.parent
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))
    from context_intelligence.managers.window_capture_manager import get_window_capture_manager
    WINDOW_CAPTURE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Window capture manager not available: {e}")
    get_window_capture_manager = None
    WINDOW_CAPTURE_AVAILABLE = False

# Import Error Handling Matrix for graceful degradation
try:
    from context_intelligence.managers.error_handling_matrix import (
        get_error_handling_matrix,
        initialize_error_handling_matrix,
        FallbackChain,
        ErrorMessageGenerator
    )
    ERROR_MATRIX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Error Handling Matrix not available: {e}")
    get_error_handling_matrix = None
    initialize_error_handling_matrix = None
    ERROR_MATRIX_AVAILABLE = False

@dataclass
class ScreenshotResult:
    """Result of a screenshot capture attempt.
    
    Attributes:
        success: Whether the capture was successful
        image: The captured PIL Image, None if failed
        method: Name of the capture method used
        space_id: ID of the desktop space captured
        error: Error message if capture failed
        timestamp: When the capture was performed
        metadata: Additional information about the capture
    """
    success: bool
    image: Optional[Image.Image]
    method: str
    space_id: Optional[int]
    error: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any]

class ReliableScreenshotCapture:
    """Multi-method screenshot capture with intelligent fallback.
    
    This class provides robust screenshot capture functionality by implementing
    multiple capture methods and automatically falling back to alternative methods
    if the primary method fails. It supports capturing from different desktop spaces
    and handles various edge cases like permission issues and system constraints.
    
    Attributes:
        methods: List of available capture methods in priority order
        error_matrix: Error handling matrix for graceful degradation (optional)
        cache: Cache for recent captures to avoid redundant operations
        cache_ttl: Time-to-live for cached results in seconds
    
    Example:
        >>> capture = ReliableScreenshotCapture()
        >>> results = capture.capture_all_spaces()
        >>> for space_id, result in results.items():
        ...     if result.success:
        ...         print(f"Captured space {space_id} using {result.method}")
    """

    def __init__(self):
        """Initialize the screenshot capture system.
        
        Sets up available capture methods in priority order and initializes
        the error handling matrix if available. The window capture manager
        is prioritized if available for better edge case handling.
        """
        # Build methods list with window_capture_manager as first choice (if available)
        self.methods = []

        if WINDOW_CAPTURE_AVAILABLE:
            self.methods.append(('window_capture_manager', self._capture_with_window_manager))

        self.methods.extend([
            ('quartz_composite', self._capture_quartz_composite),
            ('quartz_windows', self._capture_quartz_windows),
            ('appkit_screen', self._capture_appkit_screen),
            ('screencapture_cli', self._capture_screencapture_cli),
            ('window_server', self._capture_window_server)
        ])

        # Initialize Error Handling Matrix for graceful degradation
        self.error_matrix = None
        if ERROR_MATRIX_AVAILABLE:
            try:
                # Try to get existing instance
                self.error_matrix = get_error_handling_matrix() # If not available, initialize with default settings
                if not self.error_matrix: # No existing instance
                    # Initialize with default settings
                    self.error_matrix = initialize_error_handling_matrix(
                        default_timeout=10.0,
                        aggregation_strategy="first_success", # Aggregation strategy for partial results
                        recovery_strategy="continue" # Recovery strategy for errors
                    )
                logger.info("✅ Error Handling Matrix available for screenshot capture")
            except Exception as e:
                logger.warning(f"Failed to initialize Error Handling Matrix: {e}")

        self._init_capture_cache()
        logger.info(f"Reliable Screenshot Capture initialized with {len(self.methods)} methods")

    def _init_capture_cache(self) -> None:
        """Initialize cache for recent captures.
        
        Sets up an in-memory cache to store recent screenshot results
        to avoid redundant capture operations within a short time window.
        """
        self.cache = {}
        self.cache_ttl = 2  # seconds

    def capture_all_spaces(self) -> Dict[int, ScreenshotResult]:
        """Capture screenshots from all available desktop spaces.
        
        Attempts to detect all available desktop spaces and capture
        screenshots from each one using the best available method.
        
        Returns:
            Dict mapping space IDs to their corresponding ScreenshotResult.
            Each result contains the capture status, image data, and metadata.
        
        Example:
            >>> capture = ReliableScreenshotCapture()
            >>> results = capture.capture_all_spaces()
            >>> successful_captures = {k: v for k, v in results.items() if v.success}
            >>> print(f"Captured {len(successful_captures)} spaces successfully")
        """
        results = {}

        # Try to detect all spaces
        spaces = self._detect_available_spaces()

        for space_id in spaces:
            result = self.capture_space(space_id)
            results[space_id] = result

        return results

    def capture_space(self, space_id: int) -> ScreenshotResult:
        """Capture a screenshot from a specific desktop space.
        
        Attempts to capture a screenshot from the specified desktop space
        using the best available method. Falls back through multiple methods
        if the primary method fails.
        
        Args:
            space_id: The ID of the desktop space to capture
            
        Returns:
            ScreenshotResult containing the capture status, image data,
            method used, and any error information.
            
        Example:
            >>> capture = ReliableScreenshotCapture()
            >>> result = capture.capture_space(1)
            >>> if result.success:
            ...     result.image.save(f'space_{result.space_id}.png')
            ... else:
            ...     print(f"Capture failed: {result.error}")
        """
        # Check cache first
        cached = self._get_cached(space_id)
        if cached:
            return cached

        # Try each capture method in order
        for method_name, method_func in self.methods:
            try:
                result = method_func(space_id)
                if result.success:
                    self._cache_result(space_id, result)
                    logger.info(f"Successfully captured space {space_id} using {method_name}")
                    return result
            except Exception as e:
                logger.warning(f"Method {method_name} failed for space {space_id}: {e}")
                continue

        # All methods failed
        return ScreenshotResult(
            success=False,
            image=None,
            method='none',
            space_id=space_id,
            error="All capture methods failed",
            timestamp=datetime.now(),
            metadata={}
        )

    async def capture_space_with_matrix(self, space_id: int) -> ScreenshotResult:
        """Capture a screenshot using Error Handling Matrix for graceful degradation.

        This async version uses the Error Handling Matrix for:
        - Priority-based fallback execution
        - Partial result aggregation
        - User-friendly error messages
        
        Args:
            space_id: The ID of the desktop space to capture
            
        Returns:
            ScreenshotResult with enhanced error handling and reporting.
            If the Error Handling Matrix is not available, falls back to
            the standard synchronous capture method.
            
        Example:
            >>> import asyncio
            >>> capture = ReliableScreenshotCapture()
            >>> result = await capture.capture_space_with_matrix(1)
            >>> if result.success:
            ...     print(f"Captured using method: {result.method}")
        """
        # Check cache first
        cached = self._get_cached(space_id)
        if cached:
            logger.info(f"[MATRIX-CAPTURE] Using cached result for space {space_id}")
            return cached

        # Use Error Handling Matrix if available
        if self.error_matrix:
            logger.info(f"[MATRIX-CAPTURE] Using Error Handling Matrix for space {space_id}")

            # Build fallback chain
            chain = FallbackChain(f"capture_space_{space_id}") # Use space_id as fallback chain name

            # Add methods in priority order
            for i, (method_name, method_func) in enumerate(self.methods):
                # Wrap sync method in async
                async def async_wrapper(func=method_func, sid=space_id):
                    return func(sid) # Call the sync method

                if i == 0 and WINDOW_CAPTURE_AVAILABLE: # Highest priority (if available)
                    chain.add_primary(async_wrapper, name=method_name, timeout=5.0) # Capture with window_capture_manager first if available 
                elif i == 1: # Second highest priority
                    chain.add_fallback(async_wrapper, name=method_name, timeout=8.0) # Fallback to other methods next if primary fails 
                elif i == len(self.methods) - 1: # Lowest priority (last resort)
                    chain.add_last_resort(async_wrapper, name=method_name, timeout=10.0) # Last resort method with longer timeout 
                else: 
                    # All other methods
                    chain.add_secondary(async_wrapper, name=method_name, timeout=7.0) 

            # Execute chain
            report = await self.error_matrix.execute_chain(chain, stop_on_success=True)

            # Convert ExecutionReport to ScreenshotResult
            if report.success and report.final_result:
                # Cache and return the result
                self._cache_result(space_id, report.final_result)
                logger.info(f"[MATRIX-CAPTURE] ✅ Captured space {space_id} - {report.message}")
                return report.final_result
            else:
                # Generate user-friendly error message
                error_msg = ErrorMessageGenerator.generate_message(
                    report,
                    include_technical=True,
                    include_suggestions=True
                )

                logger.error(f"[MATRIX-CAPTURE] ❌ Failed to capture space {space_id}:\n{error_msg}")

                return ScreenshotResult(
                    success=False,
                    image=None,
                    method='matrix_fallback',
                    space_id=space_id,
                    error=error_msg,
                    timestamp=datetime.now(),
                    metadata={
                        "execution_report": report,
                        "methods_attempted": len(report.methods_attempted),
                        "total_duration": report.total_duration
                    }
                )

        # Fallback to regular capture if matrix not available
        logger.warning(f"[MATRIX-CAPTURE] Error Handling Matrix not available, using standard capture")
        return self.capture_space(space_id) # Fallback to regular capture

    def _capture_with_window_manager(self, space_id: int) -> ScreenshotResult:
        """Use WindowCaptureManager for robust window capture with edge case handling.

        This method attempts to capture windows from the specified space using
        the WindowCaptureManager which handles permissions, off-screen windows,
        4K/5K resizing, transparency, and fallback windows automatically.
        
        Args:
            space_id: The ID of the desktop space to capture
            
        Returns:
            ScreenshotResult with the captured image and metadata
            
        Raises:
            Exception: If window manager capture fails or no windows found
        """
        try:
            import asyncio

            # Get window manager
            window_manager = get_window_capture_manager()

            # Find windows in the target space
            try:
                from .multi_space_window_detector import MultiSpaceWindowDetector
                detector = MultiSpaceWindowDetector()
                window_data = detector.get_all_windows_across_spaces()

                # Find windows in target space
                target_windows = []
                for window in window_data.get("windows", []):
                    if hasattr(window, "space_id"):
                        if window.space_id == space_id:
                            target_windows.append(window)
                    elif isinstance(window, dict) and window.get("space") == space_id:
                        target_windows.append(window)

                if not target_windows:
                    raise Exception(f"No windows found in space {space_id}")

                # Try to capture the first non-minimized window
                for window in target_windows:
                    window_id = None
                    if hasattr(window, "window_id"):
                        window_id = window.window_id
                    elif isinstance(window, dict):
                        window_id = window.get("id")

                    if window_id:
                        # Create async event loop if needed
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)

                        # Capture using window manager
                        capture_result = loop.run_until_complete(
                            window_manager.capture_window(
                                window_id=window_id,
                                space_id=space_id,
                                use_fallback=True
                            )
                        )

                        if capture_result.success:
                            # Load image
                            image = Image.open(capture_result.image_path)

                            return ScreenshotResult(
                                success=True,
                                image=image,
                                method='window_capture_manager',
                                space_id=space_id,
                                error=None,
                                timestamp=datetime.now(),
                                metadata={
                                    'window_id': window_id,
                                    'capture_status': capture_result.status.value,
                                    'original_size': capture_result.original_size,
                                    'resized_size': capture_result.resized_size,
                                    'fallback_used': capture_result.fallback_window_id is not None
                                }
                            )

            except Exception as e:
                logger.debug(f"Window detection failed: {e}, trying next method")
                raise Exception(f"Window manager capture failed: {e}")

        except Exception as e:
            raise Exception(f"Window manager capture failed: {e}")

    def _capture_quartz_composite(self, space_id: int) -> ScreenshotResult:
        """Use Quartz to capture composite window image.
        
        Creates a composite image of all windows in the specified space
        using Quartz's CGWindowListCreateImage function.
        
        Args:
            space_id: The ID of the desktop space to capture
            
        Returns:
            ScreenshotResult with the composite image
            
        Raises:
            Exception: If no windows found or Quartz capture fails
        """
        try:
            # Get windows for the space
            windows = self._get_windows_for_space(space_id)

            if not windows:
                raise Exception(f"No windows found in space {space_id}")

            # Create composite image
            window_ids = [w['kCGWindowID'] for w in windows if 'kCGWindowID' in w]

            # Calculate bounding rect for all windows
            bounds = self._calculate_composite_bounds(windows)

            # Create composite image
            rect = CGRectMake(
                bounds['x'],
                bounds['y'],
                bounds['width'],
                bounds['height']
            )

            cg_image = CGWindowListCreateImage(
                rect,
                kCGWindowListOptionOnScreenOnly,
                kCGNullWindowID,
                kCGWindowImageDefault
            )

            if cg_image:
                # Convert to PIL Image
                image = self._cgimage_to_pil(cg_image)

                return ScreenshotResult(
                    success=True,
                    image=image,
                    method='quartz_composite',
                    space_id=space_id,
                    error=None,
                    timestamp=datetime.now(),
                    metadata={'window_count': len(windows)}
                )

        except Exception as e:
            raise Exception(f"Quartz composite capture failed: {e}")

    def _capture_quartz_windows(self, space_id: int) -> ScreenshotResult:
        """Capture individual windows and composite them.
        
        Captures each window individually using Quartz and then composites
        them into a single image based on their screen positions.
        
        Args:
            space_id: The ID of the desktop space to capture
            
        Returns:
            ScreenshotResult with the composited image
            
        Raises:
            Exception: If no windows found or individual captures fail
        """
        try:
            windows = self._get_windows_for_space(space_id)

            if not windows:
                raise Exception(f"No windows in space {space_id}")

            # Capture each window
            window_images = []
            for window in windows[:10]:  # Limit to prevent memory issues
                if 'kCGWindowID' in window:
                    cg_image = CGWindowListCreateImage(
                        CGRectNull,
                        kCGWindowListOptionOnScreenOnly,
                        window['kCGWindowID'],
                        kCGWindowImageDefault
                    )

                    if cg_image:
                        img = self._cgimage_to_pil(cg_image)
                        bounds = window.get('kCGWindowBounds', {})
                        window_images.append((img, bounds))

            # Composite the images
            if window_images:
                composite = self._composite_window_images(window_images)

                return ScreenshotResult(
                    success=True,
                    image=composite,
                    method='quartz_windows',
                    space_id=space_id,
                    error=None,
                    timestamp=datetime.now(),
                    metadata={'window_count': len(window_images)}
                )

        except Exception as e:
            raise Exception(f"Quartz windows capture failed: {e}")

    def _capture_appkit_screen(self, space_id: int) -> ScreenshotResult:
        """Use AppKit to capture the main screen.
        
        Captures the entire main screen using AppKit's NSScreen functionality
        and Quartz's window list creation.
        
        Args:
            space_id: The ID of the desktop space to capture
            
        Returns:
            ScreenshotResult with the screen capture
            
        Raises:
            Exception: If no main screen found or AppKit capture fails
        """
        try:
            # Get main screen
            screen = NSScreen.mainScreen()
            if not screen:
                raise Exception("No main screen found")

            # Get screen rect
            rect = screen.frame()

            # Capture screen
            window_list = kCGWindowListOptionOnScreenOnly
            image_rect = CGRectMake(0, 0, rect.size.width, rect.size.height)

            cg_image = CGWindowListCreateImage(
                image_rect,
                window_list,
                kCGNullWindowID,
                kCGWindowImageDefault
            )

            if cg_image:
                image = self._cgimage_to_pil(cg_image)

                return ScreenshotResult(
                    success=True,
                    image=image,
                    method='appkit_screen',
                    space_id=space_id,
                    error=None,
                    timestamp=datetime.now(),
                    metadata={'screen_size': (rect.size.width, rect.size.height)}
                )

        except Exception as e:
            raise Exception(f"AppKit screen capture failed: {e}")

    def _capture_screencapture_cli(self, space_id: int) -> ScreenshotResult:
        """Use screencapture command line tool.
        
        Uses macOS's built-in screencapture command-line utility to
        capture a screenshot. This method works even when GUI APIs fail.
        
        Args:
            space_id: The ID of the desktop space to capture
            
        Returns:
            ScreenshotResult with the captured image
            
        Raises:
            Exception: If screencapture command fails or times out
        """
        try:
            # Create temporary file
            temp_file = f"/tmp/screenshot_{space_id}_{int(time.time())}.png"

            # Run screencapture
            result = subprocess.run(
                ['screencapture', '-x', '-C', temp_file],
                capture_output=True,
                timeout=5
            )

            if result.returncode == 0 and os.path.exists(temp_file):
                # Load image
                image = Image.open(temp_file)

                # Clean up
                os.remove(temp_file)

                return ScreenshotResult(
                    success=True,
                    image=image,
                    method='screencapture_cli',
                    space_id=space_id,
                    error=None,
                    timestamp=datetime.now(),
                    metadata={}
                )

        except Exception as e:
            raise Exception(f"Screencapture CLI failed: {e}")

    def _capture_window_server(self, space_id: int) -> ScreenshotResult:
        """Direct window server capture using AppleScript.
        
        Uses AppleScript to switch to the target space and capture
        a screenshot. This method requires accessibility permissions
        and may be slower due to space switching.
        
        Args:
            space_id: The ID of the desktop space to capture
            
        Returns:
            ScreenshotResult with the captured image
            
        Raises:
            Exception: If AppleScript execution fails or permissions denied
        """
        try:
            # Use AppleScript to switch space and capture
            script = f"""
            tell application "System Events"
                key code 18 using control down
                delay 0.5
                do shell script "screencapture -x -C /tmp/space_{space_id}.png"
            end tell
            """

            subprocess.run(['osascript', '-e', script], timeout=3)

            temp_file = f"/tmp/space_{space_id}.png"
            if os.path.exists(temp_file):
                image = Image.open(temp_file)
                os.remove(temp_file)

                return ScreenshotResult(
                    success=True,
                    image=image,
                    method='window_server',
                    space_id=space_id,
                    error=None,
                    timestamp=datetime.now(),
                    metadata={}
                )

        except Exception as e:
            raise Exception(f"Window server capture failed: {e}")

    def _detect_available_spaces(self) -> List[int]:
        """Detect available desktop spaces.
        
        Attempts to detect the number of available desktop spaces.
        Currently uses a simple heuristic but can be enhanced to
        integrate with MacOSSpaceDetector for more accurate detection.
        
        Returns:
            List of space IDs that are potentially available.
            
        Note:
            This is a simplified implementation. In production,
            integrate with MacOSSpaceDetector for accurate space detection.
        """
        # Try to get space count from window positions
        windows = CGWindowListCopyWindowInfo(
            kCGWindowListOptionOnScreenOnly,
            kCGNullWindowID
        )

        # Simple heuristic: assume 4 spaces by default
        # In production, integrate with MacOSSpaceDetector
        return [1, 2, 3, 4]

    def _get_windows_for_space(self, space_id: int) -> List[Dict]:
        """Get windows for a specific desktop space.
        
        Retrieves all windows that belong to the specified desktop space.
        This is a simplified implementation that filters normal windows.
        
        Args:
            space_id: The ID of the desktop space
            
        Returns:
            List of window dictionaries containing window information.
            
        Note:
            This is a simplified implementation. For accurate space-to-window
            mapping, integrate with MacOSSpaceDetector.
        """
        all_windows = CGWindowListCopyWindowInfo(
            kCGWindowListOptionOnScreenOnly,
            kCGNullWindowID
        )

        # Filter by space (simplified - integrate with MacOSSpaceDetector)
        space_windows = []
        for window in all_windows:
            if window.get('kCGWindowLayer', 0) == 0:  # Normal windows
                space_windows.append(window)

        return space_windows

    def _calculate_composite_bounds(self, windows: List[Dict]) -> Dict[str, float]:
        """Calculate bounding box for all windows.
        
        Computes the minimum bounding rectangle that contains all
        the specified windows based on their screen positions.
        
        Args:
            windows: List of window dictionaries with bounds information
            
        Returns:
            Dictionary with 'x', 'y', 'width', 'height' keys defining
            the composite bounding rectangle.
            
        Example:
            >>> windows = [{'kCGWindowBounds': {'X': 0, 'Y': 0, 'Width': 800, 'Height': 600}}]
            >>> bounds = capture._calculate_composite_bounds(windows)
            >>> print(f"Composite size: {bounds['width']}x{bounds['height']}")
        """
        if not windows:
            return {'x': 0, 'y': 0, 'width': 1920, 'height': 1080}

        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')

        for window in windows:
            bounds = window.get('kCGWindowBounds', {})
            x = bounds.get('X', 0)
            y = bounds.get('Y', 0)
            width = bounds.get('Width', 0)
            height = bounds.get('Height', 0)

            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + width)
            max_y = max(max_y, y + height)

        return {
            'x': min_x,
            'y': min_y,
            'width': max_x - min_x,
            'height': max_y - min_y
        }

    def _cgimage_to_pil(self, cg_image) -> Image.Image:
        """Convert CGImage to PIL Image.
        
        Converts a Quartz CGImage object to a PIL Image object,
        handling the necessary color space and format conversions.
        
        Args:
            cg_image: A Quartz CGImage object
            
        Returns:
            PIL Image object in RGBA format
            
        Note:
            This method handles the conversion from BGRA (Core Graphics)
            to RGBA (PIL) color format.
        """
        width = Quartz.CGImageGetWidth(cg_image)
        height = Quartz.CGImageGetHeight(cg_image)

        bytes_per_row = Quartz.CGImageGetBytesPerRow(cg_image)
        bitmap_data = Quartz.CGDataProviderCopyData(
            Quartz.CGImageGetDataProvider(cg_image)
        )

        # Convert to numpy array
        np_array = np.frombuffer(bitmap_data, dtype=np.uint8)
        np_array = np_array.reshape((height, bytes_per_row))
        np_array = np_array[:, :width * 4]
        np_array = np_array.reshape((height, width, 4))

        # Convert BGRA to RGBA
        np_array = np_array[:, :, [2, 1, 0, 3]]

        # Create PIL Image
        return Image.fromarray(np_array, 'RGBA')

    def _composite_window_images(self, window_images: List[Tuple[Image.Image, Dict]]) -> Image.Image:
        """Composite multiple window images into one.
        
        Combines multiple window images into a single composite image
        based on their screen positions, preserving transparency and layering.
        
        Args:
            window_images: List of tuples containing (PIL Image, window bounds dict)
            
        Returns:
            Composite PIL Image containing all windows positioned correctly,
            or None if no images provided.
            
        Example:
            >>> window_images = [(img1, {'X': 0, 'Y': 0}), (img2, {'X': 100, 'Y': 100})]
            >>> composite = capture._composite_window_images(window_images)
            >>> composite.save('composite.png')
        """
        if not window_images:
            return None

        # Calculate canvas size
        bounds = self._calculate_composite_bounds([img[1] for img in window_images])

        # Create canvas
        canvas = Image.new('RGBA',
                          (int(bounds['width']), int(bounds['height'])),
                          (0, 0, 0, 255))

        # Paste windows
        for img, window_bounds in window_images:
            x = int(window_bounds.get('X', 0) - bounds['x'])
            y = int(window_bounds.get('Y', 0) - bounds['y'])
            canvas.paste(img, (x, y), img if img.mode == 'RGBA' else None)

        return canvas

    def _get_cached(self, space_id: int) -> Optional[ScreenshotResult]:
        """Get cached screenshot if available and not expired.
        
        Retrieves a previously captured screenshot from the cache
        if it exists and hasn't exceeded the time-to-live threshold.
        
        Args:
            space_id: The ID of the desktop space
            
        Returns: