#!/usr/bin/env python3
"""
Reliable Screenshot Capture System
Implements multiple capture methods with intelligent fallback
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

@dataclass
class ScreenshotResult:
    """Result of a screenshot capture attempt"""
    success: bool
    image: Optional[Image.Image]
    method: str
    space_id: Optional[int]
    error: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any]

class ReliableScreenshotCapture:
    """
    Multi-method screenshot capture with intelligent fallback
    """

    def __init__(self):
        self.methods = [
            ('quartz_composite', self._capture_quartz_composite),
            ('quartz_windows', self._capture_quartz_windows),
            ('appkit_screen', self._capture_appkit_screen),
            ('screencapture_cli', self._capture_screencapture_cli),
            ('window_server', self._capture_window_server)
        ]
        self._init_capture_cache()
        logger.info("Reliable Screenshot Capture initialized")

    def _init_capture_cache(self):
        """Initialize cache for recent captures"""
        self.cache = {}
        self.cache_ttl = 2  # seconds

    def capture_all_spaces(self) -> Dict[int, ScreenshotResult]:
        """
        Capture screenshots from all spaces with best available method
        """
        results = {}

        # Try to detect all spaces
        spaces = self._detect_available_spaces()

        for space_id in spaces:
            result = self.capture_space(space_id)
            results[space_id] = result

        return results

    def capture_space(self, space_id: int) -> ScreenshotResult:
        """
        Capture a specific space using best available method
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

    def _capture_quartz_composite(self, space_id: int) -> ScreenshotResult:
        """
        Use Quartz to capture composite window image
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
        """
        Capture individual windows and composite them
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
        """
        Use AppKit to capture screen
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
        """
        Use screencapture command line tool
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
        """
        Direct window server capture (requires permissions)
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
        """Detect available spaces"""
        # Try to get space count from window positions
        windows = CGWindowListCopyWindowInfo(
            kCGWindowListOptionOnScreenOnly,
            kCGNullWindowID
        )

        # Simple heuristic: assume 4 spaces by default
        # In production, integrate with MacOSSpaceDetector
        return [1, 2, 3, 4]

    def _get_windows_for_space(self, space_id: int) -> List[Dict]:
        """Get windows for a specific space"""
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
        """Calculate bounding box for all windows"""
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
        """Convert CGImage to PIL Image"""
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
        """Composite multiple window images into one"""
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
        """Get cached screenshot if available"""
        if space_id in self.cache:
            result, timestamp = self.cache[space_id]
            if time.time() - timestamp < self.cache_ttl:
                return result
        return None

    def _cache_result(self, space_id: int, result: ScreenshotResult):
        """Cache a screenshot result"""
        self.cache[space_id] = (result, time.time())

    def validate_capture(self, image: Image.Image) -> bool:
        """Validate that a capture is usable"""
        if not image:
            return False

        # Check dimensions
        if image.width < 100 or image.height < 100:
            return False

        # Check if not all black/white
        np_array = np.array(image)
        if np_array.std() < 10:  # Very low variance
            return False

        return True