#!/usr/bin/env python3
"""
Core Graphics Window Capture - Capture windows from any space without switching
Uses CGWindowListCreateImage to capture specific windows by ID
"""

import Quartz
import Quartz.CoreGraphics as CG
from PIL import Image
import numpy as np
import io
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class CGWindowCapture:
    """Capture windows from any space without switching desktops"""

    @staticmethod
    def get_all_windows() -> List[Dict[str, Any]]:
        """Get information about all windows across all spaces"""
        windows = []

        # Get window list from Core Graphics
        window_list = CG.CGWindowListCopyWindowInfo(
            CG.kCGWindowListOptionAll,  # Get all windows, even from other spaces
            CG.kCGNullWindowID
        )

        for window in window_list:
            # Extract window properties
            window_info = {
                'id': window.get('kCGWindowNumber', 0),
                'name': window.get('kCGWindowName', ''),
                'owner': window.get('kCGWindowOwnerName', ''),
                'bounds': window.get('kCGWindowBounds', {}),
                'layer': window.get('kCGWindowLayer', 0),
                'alpha': window.get('kCGWindowAlpha', 1.0),
                'on_screen': window.get('kCGWindowIsOnscreen', False),
                'workspace': window.get('kCGWindowWorkspace', None)  # Space ID if available
            }

            # Filter out system windows and menu bars
            if window_info['layer'] == 0 and window_info['alpha'] > 0:
                windows.append(window_info)

        return windows

    @staticmethod
    def find_window_by_name(app_name: str, window_title: str = None) -> Optional[int]:
        """Find a window ID by application name and optional window title"""
        windows = CGWindowCapture.get_all_windows()

        for window in windows:
            # Check if app name matches
            if app_name.lower() in window['owner'].lower():
                # If no specific title requested, return first match
                if window_title is None:
                    logger.info(f"Found window: {window['owner']} - {window['name']} (ID: {window['id']})")
                    return window['id']
                # Otherwise check title too
                elif window_title.lower() in window['name'].lower():
                    logger.info(f"Found window: {window['owner']} - {window['name']} (ID: {window['id']})")
                    return window['id']

        logger.warning(f"Window not found: {app_name} - {window_title}")
        return None

    @staticmethod
    def capture_window_by_id(window_id: int) -> Optional[np.ndarray]:
        """
        Capture a specific window by its ID, regardless of which space it's in.
        This works WITHOUT switching spaces!

        NOTE: Requires Screen Recording permission in macOS System Settings
        """
        try:
            # Try different capture options for better compatibility
            capture_options = [
                CG.kCGWindowImageDefault,  # Default
                CG.kCGWindowImageBoundsIgnoreFraming,  # Without frame
                CG.kCGWindowImageShouldBeOpaque,  # Force opaque
            ]

            image = None
            for option in capture_options:
                image = CG.CGWindowListCreateImage(
                    CG.CGRectNull,  # Capture the window's natural bounds
                    CG.kCGWindowListOptionIncludingWindow,  # Only this specific window
                    window_id,
                    option
                )
                if image is not None:
                    break

            if image is None:
                logger.error(f"Failed to capture window {window_id} - Check Screen Recording permission in System Settings > Privacy & Security")
                return None

            # Convert CGImage to numpy array
            width = CG.CGImageGetWidth(image)
            height = CG.CGImageGetHeight(image)

            # Create bitmap context
            colorspace = CG.CGColorSpaceCreateDeviceRGB()
            bytes_per_row = width * 4

            # Create data buffer
            data = np.zeros((height, width, 4), dtype=np.uint8)

            # Create context and draw image
            context = CG.CGBitmapContextCreate(
                data,
                width,
                height,
                8,  # bits per component
                bytes_per_row,
                colorspace,
                CG.kCGImageAlphaPremultipliedLast | CG.kCGBitmapByteOrder32Big
            )

            CG.CGContextDrawImage(
                context,
                CG.CGRectMake(0, 0, width, height),
                image
            )

            # Convert RGBA to RGB
            screenshot = data[:, :, :3]  # Drop alpha channel

            logger.info(f"Successfully captured window {window_id} ({width}x{height})")
            return screenshot

        except Exception as e:
            logger.error(f"Error capturing window {window_id}: {e}")
            return None

    @staticmethod
    def capture_app_windows(app_name: str) -> Dict[str, np.ndarray]:
        """Capture all windows from a specific application across all spaces"""
        windows = CGWindowCapture.get_all_windows()
        captures = {}

        for window in windows:
            if app_name.lower() in window['owner'].lower():
                window_id = window['id']
                screenshot = CGWindowCapture.capture_window_by_id(window_id)

                if screenshot is not None:
                    window_key = f"{window['owner']}_{window['name']}_{window_id}"
                    captures[window_key] = screenshot
                    logger.info(f"Captured: {window_key}")

        return captures

    @staticmethod
    def capture_terminal_windows() -> Dict[str, np.ndarray]:
        """Specifically capture all Terminal windows from any space"""
        return CGWindowCapture.capture_app_windows("Terminal")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # List all windows
    print("\n=== All Windows ===")
    windows = CGWindowCapture.get_all_windows()
    for w in windows:
        if w['owner']:  # Skip windows without owners
            print(f"  {w['owner']}: {w['name']} (ID: {w['id']}, On Screen: {w['on_screen']})")

    # Find and capture Terminal
    print("\n=== Capturing Terminal ===")
    terminal_id = CGWindowCapture.find_window_by_name("Terminal")
    if terminal_id:
        screenshot = CGWindowCapture.capture_window_by_id(terminal_id)
        if screenshot is not None:
            print(f"Captured Terminal window: {screenshot.shape}")
            # Save it
            Image.fromarray(screenshot).save("/tmp/terminal_capture.png")
            print("Saved to /tmp/terminal_capture.png")