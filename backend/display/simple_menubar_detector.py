#!/usr/bin/env python3
"""
Simple Menu Bar Icon Detector
==============================

Fast, reliable detection of macOS menu bar icons using screen position heuristics.
No vision processing, no timeouts, just simple math.

This module provides a simple approach to detecting and interacting with macOS menu bar
icons by using known screen coordinates rather than complex image recognition or vision
processing. It's designed for reliable automation of Control Center and AirPlay features.

Author: Derek J. Russell
Date: October 2025

Example:
    >>> detector = SimpleMenuBarDetector()
    >>> x, y = detector.get_control_center_position()
    >>> print(f"Control Center at ({x}, {y})")
    Control Center at (1235, 10)
"""

import pyautogui
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class SimpleMenuBarDetector:
    """
    Simple menu bar icon detector using screen position heuristics.

    The macOS menu bar has a predictable layout:
    - Top 30 pixels of screen
    - Icons aligned from right to left
    - Control Center is typically 3rd from right (after battery and Wi-Fi)

    This class provides methods to get known coordinates for various menu bar elements
    and Control Center menu items, avoiding the complexity and unreliability of
    image recognition approaches.

    Attributes:
        screen_width (int): Width of the screen in pixels
        screen_height (int): Height of the screen in pixels

    Example:
        >>> detector = SimpleMenuBarDetector()
        >>> control_center_pos = detector.get_control_center_position()
        >>> screen_mirror_pos = detector.get_screen_mirroring_position()
    """

    def __init__(self) -> None:
        """
        Initialize the SimpleMenuBarDetector.

        Gets the current screen dimensions and logs the screen size for debugging.
        """
        self.screen_width, self.screen_height = pyautogui.size()
        logger.info(f"[SIMPLE MENUBAR] Screen: {self.screen_width}x{self.screen_height}")

    def get_control_center_position(self) -> Tuple[int, int]:
        """
        Get Control Center icon position using known coordinates.

        Returns the hardcoded coordinates for the Control Center icon in the macOS
        menu bar. These coordinates have been verified through user testing and
        provide reliable access to the Control Center menu.

        Returns:
            Tuple[int, int]: A tuple containing (x, y) coordinates of the Control Center icon.
                x-coordinate is typically around 1235 pixels from the left edge.
                y-coordinate is typically around 10 pixels from the top edge.

        Example:
            >>> detector = SimpleMenuBarDetector()
            >>> x, y = detector.get_control_center_position()
            >>> print(f"Control Center at ({x}, {y})")
            Control Center at (1235, 10)
        """
        # Known correct coordinates (verified by user hovering)
        x = 1235
        y = 10

        logger.info(f"[SIMPLE MENUBAR] Control Center at ({x}, {y})")

        return (x, y)

    def get_screen_mirroring_position(self) -> Tuple[int, int]:
        """
        Get Screen Mirroring menu item position within Control Center dropdown.

        Returns the hardcoded coordinates for the Screen Mirroring menu item that
        appears when the Control Center dropdown is opened. These coordinates
        have been verified through user testing.

        Returns:
            Tuple[int, int]: A tuple containing (x, y) coordinates of the Screen Mirroring menu item.
                x-coordinate is typically around 1396 pixels from the left edge.
                y-coordinate is typically around 177 pixels from the top edge.

        Note:
            This position is only valid when the Control Center dropdown menu is open.
            Call get_control_center_position() and click it first to open the menu.

        Example:
            >>> detector = SimpleMenuBarDetector()
            >>> x, y = detector.get_screen_mirroring_position()
            >>> print(f"Screen Mirroring at ({x}, {y})")
            Screen Mirroring at (1396, 177)
        """
        # Known correct coordinates (verified by user)
        x = 1396
        y = 177

        logger.info(f"[SIMPLE MENUBAR] Screen Mirroring at ({x}, {y})")

        return (x, y)

    def get_living_room_tv_position(self) -> Tuple[int, int]:
        """
        Get Living Room TV position within Screen Mirroring submenu.

        Returns the hardcoded coordinates for the "Living Room TV" AirPlay device
        that appears in the Screen Mirroring submenu. These coordinates have been
        verified through user testing.

        Returns:
            Tuple[int, int]: A tuple containing (x, y) coordinates of the Living Room TV option.
                x-coordinate is typically around 1223 pixels from the left edge.
                y-coordinate is typically around 115 pixels from the top edge.

        Note:
            This position is only valid when the Screen Mirroring submenu is open.
            You must first open Control Center, then click Screen Mirroring to
            access this menu item.

        Example:
            >>> detector = SimpleMenuBarDetector()
            >>> x, y = detector.get_living_room_tv_position()
            >>> print(f"Living Room TV at ({x}, {y})")
            Living Room TV at (1223, 115)
        """
        # Known correct coordinates (verified by user)
        x = 1223
        y = 115

        logger.info(f"[SIMPLE MENUBAR] Living Room TV at ({x}, {y})")

        return (x, y)


def get_simple_menubar_detector() -> SimpleMenuBarDetector:
    """
    Get singleton instance of SimpleMenuBarDetector.

    Implements the singleton pattern to ensure only one detector instance exists
    throughout the application lifecycle. This is useful for maintaining consistent
    screen dimensions and avoiding repeated initialization.

    Returns:
        SimpleMenuBarDetector: The singleton instance of the detector.

    Example:
        >>> detector1 = get_simple_menubar_detector()
        >>> detector2 = get_simple_menubar_detector()
        >>> assert detector1 is detector2  # Same instance
        True
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = SimpleMenuBarDetector()
    return _detector_instance


# Global singleton instance holder
_detector_instance: Optional[SimpleMenuBarDetector] = None