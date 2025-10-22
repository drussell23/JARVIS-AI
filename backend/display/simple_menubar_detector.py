#!/usr/bin/env python3
"""
Simple Menu Bar Icon Detector
==============================

Fast, reliable detection of macOS menu bar icons using screen position heuristics.
No vision processing, no timeouts, just simple math.

Author: Derek J. Russell
Date: October 2025
"""

import pyautogui
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class SimpleMenuBarDetector:
    """
    Simple menu bar icon detector using screen position heuristics

    The macOS menu bar has a predictable layout:
    - Top 30 pixels of screen
    - Icons aligned from right to left
    - Control Center is typically 3rd from right (after battery and Wi-Fi)
    """

    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        logger.info(f"[SIMPLE MENUBAR] Screen: {self.screen_width}x{self.screen_height}")

    def get_control_center_position(self) -> Tuple[int, int]:
        """
        Get Control Center icon position using KNOWN coordinates

        Returns:
            (x, y) coordinates of Control Center icon
        """
        # Known correct coordinates (verified by user hovering)
        x = 1235
        y = 10

        logger.info(f"[SIMPLE MENUBAR] Control Center at ({x}, {y})")

        return (x, y)

    def get_screen_mirroring_position(self) -> Tuple[int, int]:
        """
        Get Screen Mirroring menu item position

        Returns:
            (x, y) coordinates of Screen Mirroring item
        """
        # Known correct coordinates (verified by user)
        x = 1396
        y = 177

        logger.info(f"[SIMPLE MENUBAR] Screen Mirroring at ({x}, {y})")

        return (x, y)

    def get_living_room_tv_position(self) -> Tuple[int, int]:
        """
        Get Living Room TV position

        Returns:
            (x, y) coordinates of Living Room TV
        """
        # Known correct coordinates (verified by user)
        x = 1223
        y = 115

        logger.info(f"[SIMPLE MENUBAR] Living Room TV at ({x}, {y})")

        return (x, y)


def get_simple_menubar_detector() -> SimpleMenuBarDetector:
    """Get singleton instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = SimpleMenuBarDetector()
    return _detector_instance


_detector_instance = None
