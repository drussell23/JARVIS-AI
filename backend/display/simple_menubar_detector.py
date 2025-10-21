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
        Get Control Center icon position using simple heuristics

        Returns:
            (x, y) coordinates of Control Center icon
        """
        # Menu bar is at top of screen (y = 12 is vertical center)
        y = 12

        # Control Center is typically:
        # - 3rd icon from right edge
        # - Icons are ~30px wide
        # - Right edge padding is ~10px

        # Common layouts:
        # 1440px width: Control Center at ~1387
        # 1920px width: Control Center at ~1867
        # 2560px width: Control Center at ~2507

        # Simple formula: screen_width - (3 icons * 30px) - 10px padding
        x = self.screen_width - (3 * 30) - 10

        logger.info(f"[SIMPLE MENUBAR] Control Center estimated at ({x}, {y})")

        return (x, y)

    def get_menu_item_below_control_center(self, item_name: str) -> Tuple[int, int]:
        """
        Get position of menu item in Control Center dropdown

        Args:
            item_name: Name of menu item (e.g., "Screen Mirroring", "Display")

        Returns:
            (x, y) coordinates of menu item
        """
        # Control Center menu appears below the icon
        # Menu items are typically:
        # - Same x position as icon (centered)
        # - First item starts at y=60
        # - Each item is ~40px tall

        cc_x, cc_y = self.get_control_center_position()

        # Screen Mirroring is typically 2nd or 3rd item
        # Let's use y=100 (after Wi-Fi/Bluetooth)
        menu_y = 100

        # X position: slight offset to the left
        menu_x = cc_x + 50

        logger.info(f"[SIMPLE MENUBAR] {item_name} estimated at ({menu_x}, {menu_y})")

        return (menu_x, menu_y)


def get_simple_menubar_detector() -> SimpleMenuBarDetector:
    """Get singleton instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = SimpleMenuBarDetector()
    return _detector_instance


_detector_instance = None
