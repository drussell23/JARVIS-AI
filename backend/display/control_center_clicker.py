#!/usr/bin/env python3
"""
Simple Control Center clicker - no vision needed
Just clicks the verified coordinates (1245, 12)
"""

import pyautogui
import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ControlCenterClicker:
    """Click Control Center icon and Screen Mirroring at verified coordinates"""

    # Verified Control Center coordinates for 1440x900 screen
    CONTROL_CENTER_X = 1245
    CONTROL_CENTER_Y = 12

    # Verified Screen Mirroring icon coordinates (inside Control Center menu)
    SCREEN_MIRRORING_X = 1393
    SCREEN_MIRRORING_Y = 177

    def __init__(self):
        self.logger = logger

    def open_control_center(self, wait_after_click: float = 0.5) -> Dict[str, Any]:
        """
        Open Control Center by clicking at verified coordinates

        Args:
            wait_after_click: Seconds to wait after clicking (for menu to open)

        Returns:
            Dict with success status and message
        """
        try:
            self.logger.info(f"Opening Control Center at ({self.CONTROL_CENTER_X}, {self.CONTROL_CENTER_Y})")

            # Move to Control Center icon
            pyautogui.moveTo(self.CONTROL_CENTER_X, self.CONTROL_CENTER_Y, duration=0.3)

            # Click it
            pyautogui.click(self.CONTROL_CENTER_X, self.CONTROL_CENTER_Y)

            # Wait for menu to open
            time.sleep(wait_after_click)

            self.logger.info("✓ Control Center opened")
            return {
                "success": True,
                "message": "Control Center opened",
                "coordinates": (self.CONTROL_CENTER_X, self.CONTROL_CENTER_Y)
            }

        except Exception as e:
            self.logger.error(f"Failed to open Control Center: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to open Control Center: {str(e)}",
                "error": str(e)
            }

    def close_control_center(self) -> Dict[str, Any]:
        """
        Close Control Center menu by pressing ESC

        Returns:
            Dict with success status and message
        """
        try:
            self.logger.info("Closing Control Center menu")
            pyautogui.press('escape')
            time.sleep(0.3)

            self.logger.info("✓ Control Center closed")
            return {
                "success": True,
                "message": "Control Center closed"
            }

        except Exception as e:
            self.logger.error(f"Failed to close Control Center: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to close Control Center: {str(e)}",
                "error": str(e)
            }

    def open_screen_mirroring(self, wait_after_click: float = 0.5) -> Dict[str, Any]:
        """
        Click Screen Mirroring icon in Control Center menu

        Args:
            wait_after_click: Seconds to wait after clicking (for submenu to open)

        Returns:
            Dict with success status and message
        """
        try:
            self.logger.info(f"Clicking Screen Mirroring at ({self.SCREEN_MIRRORING_X}, {self.SCREEN_MIRRORING_Y})")

            # Move to Screen Mirroring icon
            pyautogui.moveTo(self.SCREEN_MIRRORING_X, self.SCREEN_MIRRORING_Y, duration=0.3)

            # Click it
            pyautogui.click(self.SCREEN_MIRRORING_X, self.SCREEN_MIRRORING_Y)

            # Wait for submenu to open
            time.sleep(wait_after_click)

            self.logger.info("✓ Screen Mirroring menu opened")
            return {
                "success": True,
                "message": "Screen Mirroring menu opened",
                "coordinates": (self.SCREEN_MIRRORING_X, self.SCREEN_MIRRORING_Y)
            }

        except Exception as e:
            self.logger.error(f"Failed to open Screen Mirroring: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to open Screen Mirroring: {str(e)}",
                "error": str(e)
            }

    def open_control_center_and_screen_mirroring(self) -> Dict[str, Any]:
        """
        Complete flow: Open Control Center → Click Screen Mirroring

        Returns:
            Dict with success status and message
        """
        try:
            # Step 1: Open Control Center
            self.logger.info("Step 1: Opening Control Center...")
            cc_result = self.open_control_center(wait_after_click=0.5)

            if not cc_result.get('success'):
                return cc_result

            # Step 2: Click Screen Mirroring
            self.logger.info("Step 2: Opening Screen Mirroring menu...")
            sm_result = self.open_screen_mirroring(wait_after_click=0.5)

            if not sm_result.get('success'):
                return sm_result

            self.logger.info("✓ Control Center → Screen Mirroring flow complete")
            return {
                "success": True,
                "message": "Screen Mirroring menu is now open",
                "control_center_coords": (self.CONTROL_CENTER_X, self.CONTROL_CENTER_Y),
                "screen_mirroring_coords": (self.SCREEN_MIRRORING_X, self.SCREEN_MIRRORING_Y)
            }

        except Exception as e:
            self.logger.error(f"Failed to open Screen Mirroring flow: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed: {str(e)}",
                "error": str(e)
            }

    def click_control_center_icon(self, wait_for_menu: float = 0.5) -> Dict[str, Any]:
        """
        Convenience method: Open Control Center and return result

        Args:
            wait_for_menu: Seconds to wait for menu to open

        Returns:
            Dict with success status
        """
        return self.open_control_center(wait_after_click=wait_for_menu)


# Singleton instance
_control_center_clicker = None

def get_control_center_clicker() -> ControlCenterClicker:
    """Get singleton Control Center clicker instance"""
    global _control_center_clicker
    if _control_center_clicker is None:
        _control_center_clicker = ControlCenterClicker()
    return _control_center_clicker


def test_control_center_clicker():
    """Test the Control Center clicker with Screen Mirroring"""
    clicker = get_control_center_clicker()

    print("Testing Control Center → Screen Mirroring Flow\n")
    print("=" * 60)

    # Test complete flow
    print("\n1. Testing complete flow: Control Center → Screen Mirroring...")
    result = clicker.open_control_center_and_screen_mirroring()
    print(f"Result: {result}")

    if result["success"]:
        print("\n✓ Screen Mirroring menu is now open!")
        print(f"   Control Center clicked at: {result['control_center_coords']}")
        print(f"   Screen Mirroring clicked at: {result['screen_mirroring_coords']}")

        print("\n2. Waiting 3 seconds...")
        time.sleep(3)

        print("\n3. Closing menus...")
        pyautogui.press('escape')
        time.sleep(0.5)
        pyautogui.press('escape')

    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    test_control_center_clicker()
