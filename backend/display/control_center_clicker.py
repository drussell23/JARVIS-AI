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

    # Verified Living Room TV coordinates (inside Screen Mirroring submenu)
    LIVING_ROOM_TV_X = 1221
    LIVING_ROOM_TV_Y = 116

    # Verified Stop Screen Mirroring coordinates (inside Screen Mirroring submenu)
    STOP_MIRRORING_X = 1346
    STOP_MIRRORING_Y = 345

    # Change button (opens mode selection menu)
    CHANGE_BUTTON_X = 1218
    CHANGE_BUTTON_Y = 345

    # Mirroring mode options (after clicking Change button)
    ENTIRE_SCREEN_X = 553
    ENTIRE_SCREEN_Y = 285

    WINDOW_OR_APP_X = 723
    WINDOW_OR_APP_Y = 285

    EXTENDED_DISPLAY_X = 889
    EXTENDED_DISPLAY_Y = 283

    START_MIRRORING_X = 930
    START_MIRRORING_Y = 457

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

    def click_living_room_tv(self, wait_after_click: float = 0.5) -> Dict[str, Any]:
        """
        Click Living Room TV in Screen Mirroring submenu

        Args:
            wait_after_click: Seconds to wait after clicking (for connection to start)

        Returns:
            Dict with success status and message
        """
        try:
            self.logger.info(f"Clicking Living Room TV at ({self.LIVING_ROOM_TV_X}, {self.LIVING_ROOM_TV_Y})")

            # Move to Living Room TV
            pyautogui.moveTo(self.LIVING_ROOM_TV_X, self.LIVING_ROOM_TV_Y, duration=0.3)

            # Click it
            pyautogui.click(self.LIVING_ROOM_TV_X, self.LIVING_ROOM_TV_Y)

            # Wait for connection to initiate
            time.sleep(wait_after_click)

            self.logger.info("✓ Living Room TV clicked - connection initiated")
            return {
                "success": True,
                "message": "Living Room TV connection initiated",
                "coordinates": (self.LIVING_ROOM_TV_X, self.LIVING_ROOM_TV_Y)
            }

        except Exception as e:
            self.logger.error(f"Failed to click Living Room TV: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to click Living Room TV: {str(e)}",
                "error": str(e)
            }

    def connect_to_living_room_tv(self) -> Dict[str, Any]:
        """
        Complete flow: Control Center → Screen Mirroring → Living Room TV

        Returns:
            Dict with success status and message
        """
        try:
            # Step 1: Open Control Center
            self.logger.info("🎯 Step 1/3: Opening Control Center...")
            cc_result = self.open_control_center(wait_after_click=0.5)

            if not cc_result.get('success'):
                return cc_result

            # Step 2: Click Screen Mirroring
            self.logger.info("🎯 Step 2/3: Opening Screen Mirroring menu...")
            sm_result = self.open_screen_mirroring(wait_after_click=0.5)

            if not sm_result.get('success'):
                return sm_result

            # Step 3: Click Living Room TV
            self.logger.info("🎯 Step 3/3: Clicking Living Room TV...")
            tv_result = self.click_living_room_tv(wait_after_click=1.0)

            if not tv_result.get('success'):
                return tv_result

            self.logger.info("✅ Successfully connected to Living Room TV!")
            return {
                "success": True,
                "message": "Connected to Living Room TV",
                "control_center_coords": (self.CONTROL_CENTER_X, self.CONTROL_CENTER_Y),
                "screen_mirroring_coords": (self.SCREEN_MIRRORING_X, self.SCREEN_MIRRORING_Y),
                "living_room_tv_coords": (self.LIVING_ROOM_TV_X, self.LIVING_ROOM_TV_Y),
                "method": "direct_coordinates"
            }

        except Exception as e:
            self.logger.error(f"Failed to connect to Living Room TV: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed: {str(e)}",
                "error": str(e)
            }

    def open_control_center_and_screen_mirroring(self) -> Dict[str, Any]:
        """
        Partial flow: Open Control Center → Click Screen Mirroring

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

    def click_stop_mirroring(self, wait_after_click: float = 0.5) -> Dict[str, Any]:
        """
        Click Stop Screen Mirroring in Screen Mirroring submenu

        Args:
            wait_after_click: Seconds to wait after clicking (for disconnection to complete)

        Returns:
            Dict with success status and message
        """
        try:
            self.logger.info(f"Clicking Stop Mirroring at ({self.STOP_MIRRORING_X}, {self.STOP_MIRRORING_Y})")

            # Move to Stop Mirroring button
            pyautogui.moveTo(self.STOP_MIRRORING_X, self.STOP_MIRRORING_Y, duration=0.3)

            # Click it
            pyautogui.click(self.STOP_MIRRORING_X, self.STOP_MIRRORING_Y)

            # Wait for disconnection to complete
            time.sleep(wait_after_click)

            self.logger.info("✓ Stop Mirroring clicked - disconnection initiated")
            return {
                "success": True,
                "message": "Screen mirroring stopped",
                "coordinates": (self.STOP_MIRRORING_X, self.STOP_MIRRORING_Y)
            }

        except Exception as e:
            self.logger.error(f"Failed to click Stop Mirroring: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to stop mirroring: {str(e)}",
                "error": str(e)
            }

    def disconnect_from_living_room_tv(self) -> Dict[str, Any]:
        """
        Complete flow: Control Center → Screen Mirroring → Stop Mirroring

        Returns:
            Dict with success status and message
        """
        try:
            # Step 1: Open Control Center
            self.logger.info("🎯 Step 1/3: Opening Control Center...")
            cc_result = self.open_control_center(wait_after_click=0.5)

            if not cc_result.get('success'):
                return cc_result

            # Step 2: Click Screen Mirroring
            self.logger.info("🎯 Step 2/3: Opening Screen Mirroring menu...")
            sm_result = self.open_screen_mirroring(wait_after_click=0.5)

            if not sm_result.get('success'):
                return sm_result

            # Step 3: Click Stop Mirroring
            self.logger.info("🎯 Step 3/3: Clicking Stop Mirroring...")
            stop_result = self.click_stop_mirroring(wait_after_click=1.0)

            if not stop_result.get('success'):
                return stop_result

            self.logger.info("✅ Successfully disconnected from Living Room TV!")
            return {
                "success": True,
                "message": "Disconnected from Living Room TV",
                "control_center_coords": (self.CONTROL_CENTER_X, self.CONTROL_CENTER_Y),
                "screen_mirroring_coords": (self.SCREEN_MIRRORING_X, self.SCREEN_MIRRORING_Y),
                "stop_mirroring_coords": (self.STOP_MIRRORING_X, self.STOP_MIRRORING_Y),
                "method": "direct_coordinates"
            }

        except Exception as e:
            self.logger.error(f"Failed to disconnect from Living Room TV: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed: {str(e)}",
                "error": str(e)
            }

    def change_mirroring_mode(self, mode: str = "extended") -> Dict[str, Any]:
        """
        Change screen mirroring mode

        Args:
            mode: Mirroring mode - "entire", "window", or "extended"

        Returns:
            Dict with success status and message
        """
        try:
            # Step 1: Open Control Center
            self.logger.info("🎯 Step 1/5: Opening Control Center...")
            cc_result = self.open_control_center(wait_after_click=0.5)

            if not cc_result.get('success'):
                return cc_result

            # Step 2: Click Screen Mirroring
            self.logger.info("🎯 Step 2/5: Opening Screen Mirroring menu...")
            sm_result = self.open_screen_mirroring(wait_after_click=0.5)

            if not sm_result.get('success'):
                return sm_result

            # Step 3: Click Change button to open mode selection
            self.logger.info(f"🎯 Step 3/5: Clicking Change button at ({self.CHANGE_BUTTON_X}, {self.CHANGE_BUTTON_Y})...")
            pyautogui.moveTo(self.CHANGE_BUTTON_X, self.CHANGE_BUTTON_Y, duration=0.3)
            pyautogui.click(self.CHANGE_BUTTON_X, self.CHANGE_BUTTON_Y)
            time.sleep(0.5)

            # Step 4: Select mirroring mode
            mode_lower = mode.lower()
            if mode_lower in ["entire", "entire screen"]:
                mode_x, mode_y = self.ENTIRE_SCREEN_X, self.ENTIRE_SCREEN_Y
                mode_name = "Entire Screen"
            elif mode_lower in ["window", "window or app", "app"]:
                mode_x, mode_y = self.WINDOW_OR_APP_X, self.WINDOW_OR_APP_Y
                mode_name = "Window or App"
            elif mode_lower in ["extended", "extended display", "extend"]:
                mode_x, mode_y = self.EXTENDED_DISPLAY_X, self.EXTENDED_DISPLAY_Y
                mode_name = "Extended Display"
            else:
                return {
                    "success": False,
                    "message": f"Invalid mode: {mode}. Use 'entire', 'window', or 'extended'.",
                    "error": "Invalid mode"
                }

            self.logger.info(f"🎯 Step 4/5: Selecting {mode_name} mode at ({mode_x}, {mode_y})...")
            pyautogui.moveTo(mode_x, mode_y, duration=0.3)
            pyautogui.click(mode_x, mode_y)
            time.sleep(0.5)

            # Step 5: Click Start Mirroring
            self.logger.info(f"🎯 Step 5/5: Clicking Start Mirroring at ({self.START_MIRRORING_X}, {self.START_MIRRORING_Y})...")
            pyautogui.moveTo(self.START_MIRRORING_X, self.START_MIRRORING_Y, duration=0.3)
            pyautogui.click(self.START_MIRRORING_X, self.START_MIRRORING_Y)
            time.sleep(1.0)

            self.logger.info(f"✅ Successfully changed to {mode_name} mode!")
            return {
                "success": True,
                "message": f"Changed to {mode_name} mode",
                "mode": mode_name,
                "control_center_coords": (self.CONTROL_CENTER_X, self.CONTROL_CENTER_Y),
                "screen_mirroring_coords": (self.SCREEN_MIRRORING_X, self.SCREEN_MIRRORING_Y),
                "change_button_coords": (self.CHANGE_BUTTON_X, self.CHANGE_BUTTON_Y),
                "mode_coords": (mode_x, mode_y),
                "start_mirroring_coords": (self.START_MIRRORING_X, self.START_MIRRORING_Y),
                "method": "direct_coordinates"
            }

        except Exception as e:
            self.logger.error(f"Failed to change mirroring mode: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed: {str(e)}",
                "error": str(e)
            }

    def click_mirroring_mode(self, mode: str, wait_after_click: float = 0.5) -> Dict[str, Any]:
        """
        Click a specific mirroring mode option

        Args:
            mode: "entire", "window", or "extended"
            wait_after_click: Seconds to wait after clicking

        Returns:
            Dict with success status and message
        """
        try:
            mode_lower = mode.lower()
            if mode_lower in ["entire", "entire screen"]:
                x, y = self.ENTIRE_SCREEN_X, self.ENTIRE_SCREEN_Y
                mode_name = "Entire Screen"
            elif mode_lower in ["window", "window or app", "app"]:
                x, y = self.WINDOW_OR_APP_X, self.WINDOW_OR_APP_Y
                mode_name = "Window or App"
            elif mode_lower in ["extended", "extended display", "extend"]:
                x, y = self.EXTENDED_DISPLAY_X, self.EXTENDED_DISPLAY_Y
                mode_name = "Extended Display"
            else:
                return {
                    "success": False,
                    "message": f"Invalid mode: {mode}",
                    "error": "Invalid mode"
                }

            self.logger.info(f"Clicking {mode_name} at ({x}, {y})")
            pyautogui.moveTo(x, y, duration=0.3)
            pyautogui.click(x, y)
            time.sleep(wait_after_click)

            self.logger.info(f"✓ {mode_name} selected")
            return {
                "success": True,
                "message": f"{mode_name} selected",
                "coordinates": (x, y)
            }

        except Exception as e:
            self.logger.error(f"Failed to click mirroring mode: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed: {str(e)}",
                "error": str(e)
            }

    def click_start_mirroring(self, wait_after_click: float = 1.0) -> Dict[str, Any]:
        """
        Click Start Mirroring button

        Args:
            wait_after_click: Seconds to wait after clicking

        Returns:
            Dict with success status and message
        """
        try:
            self.logger.info(f"Clicking Start Mirroring at ({self.START_MIRRORING_X}, {self.START_MIRRORING_Y})")
            pyautogui.moveTo(self.START_MIRRORING_X, self.START_MIRRORING_Y, duration=0.3)
            pyautogui.click(self.START_MIRRORING_X, self.START_MIRRORING_Y)
            time.sleep(wait_after_click)

            self.logger.info("✓ Start Mirroring clicked")
            return {
                "success": True,
                "message": "Start Mirroring clicked",
                "coordinates": (self.START_MIRRORING_X, self.START_MIRRORING_Y)
            }

        except Exception as e:
            self.logger.error(f"Failed to click Start Mirroring: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed: {str(e)}",
                "error": str(e)
            }


# Singleton instance
_control_center_clicker = None

def get_control_center_clicker() -> ControlCenterClicker:
    """Get singleton Control Center clicker instance"""
    global _control_center_clicker
    if _control_center_clicker is None:
        _control_center_clicker = ControlCenterClicker()
    return _control_center_clicker


def test_control_center_clicker():
    """Test the complete Living Room TV connection flow"""
    clicker = get_control_center_clicker()

    print("Testing Complete Flow: Control Center → Screen Mirroring → Living Room TV\n")
    print("=" * 75)

    # Test complete flow
    print("\n🎯 Testing complete connection flow...")
    result = clicker.connect_to_living_room_tv()
    print(f"\nResult: {result}")

    if result["success"]:
        print("\n✅ Successfully connected to Living Room TV!")
        print(f"   1. Control Center clicked at: {result['control_center_coords']}")
        print(f"   2. Screen Mirroring clicked at: {result['screen_mirroring_coords']}")
        print(f"   3. Living Room TV clicked at: {result['living_room_tv_coords']}")
        print(f"   Method: {result['method']}")

        print("\n⏳ Screen mirroring should now be connecting...")
        print("   Check your TV to verify the connection!")

        print("\n⏱️  Waiting 5 seconds...")
        time.sleep(5)

    print("\n" + "=" * 75)
    print("✓ Test complete!")


if __name__ == "__main__":
    test_control_center_clicker()
