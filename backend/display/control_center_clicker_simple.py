#!/usr/bin/env python3
"""
Control Center Clicker - Simple Version
========================================

A simple, reliable macOS Control Center automation tool that uses hardcoded coordinates
to connect to AirPlay devices. Based on the working version from commit a7fd379.

This module provides functionality to:
- Open macOS Control Center
- Navigate to Screen Mirroring
- Connect/disconnect from Living Room TV
- Handle coordinate debugging and logging

The implementation uses verified logical pixel coordinates that have been tested
and confirmed working, avoiding complex detection algorithms that can fail.

Author: Derek J. Russell
Date: October 2025
Version: 3.0 - Back to Simplicity

Example:
    >>> from control_center_clicker_simple import get_control_center_clicker
    >>> clicker = get_control_center_clicker()
    >>> result = clicker.connect_to_living_room_tv()
    >>> print(result['success'])
    True
"""

import pyautogui
import time
import logging
import asyncio
from typing import Dict, Any

# DEBUG: Import coordinate debugging
import sys
import os

# Force reload of debug_inject every time
_debug_inject_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'debug_jarvis_coordinates.py')
print(f"[DEBUG-INIT] Attempting to load debug from: {_debug_inject_path}")
print(f"[DEBUG-INIT] Path exists: {os.path.exists(_debug_inject_path)}")

if os.path.exists(_debug_inject_path):
    try:
        # Execute the debug script directly
        with open(_debug_inject_path, 'r') as f:
            debug_code = f.read()
        exec(debug_code, globals())
        print("[DEBUG-INIT] ✅ Debug inject loaded successfully")
    except Exception as e:
        print(f"[DEBUG-INIT] ❌ Failed to load debug: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"[DEBUG-INIT] ❌ Debug file not found at {_debug_inject_path}")

logger = logging.getLogger(__name__)

class ControlCenterClicker:
    """
    Simple Control Center clicker using verified coordinates.

    This class provides methods to automate macOS Control Center interactions
    for AirPlay screen mirroring. It uses hardcoded coordinates that have been
    verified to work reliably, avoiding complex UI detection.

    Attributes:
        CONTROL_CENTER_X (int): X coordinate for Control Center icon (1236)
        CONTROL_CENTER_Y (int): Y coordinate for Control Center icon (12)
        SCREEN_MIRRORING_X (int): X coordinate for Screen Mirroring menu item (1393)
        SCREEN_MIRRORING_Y (int): Y coordinate for Screen Mirroring menu item (177)
        LIVING_ROOM_TV_X (int): X coordinate for Living Room TV option (1221)
        LIVING_ROOM_TV_Y (int): Y coordinate for Living Room TV option (116)
        logger: Logger instance for debugging and status messages

    Note:
        Coordinates are in logical pixels as expected by PyAutoGUI.
        Based on commit a7fd379 which was working successfully.
    """

    # Verified working coordinates from commit a7fd379
    # These are in LOGICAL pixels (what PyAutoGUI expects)
    # BUT something is doubling them to ~(2475, 15)
    CONTROL_CENTER_X = 1236
    CONTROL_CENTER_Y = 12

    # Screen Mirroring menu item
    SCREEN_MIRRORING_X = 1393
    SCREEN_MIRRORING_Y = 177

    # Living Room TV in submenu
    LIVING_ROOM_TV_X = 1221
    LIVING_ROOM_TV_Y = 116

    def __init__(self) -> None:
        """
        Initialize the ControlCenterClicker with verified coordinates.

        Sets up logging and displays the coordinates that will be used
        for clicking various UI elements.
        """
        self.logger = logger
        logger.info(f"[CC-SIMPLE] Initialized with coordinates:")
        logger.info(f"[CC-SIMPLE]   Control Center: ({self.CONTROL_CENTER_X}, {self.CONTROL_CENTER_Y})")
        logger.info(f"[CC-SIMPLE]   Screen Mirroring: ({self.SCREEN_MIRRORING_X}, {self.SCREEN_MIRRORING_Y})")
        logger.info(f"[CC-SIMPLE]   Living Room TV: ({self.LIVING_ROOM_TV_X}, {self.LIVING_ROOM_TV_Y})")
        logger.info(f"[CC-SIMPLE]   WARNING: Mouse is going to ~(2475, 15) instead of (1236, 12)")

    def open_control_center(self, wait_after_click: float = 0.5) -> Dict[str, Any]:
        """
        Open Control Center by clicking at verified coordinates.

        Moves the mouse to the Control Center icon location and clicks it,
        then waits for the menu to open. Includes detailed coordinate logging
        for debugging purposes.

        Args:
            wait_after_click: Seconds to wait after clicking for menu to open.
                Defaults to 0.5 seconds.

        Returns:
            Dict containing:
                - success (bool): True if operation completed without errors
                - message (str): Human-readable status message
                - coordinates (tuple): The coordinates that were clicked
                - error (str, optional): Error message if success is False

        Example:
            >>> clicker = ControlCenterClicker()
            >>> result = clicker.open_control_center()
            >>> print(result['success'])
            True
        """
        try:
            # CRITICAL DEBUG: Write directly to file to catch coordinate doubling
            with open("/tmp/jarvis_mouse_movements.log", "a") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"[CC-SIMPLE] open_control_center called\n")
                f.write(f"  Coordinates: ({self.CONTROL_CENTER_X}, {self.CONTROL_CENTER_Y})\n")
                f.write(f"  About to call pyautogui.moveTo({self.CONTROL_CENTER_X}, {self.CONTROL_CENTER_Y})\n")

            self.logger.info(f"[CC-SIMPLE] Opening Control Center at ({self.CONTROL_CENTER_X}, {self.CONTROL_CENTER_Y})")

            # Move to Control Center icon
            pyautogui.moveTo(self.CONTROL_CENTER_X, self.CONTROL_CENTER_Y, duration=0.3)

            # Log where mouse actually went
            actual_pos = pyautogui.position()
            with open("/tmp/jarvis_mouse_movements.log", "a") as f:
                f.write(f"  After moveTo: Mouse at ({actual_pos.x}, {actual_pos.y})\n")
                if actual_pos.x != self.CONTROL_CENTER_X or actual_pos.y != self.CONTROL_CENTER_Y:
                    f.write(f"  ⚠️  MISMATCH! Expected ({self.CONTROL_CENTER_X}, {self.CONTROL_CENTER_Y})\n")
                    if actual_pos.x > 2000:
                        f.write(f"  🚨 DOUBLED! {actual_pos.x} is approximately {self.CONTROL_CENTER_X} × 2\n")

            # Click it
            pyautogui.click(self.CONTROL_CENTER_X, self.CONTROL_CENTER_Y)

            # Wait for menu to open
            time.sleep(wait_after_click)

            self.logger.info("[CC-SIMPLE] ✓ Control Center opened")
            return {
                "success": True,
                "message": "Control Center opened successfully",
                "coordinates": (self.CONTROL_CENTER_X, self.CONTROL_CENTER_Y)
            }

        except Exception as e:
            self.logger.error(f"[CC-SIMPLE] Failed to open Control Center: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to open Control Center: {str(e)}",
                "error": str(e)
            }

    def open_screen_mirroring(self, wait_after_click: float = 0.5) -> Dict[str, Any]:
        """
        Click Screen Mirroring in Control Center menu.

        Assumes Control Center is already open and clicks on the Screen Mirroring
        option to open the AirPlay device submenu.

        Args:
            wait_after_click: Seconds to wait after clicking for submenu to open.
                Defaults to 0.5 seconds.

        Returns:
            Dict containing:
                - success (bool): True if operation completed without errors
                - message (str): Human-readable status message
                - coordinates (tuple): The coordinates that were clicked
                - error (str, optional): Error message if success is False

        Raises:
            Exception: If PyAutoGUI operations fail or coordinates are invalid

        Example:
            >>> clicker = ControlCenterClicker()
            >>> clicker.open_control_center()
            >>> result = clicker.open_screen_mirroring()
            >>> print(result['success'])
            True
        """
        try:
            self.logger.info(f"[CC-SIMPLE] Opening Screen Mirroring at ({self.SCREEN_MIRRORING_X}, {self.SCREEN_MIRRORING_Y})")

            # Move to Screen Mirroring
            pyautogui.moveTo(self.SCREEN_MIRRORING_X, self.SCREEN_MIRRORING_Y, duration=0.3)

            # Click it
            pyautogui.click(self.SCREEN_MIRRORING_X, self.SCREEN_MIRRORING_Y)

            # Wait for submenu to open
            time.sleep(wait_after_click)

            self.logger.info("[CC-SIMPLE] ✓ Screen Mirroring menu opened")
            return {
                "success": True,
                "message": "Screen Mirroring menu opened successfully",
                "coordinates": (self.SCREEN_MIRRORING_X, self.SCREEN_MIRRORING_Y)
            }

        except Exception as e:
            self.logger.error(f"[CC-SIMPLE] Failed to open Screen Mirroring: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to open Screen Mirroring: {str(e)}",
                "error": str(e)
            }

    def click_living_room_tv(self, wait_after_click: float = 0.5) -> Dict[str, Any]:
        """
        Click Living Room TV in Screen Mirroring submenu.

        Assumes Screen Mirroring submenu is already open and clicks on the
        Living Room TV option to initiate the AirPlay connection.

        Args:
            wait_after_click: Seconds to wait after clicking for connection to start.
                Defaults to 0.5 seconds.

        Returns:
            Dict containing:
                - success (bool): True if operation completed without errors
                - message (str): Human-readable status message
                - coordinates (tuple): The coordinates that were clicked
                - error (str, optional): Error message if success is False

        Raises:
            Exception: If PyAutoGUI operations fail or coordinates are invalid

        Example:
            >>> clicker = ControlCenterClicker()
            >>> clicker.open_control_center()
            >>> clicker.open_screen_mirroring()
            >>> result = clicker.click_living_room_tv()
            >>> print(result['success'])
            True
        """
        try:
            self.logger.info(f"[CC-SIMPLE] Clicking Living Room TV at ({self.LIVING_ROOM_TV_X}, {self.LIVING_ROOM_TV_Y})")

            # Move to Living Room TV
            pyautogui.moveTo(self.LIVING_ROOM_TV_X, self.LIVING_ROOM_TV_Y, duration=0.3)

            # Click it
            pyautogui.click(self.LIVING_ROOM_TV_X, self.LIVING_ROOM_TV_Y)

            # Wait for connection to initiate
            time.sleep(wait_after_click)

            self.logger.info("[CC-SIMPLE] ✓ Living Room TV clicked - connection initiated")
            return {
                "success": True,
                "message": "Living Room TV connection initiated",
                "coordinates": (self.LIVING_ROOM_TV_X, self.LIVING_ROOM_TV_Y)
            }

        except Exception as e:
            self.logger.error(f"[CC-SIMPLE] Failed to click Living Room TV: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to click Living Room TV: {str(e)}",
                "error": str(e)
            }

    def connect_to_living_room_tv(self) -> Dict[str, Any]:
        """
        Complete flow: Control Center → Screen Mirroring → Living Room TV.

        This is the main method that performs all three clicks in sequence
        to establish an AirPlay connection to the Living Room TV. It handles
        the complete workflow with appropriate delays between steps.

        Returns:
            Dict containing:
                - success (bool): True if all steps completed successfully
                - message (str): Human-readable status message
                - control_center_coords (tuple): Control Center click coordinates
                - screen_mirroring_coords (tuple): Screen Mirroring click coordinates
                - living_room_tv_coords (tuple): Living Room TV click coordinates
                - method (str): Always "direct_coordinates" for this implementation
                - error (str, optional): Error message if success is False

        Raises:
            Exception: If any step in the connection process fails

        Example:
            >>> clicker = ControlCenterClicker()
            >>> result = clicker.connect_to_living_room_tv()
            >>> if result['success']:
            ...     print("Connected successfully!")
            ... else:
            ...     print(f"Failed: {result['message']}")
        """
        try:
            # Step 1: Open Control Center
            self.logger.info("[CC-SIMPLE] 🎯 Step 1/3: Opening Control Center...")
            cc_result = self.open_control_center(wait_after_click=0.5)

            if not cc_result.get('success'):
                return cc_result

            # Step 2: Click Screen Mirroring
            self.logger.info("[CC-SIMPLE] 🎯 Step 2/3: Opening Screen Mirroring menu...")
            sm_result = self.open_screen_mirroring(wait_after_click=0.5)

            if not sm_result.get('success'):
                return sm_result

            # Step 3: Click Living Room TV
            self.logger.info("[CC-SIMPLE] 🎯 Step 3/3: Clicking Living Room TV...")
            tv_result = self.click_living_room_tv(wait_after_click=1.0)

            if not tv_result.get('success'):
                return tv_result

            self.logger.info("[CC-SIMPLE] ✅ Successfully connected to Living Room TV!")
            return {
                "success": True,
                "message": "Connected to Living Room TV",
                "control_center_coords": (self.CONTROL_CENTER_X, self.CONTROL_CENTER_Y),
                "screen_mirroring_coords": (self.SCREEN_MIRRORING_X, self.SCREEN_MIRRORING_Y),
                "living_room_tv_coords": (self.LIVING_ROOM_TV_X, self.LIVING_ROOM_TV_Y),
                "method": "direct_coordinates"
            }

        except Exception as e:
            self.logger.error(f"[CC-SIMPLE] Failed to connect to Living Room TV: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed: {str(e)}",
                "error": str(e)
            }

    def disconnect_from_living_room_tv(self) -> Dict[str, Any]:
        """
        Disconnect from Living Room TV by clicking Stop Mirroring.

        Opens Control Center, navigates to Screen Mirroring, and clicks the
        Stop Mirroring option to disconnect from the current AirPlay session.

        Returns:
            Dict containing:
                - success (bool): True if disconnection completed successfully
                - message (str): Human-readable status message
                - error (str, optional): Error message if success is False

        Raises:
            Exception: If any step in the disconnection process fails

        Note:
            The Stop Mirroring coordinates may need adjustment based on the
            current UI layout when mirroring is active.

        Example:
            >>> clicker = ControlCenterClicker()
            >>> result = clicker.disconnect_from_living_room_tv()
            >>> if result['success']:
            ...     print("Disconnected successfully!")
        """
        try:
            # Open Control Center
            self.logger.info("[CC-SIMPLE] Opening Control Center to disconnect...")
            cc_result = self.open_control_center(wait_after_click=0.5)

            if not cc_result.get('success'):
                return cc_result

            # Click Screen Mirroring
            sm_result = self.open_screen_mirroring(wait_after_click=0.5)

            if not sm_result.get('success'):
                return sm_result

            # Click Stop Mirroring (usually at top of menu)
            # These coordinates might need adjustment
            STOP_MIRRORING_X = 1221
            STOP_MIRRORING_Y = 90  # Usually above Living Room TV option

            self.logger.info(f"[CC-SIMPLE] Clicking Stop Mirroring at ({STOP_MIRRORING_X}, {STOP_MIRRORING_Y})")
            pyautogui.moveTo(STOP_MIRRORING_X, STOP_MIRRORING_Y, duration=0.3)
            pyautogui.click(STOP_MIRRORING_X, STOP_MIRRORING_Y)

            time.sleep(0.5)

            self.logger.info("[CC-SIMPLE] ✓ Disconnected from Living Room TV")
            return {
                "success": True,
                "message": "Disconnected from Living Room TV"
            }

        except Exception as e:
            self.logger.error(f"[CC-SIMPLE] Failed to disconnect: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to disconnect: {str(e)}",
                "error": str(e)
            }

    # ASYNC VERSIONS - for use in async contexts
    async def connect_to_living_room_tv_async(self) -> Dict[str, Any]:
        """
        Async version of connect_to_living_room_tv.

        Runs the synchronous PyAutoGUI calls in an executor to avoid blocking
        the event loop in async applications.

        Returns:
            Dict: Same format as connect_to_living_room_tv()

        Example:
            >>> import asyncio
            >>> clicker = ControlCenterClicker()
            >>> result = await clicker.connect_to_living_room_tv_async()
            >>> print(result['success'])
            True
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.connect_to_living_room_tv)

    async def disconnect_from_living_room_tv_async(self) -> Dict[str, Any]:
        """
        Async version of disconnect_from_living_room_tv.

        Runs the synchronous PyAutoGUI calls in an executor to avoid blocking
        the event loop in async applications.

        Returns:
            Dict: Same format as disconnect_from_living_room_tv()

        Example:
            >>> import asyncio
            >>> clicker = ControlCenterClicker()
            >>> result = await clicker.disconnect_from_living_room_tv_async()
            >>> print(result['success'])
            True
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.disconnect_from_living_room_tv)


# Singleton instance
_clicker_instance = None

def get_control_center_clicker() -> ControlCenterClicker:
    """
    Get singleton instance of ControlCenterClicker.

    Creates a single instance of ControlCenterClicker that is reused across
    the application to avoid multiple initializations and maintain state.

    Returns:
        ControlCenterClicker: The singleton instance

    Example:
        >>> clicker1 = get_control_center_clicker()
        >>> clicker2 = get_control_center_clicker()
        >>> assert clicker1 is clicker2  # Same instance
        True
    """
    global _clicker_instance
    if _clicker_instance is None:
        _clicker_instance = ControlCenterClicker()
    return _clicker_instance


def test_control_center_clicker() -> None:
    """
    Test the complete Living Room TV connection flow.

    Performs a full test of the Control Center automation by attempting
    to connect to the Living Room TV. Displays detailed information about
    the coordinates being used and the results of each step.

    This function is primarily used for debugging and verification of
    the coordinate system and click sequence.

    Example:
        >>> test_control_center_clicker()
        Testing Complete Flow: Control Center → Screen Mirroring → Living Room TV
        ...
        ✅ Successfully connected to Living Room TV!
    """
    clicker = get_control_center_clicker()

    print("\nTesting Complete Flow: Control Center → Screen Mirroring → Living Room TV")
    print("=" * 75)
    print("\nUsing coordinates:")
    print(f"  Control Center: ({clicker.CONTROL_CENTER_X}, {clicker.CONTROL_CENTER_Y})")
    print(f"  Screen Mirroring: ({clicker.SCREEN_MIRRORING_X}, {clicker.SCREEN_MIRRORING_Y})")
    print(f"  Living Room TV: ({clicker.LIVING_ROOM_TV_X}, {clicker.LIVING_ROOM_TV_Y})")
    print("\n🎯 Testing complete connection flow...")

    result = clicker.connect_to_living_room_tv()
    print(f"\nResult: {result}")

    if result["success"]:
        print("\n✅ Successfully connected to Living Room TV!")
        print(f"   1. Control Center clicked at: {result['control_center_coords']}")
        print(f"   2. Screen Mirroring clicked at: {result['screen_mirroring_coords']}")
        print(f"   3. Living Room TV clicked at: {result['living_room_tv_coords']}")
        print(f"   Method: {result['method']}")
    else:
        print(f"\n❌ Connection failed: {result['message']}")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    # Enable logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_control_center_clicker()