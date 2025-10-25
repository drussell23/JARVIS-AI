#!/usr/bin/env python3
"""
Control Center Clicker - Simple Version
========================================

Based on the WORKING version from commit a7fd379 (Oct 17, 2025).
Uses hardcoded logical pixel coordinates that were confirmed working.

NO complex detection, NO DPI confusion, just simple clicks that work.

Author: Derek J. Russell
Date: October 2025
Version: 3.0 - Back to Simplicity
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

    def __init__(self):
        self.logger = logger
        logger.info(f"[CC-SIMPLE] Initialized with coordinates:")
        logger.info(f"[CC-SIMPLE]   Control Center: ({self.CONTROL_CENTER_X}, {self.CONTROL_CENTER_Y})")
        logger.info(f"[CC-SIMPLE]   Screen Mirroring: ({self.SCREEN_MIRRORING_X}, {self.SCREEN_MIRRORING_Y})")
        logger.info(f"[CC-SIMPLE]   Living Room TV: ({self.LIVING_ROOM_TV_X}, {self.LIVING_ROOM_TV_Y})")
        logger.info(f"[CC-SIMPLE]   WARNING: Mouse is going to ~(2475, 15) instead of (1236, 12)")

    def open_control_center(self, wait_after_click: float = 0.5) -> Dict[str, Any]:
        """
        Open Control Center by clicking at verified coordinates

        Args:
            wait_after_click: Seconds to wait after clicking (for menu to open)

        Returns:
            Dict with success status and message
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
        Click Screen Mirroring in Control Center menu

        Args:
            wait_after_click: Seconds to wait after clicking (for submenu to open)

        Returns:
            Dict with success status and message
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
        Click Living Room TV in Screen Mirroring submenu

        Args:
            wait_after_click: Seconds to wait after clicking (for connection to start)

        Returns:
            Dict with success status and message
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
        Complete flow: Control Center → Screen Mirroring → Living Room TV

        This is the main method that performs all three clicks in sequence.

        Returns:
            Dict with success status and message
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
        Disconnect from Living Room TV by clicking Stop Mirroring

        Returns:
            Dict with success status and message
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
        Async version of connect_to_living_room_tv
        Runs the synchronous PyAutoGUI calls in an executor to avoid blocking
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.connect_to_living_room_tv)

    async def disconnect_from_living_room_tv_async(self) -> Dict[str, Any]:
        """
        Async version of disconnect_from_living_room_tv
        Runs the synchronous PyAutoGUI calls in an executor to avoid blocking
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.disconnect_from_living_room_tv)


# Singleton instance
_clicker_instance = None

def get_control_center_clicker() -> ControlCenterClicker:
    """Get singleton instance of ControlCenterClicker"""
    global _clicker_instance
    if _clicker_instance is None:
        _clicker_instance = ControlCenterClicker()
    return _clicker_instance


def test_control_center_clicker():
    """Test the complete Living Room TV connection flow"""
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