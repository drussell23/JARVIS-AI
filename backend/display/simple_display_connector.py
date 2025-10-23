#!/usr/bin/env python3
"""
Simple Display Connector - Hardcoded Logical Pixel Approach
===========================================================

KISS Principle: Use known logical pixel coordinates.
No complex pipelines, no DPI confusion, just works.

Coordinates (in LOGICAL pixels for PyAutoGUI):
- Control Center: (1235, 10)
- Screen Mirroring: (1396, 177)
- Living Room TV: (1223, 115)

Author: Derek J. Russell
Date: October 2025
Version: 2.0 - Simplified
"""

import asyncio
import logging
import time
from typing import Dict, Any
import pyautogui

logger = logging.getLogger(__name__)

# ============================================================================
# LOGICAL PIXEL COORDINATES (for PyAutoGUI)
# ============================================================================
# These are the EXACT positions of UI elements in logical pixel space.
# They work directly with PyAutoGUI - NO conversion needed!

CONTROL_CENTER_POS = (1235, 10)      # Control Center icon in menu bar
SCREEN_MIRRORING_POS = (1396, 177)   # Screen Mirroring menu item
LIVING_ROOM_TV_POS = (1223, 115)     # Living Room TV option

# Timing (seconds)
CLICK_DELAY = 0.3        # Delay after click for menu to open
DRAG_DURATION = 0.4      # Duration of drag motion
MOVE_DURATION = 0.3      # Duration of move motion


class SimpleDisplayConnector:
    """
    Dead simple display connector using hardcoded logical pixel coordinates.

    Why this works:
    1. Coordinates are in logical pixels (what PyAutoGUI expects)
    2. No DPI conversion needed
    3. Fast and reliable
    4. Easy to debug

    Trade-offs:
    - Coordinates may need updating if UI changes
    - No automatic adaptation to layout changes
    - But: Much simpler and more maintainable than vision-based approach
    """

    def __init__(self):
        """Initialize simple display connector"""
        logger.info("[SIMPLE CONNECTOR] Initialized with hardcoded coordinates")
        logger.info(f"[SIMPLE CONNECTOR] Control Center: {CONTROL_CENTER_POS}")
        logger.info(f"[SIMPLE CONNECTOR] Screen Mirroring: {SCREEN_MIRRORING_POS}")
        logger.info(f"[SIMPLE CONNECTOR] Living Room TV: {LIVING_ROOM_TV_POS}")

        # Disable PyAutoGUI failsafe for automated clicking
        # (can re-enable for safety if needed)
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1

    async def connect_to_living_room_tv(self) -> Dict[str, Any]:
        """
        Connect to Living Room TV using hardcoded coordinates.

        Flow:
        1. Drag to Control Center (opens Control Center menu)
        2. Click Screen Mirroring (opens Screen Mirroring submenu)
        3. Click Living Room TV (initiates connection)

        Returns:
            Result dict with success status and metadata
        """
        start_time = time.time()

        try:
            logger.info("[SIMPLE CONNECTOR] ========================================")
            logger.info("[SIMPLE CONNECTOR] Starting connection to Living Room TV")
            logger.info("[SIMPLE CONNECTOR] ========================================")

            # Step 1: Open Control Center
            logger.info(f"[SIMPLE CONNECTOR] Step 1: Dragging to Control Center at {CONTROL_CENTER_POS}")
            success = await self._click_control_center()

            if not success:
                return self._error_result("Failed to click Control Center", time.time() - start_time)

            await asyncio.sleep(CLICK_DELAY)

            # Step 2: Click Screen Mirroring
            logger.info(f"[SIMPLE CONNECTOR] Step 2: Clicking Screen Mirroring at {SCREEN_MIRRORING_POS}")
            success = await self._click_screen_mirroring()

            if not success:
                return self._error_result("Failed to click Screen Mirroring", time.time() - start_time)

            await asyncio.sleep(CLICK_DELAY)

            # Step 3: Click Living Room TV
            logger.info(f"[SIMPLE CONNECTOR] Step 3: Clicking Living Room TV at {LIVING_ROOM_TV_POS}")
            success = await self._click_living_room_tv()

            if not success:
                return self._error_result("Failed to click Living Room TV", time.time() - start_time)

            duration = time.time() - start_time
            logger.info(f"[SIMPLE CONNECTOR] ✅ SUCCESS! Connected in {duration:.2f}s")

            return {
                'success': True,
                'display': 'Living Room TV',
                'method': 'hardcoded_coordinates',
                'duration': duration,
                'steps': [
                    {'step': 1, 'action': 'Control Center', 'coordinates': CONTROL_CENTER_POS},
                    {'step': 2, 'action': 'Screen Mirroring', 'coordinates': SCREEN_MIRRORING_POS},
                    {'step': 3, 'action': 'Living Room TV', 'coordinates': LIVING_ROOM_TV_POS}
                ]
            }

        except Exception as e:
            logger.error(f"[SIMPLE CONNECTOR] Connection failed: {e}", exc_info=True)
            return self._error_result(str(e), time.time() - start_time)

    async def _click_control_center(self) -> bool:
        """
        Click Control Center icon using drag motion.

        Control Center requires a DRAG motion (not just moveTo) to activate.
        This simulates the user dragging down on the icon to open it.

        Returns:
            True if successful
        """
        try:
            x, y = CONTROL_CENTER_POS

            # Get current mouse position
            current_pos = pyautogui.position()
            logger.info(f"[SIMPLE CONNECTOR] Current mouse: {current_pos}")

            # Drag to Control Center (this opens it)
            logger.info(f"[SIMPLE CONNECTOR] Dragging to ({x}, {y}) [duration={DRAG_DURATION}s]")
            pyautogui.dragTo(x, y, duration=DRAG_DURATION, button='left')

            # Verify final position
            final_pos = pyautogui.position()
            logger.info(f"[SIMPLE CONNECTOR] Final mouse: {final_pos}")

            if final_pos.x == x and final_pos.y == y:
                logger.info("[SIMPLE CONNECTOR] ✅ Control Center drag successful")
                return True
            else:
                logger.error(f"[SIMPLE CONNECTOR] ❌ Mouse at wrong position: expected ({x}, {y}), got {final_pos}")
                return False

        except Exception as e:
            logger.error(f"[SIMPLE CONNECTOR] Control Center click failed: {e}")
            return False

    async def _click_screen_mirroring(self) -> bool:
        """
        Click Screen Mirroring menu item.

        Returns:
            True if successful
        """
        try:
            x, y = SCREEN_MIRRORING_POS

            # Move to position
            logger.info(f"[SIMPLE CONNECTOR] Moving to ({x}, {y}) [duration={MOVE_DURATION}s]")
            pyautogui.moveTo(x, y, duration=MOVE_DURATION)

            # Click
            await asyncio.sleep(0.1)
            logger.info(f"[SIMPLE CONNECTOR] Clicking at ({x}, {y})")
            pyautogui.click()

            logger.info("[SIMPLE CONNECTOR] ✅ Screen Mirroring click successful")
            return True

        except Exception as e:
            logger.error(f"[SIMPLE CONNECTOR] Screen Mirroring click failed: {e}")
            return False

    async def _click_living_room_tv(self) -> bool:
        """
        Click Living Room TV option.

        Returns:
            True if successful
        """
        try:
            x, y = LIVING_ROOM_TV_POS

            # Move to position
            logger.info(f"[SIMPLE CONNECTOR] Moving to ({x}, {y}) [duration={MOVE_DURATION}s]")
            pyautogui.moveTo(x, y, duration=MOVE_DURATION)

            # Click
            await asyncio.sleep(0.1)
            logger.info(f"[SIMPLE CONNECTOR] Clicking at ({x}, {y})")
            pyautogui.click()

            logger.info("[SIMPLE CONNECTOR] ✅ Living Room TV click successful")
            return True

        except Exception as e:
            logger.error(f"[SIMPLE CONNECTOR] Living Room TV click failed: {e}")
            return False

    def _error_result(self, error: str, duration: float) -> Dict[str, Any]:
        """Create error result dict"""
        return {
            'success': False,
            'display': 'Living Room TV',
            'method': 'hardcoded_coordinates',
            'duration': duration,
            'error': error
        }


# ============================================================================
# Convenience Functions
# ============================================================================

_connector_instance = None

def get_simple_connector() -> SimpleDisplayConnector:
    """Get singleton instance of simple connector"""
    global _connector_instance
    if _connector_instance is None:
        _connector_instance = SimpleDisplayConnector()
    return _connector_instance


async def connect_living_room_tv() -> Dict[str, Any]:
    """
    Convenience function to connect to Living Room TV.

    Usage:
        result = await connect_living_room_tv()
        if result['success']:
            print("Connected!")
        else:
            print(f"Failed: {result['error']}")

    Returns:
        Result dict with success status
    """
    connector = get_simple_connector()
    return await connector.connect_to_living_room_tv()


# ============================================================================
# Testing
# ============================================================================

async def test_connection():
    """Test the connection flow"""
    print("\n" + "="*80)
    print("TESTING SIMPLE DISPLAY CONNECTOR")
    print("="*80 + "\n")

    print("This will:")
    print("1. Drag to Control Center (1235, 10)")
    print("2. Click Screen Mirroring (1396, 177)")
    print("3. Click Living Room TV (1223, 115)")
    print("\nWatch the mouse carefully!")
    print("-" * 80)

    await asyncio.sleep(2)  # Give time to read

    result = await connect_living_room_tv()

    print("\n" + "-" * 80)
    print("RESULT:")
    print(f"  Success: {result['success']}")
    print(f"  Duration: {result['duration']:.2f}s")
    if result['success']:
        print(f"  Method: {result['method']}")
        print(f"  Steps completed: {len(result['steps'])}")
    else:
        print(f"  Error: {result['error']}")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Enable logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    asyncio.run(test_connection())
