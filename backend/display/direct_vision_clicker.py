#!/usr/bin/env python3
"""
Direct Vision Clicker
=====================

Simple, fast approach: Take screenshot → Ask Claude where text is → Click it.
No complex pipelines, no guessing about Control Center icon shapes.

Author: Derek Russell
Date: 2025-10-16
Version: 1.0
"""

import asyncio
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from PIL import Image
import pyautogui
import re

logger = logging.getLogger(__name__)


class DirectVisionClicker:
    """
    Dead simple vision-based clicking:
    1. Take screenshot
    2. Ask Claude "Where is this text?"
    3. Click those coordinates

    No pipelines, no complexity, just works.
    """

    def __init__(self, vision_analyzer=None):
        """
        Initialize direct vision clicker

        Args:
            vision_analyzer: Claude Vision analyzer instance
        """
        self.vision_analyzer = vision_analyzer
        self.screenshots_dir = Path("/tmp/jarvis_vision_clicks")
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[DIRECT CLICKER] Initialized - simple screenshot → click approach")

    def set_vision_analyzer(self, analyzer):
        """Set or update the vision analyzer"""
        self.vision_analyzer = analyzer
        logger.info("[DIRECT CLICKER] Vision analyzer connected")

    async def _take_screenshot(self) -> Optional[Path]:
        """
        Take a screenshot of the entire screen

        Returns:
            Path to screenshot file, or None if failed
        """
        try:
            timestamp = int(time.time() * 1000)
            screenshot_path = self.screenshots_dir / f"screen_{timestamp}.png"

            # Use screencapture for reliability
            process = await asyncio.create_subprocess_exec(
                'screencapture', '-x', str(screenshot_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            await process.communicate()

            if screenshot_path.exists():
                logger.info(f"[DIRECT CLICKER] Screenshot saved: {screenshot_path}")
                return screenshot_path
            else:
                logger.error("[DIRECT CLICKER] Screenshot file not created")
                return None

        except Exception as e:
            logger.error(f"[DIRECT CLICKER] Screenshot failed: {e}")
            return None

    async def _find_text_coordinates(
        self,
        screenshot_path: Path,
        text_to_find: str
    ) -> Optional[Tuple[int, int]]:
        """
        Use Claude Vision to find text coordinates in screenshot

        Args:
            screenshot_path: Path to screenshot
            text_to_find: Text to locate (e.g., "Living Room TV")

        Returns:
            (x, y) coordinates of text center, or None if not found
        """
        if not self.vision_analyzer:
            logger.error("[DIRECT CLICKER] No vision analyzer available")
            return None

        try:
            # Load screenshot
            image = Image.open(screenshot_path)
            width, height = image.size

            logger.info(f"[DIRECT CLICKER] Asking Claude to find '{text_to_find}'...")
            logger.info(f"[DIRECT CLICKER] Screenshot size: {width}x{height}")

            # Ask Claude Vision where the text is
            prompt = f"""Look at this screenshot and find the text "{text_to_find}".

IMPORTANT: Return ONLY the coordinates in this EXACT format:
COORDINATES: x=<number>, y=<number>

Where:
- x is the horizontal pixel position from the left edge
- y is the vertical pixel position from the top edge
- Target the CENTER of the text "{text_to_find}"

Example response:
COORDINATES: x=450, y=125

If you cannot find "{text_to_find}", respond with:
NOT_FOUND

Do not include any other text or explanation."""

            # Analyze with Claude Vision
            result = await self.vision_analyzer.analyze_screenshot(
                image=image,
                prompt=prompt,
                use_cache=False
            )

            # Extract response
            if isinstance(result, tuple):
                analysis, metrics = result
                response_text = analysis.get('analysis', '')
            else:
                response_text = result.get('analysis', '')

            logger.info(f"[DIRECT CLICKER] Claude response: {response_text}")

            # Check for NOT_FOUND
            if "NOT_FOUND" in response_text:
                logger.warning(f"[DIRECT CLICKER] Claude could not find '{text_to_find}'")
                return None

            # Parse coordinates from response
            # Look for pattern: COORDINATES: x=123, y=456
            coord_match = re.search(r'x[=:]\s*(\d+).*?y[=:]\s*(\d+)', response_text, re.IGNORECASE)

            if coord_match:
                x = int(coord_match.group(1))
                y = int(coord_match.group(2))

                logger.info(f"[DIRECT CLICKER] ✅ Found '{text_to_find}' at ({x}, {y})")

                # Validate coordinates are within screen bounds
                if 0 <= x <= width and 0 <= y <= height:
                    return (x, y)
                else:
                    logger.warning(f"[DIRECT CLICKER] Coordinates out of bounds: ({x}, {y})")
                    return None
            else:
                logger.warning(f"[DIRECT CLICKER] Could not parse coordinates from: {response_text}")
                return None

        except Exception as e:
            logger.error(f"[DIRECT CLICKER] Vision analysis failed: {e}", exc_info=True)
            return None

    def _click_coordinates(self, x: int, y: int):
        """
        Click at the specified coordinates

        Args:
            x: X coordinate (in PHYSICAL pixels from screenshot)
            y: Y coordinate (in PHYSICAL pixels from screenshot)
        """
        try:
            logger.info(f"[DIRECT CLICKER] Received coordinates: ({x}, {y}) (physical pixels)")

            # CRITICAL FIX: Screenshots are captured at physical pixel resolution (e.g., 2880x1800 on Retina)
            # but PyAutoGUI works in logical pixels (e.g., 1440x900 on Retina).
            # We MUST convert from physical to logical pixels!

            try:
                from AppKit import NSScreen
                main_screen = NSScreen.mainScreen()
                dpi_scale = main_screen.backingScaleFactor()
            except:
                dpi_scale = 1.0
                logger.warning("[DIRECT CLICKER] Could not detect DPI scale, assuming 1.0x")

            # Convert from physical to logical pixels
            logical_x = int(round(x / dpi_scale))
            logical_y = int(round(y / dpi_scale))

            logger.info(
                f"[DIRECT CLICKER] DPI correction: "
                f"Physical ({x}, {y}) -> Logical ({logical_x}, {logical_y}) [scale={dpi_scale}x]"
            )

            logger.info(f"[DIRECT CLICKER] Clicking at ({logical_x}, {logical_y}) (logical pixels)")

            # Move mouse and click using LOGICAL pixels
            pyautogui.moveTo(logical_x, logical_y, duration=0.3)
            time.sleep(0.1)
            pyautogui.click()

            logger.info(f"[DIRECT CLICKER] ✅ Click executed at ({logical_x}, {logical_y})")

        except Exception as e:
            logger.error(f"[DIRECT CLICKER] Click failed: {e}")
            raise

    async def find_and_click_text(
        self,
        text_to_find: str,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Find text in current screen and click it

        Args:
            text_to_find: Text to locate and click (e.g., "Living Room TV")
            max_retries: Number of retry attempts if first fails

        Returns:
            Result dictionary with success status and details
        """
        start_time = time.time()

        logger.info(f"[DIRECT CLICKER] ========================================")
        logger.info(f"[DIRECT CLICKER] Finding and clicking: '{text_to_find}'")
        logger.info(f"[DIRECT CLICKER] Method: Screenshot → Claude Vision → Click")
        logger.info(f"[DIRECT CLICKER] ========================================")

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"[DIRECT CLICKER] Retry attempt {attempt}/{max_retries}")
                    await asyncio.sleep(1.0)  # Brief delay between retries

                # Step 1: Take screenshot
                screenshot_path = await self._take_screenshot()
                if not screenshot_path:
                    continue

                # Step 2: Find text coordinates using Claude Vision
                coordinates = await self._find_text_coordinates(screenshot_path, text_to_find)
                if not coordinates:
                    logger.warning(f"[DIRECT CLICKER] Could not locate '{text_to_find}' (attempt {attempt + 1})")
                    continue

                x, y = coordinates

                # Step 3: Click the coordinates
                self._click_coordinates(x, y)

                duration = time.time() - start_time

                logger.info(f"[DIRECT CLICKER] ✅ SUCCESS in {duration:.2f}s")
                logger.info(f"[DIRECT CLICKER] Found and clicked '{text_to_find}' at ({x}, {y})")
                logger.info(f"[DIRECT CLICKER] ========================================")

                return {
                    "success": True,
                    "message": f"Clicked '{text_to_find}' at ({x}, {y})",
                    "coordinates": {"x": x, "y": y},
                    "duration": duration,
                    "attempts": attempt + 1,
                    "method": "direct_vision_click"
                }

            except Exception as e:
                logger.error(f"[DIRECT CLICKER] Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    duration = time.time() - start_time
                    return {
                        "success": False,
                        "message": f"Failed to find or click '{text_to_find}' after {max_retries + 1} attempts",
                        "error": str(e),
                        "duration": duration,
                        "attempts": attempt + 1
                    }

        duration = time.time() - start_time
        return {
            "success": False,
            "message": f"Could not locate '{text_to_find}' in {max_retries + 1} attempts",
            "duration": duration,
            "attempts": max_retries + 1
        }


# Singleton instance
_direct_clicker: Optional[DirectVisionClicker] = None


def get_direct_clicker() -> DirectVisionClicker:
    """Get singleton direct clicker instance"""
    global _direct_clicker
    if _direct_clicker is None:
        _direct_clicker = DirectVisionClicker()
    return _direct_clicker


if __name__ == "__main__":
    # Test the direct clicker
    async def test():
        logging.basicConfig(level=logging.INFO)

        clicker = get_direct_clicker()

        # Note: In real usage, you'd connect a vision analyzer here
        # clicker.set_vision_analyzer(your_vision_analyzer)

        print("\n=== Testing Direct Vision Clicker ===\n")
        print("This would find and click 'Living Room TV' if vision analyzer was connected")

        # Simulate the API
        result = await clicker.find_and_click_text("Living Room TV")
        print(f"\nResult: {result}")

    asyncio.run(test())
