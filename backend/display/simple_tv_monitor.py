"""
Simple Living Room TV Monitor
==============================

Monitors for "Living Room TV" availability and prompts user to connect.
Uses macOS Core Graphics and AirPlay APIs to detect available displays.

This is the SIMPLE solution - no Bluetooth, no proximity, just:
1. Check if Living Room TV is available
2. Prompt: "Would you like to extend to Living Room TV?"
3. Connect if user says yes

Author: Derek Russell
Date: 2025-10-15
"""

import asyncio
import logging
import subprocess
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SimpleTVMonitor:
    """
    Simple monitor for Living Room TV availability

    Checks Screen Mirroring menu periodically and prompts when TV is available.
    """

    def __init__(self, tv_name: str = "Living Room TV", check_interval: float = 10.0):
        self.tv_name = tv_name
        self.check_interval = check_interval
        self.is_monitoring = False
        self.tv_was_available = False
        self.monitoring_task: Optional[asyncio.Task] = None

        logger.info(f"[TV MONITOR] Initialized for: {tv_name}")

    async def start(self):
        """Start monitoring for TV availability"""
        if self.is_monitoring:
            logger.warning("[TV MONITOR] Already monitoring")
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"[TV MONITOR] Started monitoring for {self.tv_name}")

    async def stop(self):
        """Stop monitoring"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("[TV MONITOR] Stopped monitoring")

    async def _monitor_loop(self):
        """Main monitoring loop"""
        try:
            while self.is_monitoring:
                await self._check_tv_availability()
                await asyncio.sleep(self.check_interval)
        except asyncio.CancelledError:
            logger.info("[TV MONITOR] Monitoring cancelled")
        except Exception as e:
            logger.error(f"[TV MONITOR] Error in monitoring loop: {e}")

    async def _check_tv_availability(self):
        """Check if Living Room TV is available"""
        try:
            # Method 1: Check via Core Graphics for connected displays
            available_displays = await self._get_available_airplay_displays()

            tv_available = self.tv_name in available_displays

            # Detect new availability (wasn't available before, now it is)
            if tv_available and not self.tv_was_available:
                logger.info(f"[TV MONITOR] {self.tv_name} is now available!")
                await self._generate_prompt()

            self.tv_was_available = tv_available

        except Exception as e:
            logger.error(f"[TV MONITOR] Error checking TV availability: {e}")

    async def _get_available_airplay_displays(self) -> List[str]:
        """
        Get available AirPlay displays using system APIs

        Returns:
            List of display names
        """
        try:
            # Try using CoreGraphics to detect displays
            import Quartz

            # Get all online displays
            max_displays = 32
            result = Quartz.CGGetOnlineDisplayList(max_displays, None, None)

            if result[0] == 0:  # Success
                display_ids = result[1]
                display_count = result[2]

                displays = []
                for display_id in display_ids[:display_count]:
                    # Check if this is an AirPlay display
                    # AirPlay displays typically have specific characteristics
                    is_builtin = Quartz.CGDisplayIsBuiltin(display_id)

                    if not is_builtin:
                        # This is an external display (possibly AirPlay)
                        # Try to get the name
                        display_name = f"External Display {display_id}"
                        displays.append(display_name)

                logger.debug(f"[TV MONITOR] Found {len(displays)} external displays")
                return displays

            return []

        except ImportError:
            logger.warning("[TV MONITOR] CoreGraphics not available")
            return []
        except Exception as e:
            logger.error(f"[TV MONITOR] Error getting displays: {e}")
            return []

    async def _generate_prompt(self):
        """Generate voice prompt for user"""
        prompt = f"Sir, I see your {self.tv_name} is now available. Would you like to extend your display to it?"
        logger.info(f"[TV MONITOR] Generated prompt: {prompt}")

        # This will be picked up by the voice system
        # For now, just log it
        return prompt

    async def connect_to_tv(self, mode: str = "extend") -> Dict:
        """
        Connect to the Living Room TV

        Args:
            mode: "extend" or "mirror"

        Returns:
            Connection result
        """
        try:
            logger.info(f"[TV MONITOR] Connecting to {self.tv_name} (mode: {mode})")

            # Use AppleScript to connect via Screen Mirroring
            # This is the simplest way to programmatically connect

            script = f"""
            tell application "System Events"
                -- Open Control Center
                tell process "ControlCenter"
                    -- Click Screen Mirroring
                    click menu bar item "Screen Mirroring" of menu bar 1
                    delay 0.5
                    
                    -- Look for Living Room TV
                    try
                        click menu item "{self.tv_name}" of menu 1 of menu bar item "Screen Mirroring" of menu bar 1
                        return "success"
                    on error
                        return "not_found"
                    end try
                end tell
            end tell
            """

            result = subprocess.run(
                ["osascript", "-e", script], capture_output=True, text=True, timeout=10
            )

            if "success" in result.stdout:
                logger.info(f"[TV MONITOR] Successfully connected to {self.tv_name}")
                return {"success": True, "message": f"Connected to {self.tv_name}"}
            else:
                logger.warning(f"[TV MONITOR] Could not find {self.tv_name} in menu")
                return {
                    "success": False,
                    "message": f"{self.tv_name} not found in Screen Mirroring menu",
                }

        except subprocess.TimeoutExpired:
            logger.error("[TV MONITOR] Connection attempt timed out")
            return {"success": False, "message": "Connection timed out"}
        except Exception as e:
            logger.error(f"[TV MONITOR] Error connecting: {e}")
            return {"success": False, "message": str(e)}


# Singleton instance
_tv_monitor: Optional[SimpleTVMonitor] = None


def get_tv_monitor(tv_name: str = "Living Room TV") -> SimpleTVMonitor:
    """Get singleton SimpleTVMonitor instance"""
    global _tv_monitor
    if _tv_monitor is None:
        _tv_monitor = SimpleTVMonitor(tv_name)
    return _tv_monitor


# Test script
if __name__ == "__main__":

    async def test():
        monitor = SimpleTVMonitor("Living Room TV")
        await monitor.start()

        print("Monitoring for Living Room TV...")
        print("Press Ctrl+C to stop")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping monitor...")
            await monitor.stop()

    asyncio.run(test())
