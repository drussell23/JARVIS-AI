"""
Direct voice handler for display connections
Handles commands like "Living Room TV" or "Connect to Living Room TV"
"""

import logging
import asyncio
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DisplayVoiceHandler:
    """Handles voice commands for display connections"""

    def __init__(self):
        self.display_monitor = None
        self.vision_navigator = None

    async def initialize(self):
        """Initialize connections to display monitor and vision navigator"""
        try:
            # Get display monitor
            from display import get_display_monitor
            self.display_monitor = get_display_monitor()

            # Get vision navigator
            from display.vision_ui_navigator import get_vision_navigator
            self.vision_navigator = get_vision_navigator()

            logger.info("[DISPLAY VOICE] Initialized display voice handler")
            return True
        except Exception as e:
            logger.error(f"[DISPLAY VOICE] Failed to initialize: {e}")
            return False

    async def handle_command(self, command: str) -> Dict[str, Any]:
        """
        Handle display-related voice commands

        Args:
            command: Voice command text

        Returns:
            Response dictionary
        """
        command_lower = command.lower().strip()

        # Check for display connection commands
        display_keywords = {
            "living room tv": "living_room_tv",
            "living room": "living_room_tv",
            "tv": "living_room_tv",
            "television": "living_room_tv",
            "connect to living room": "living_room_tv",
            "connect to tv": "living_room_tv",
            "connect to the tv": "living_room_tv",
            "connect tv": "living_room_tv",
            "connect living room": "living_room_tv",
            "mirror to tv": "living_room_tv",
            "mirror to living room": "living_room_tv",
            "extend to tv": "living_room_tv",
            "extend to living room": "living_room_tv"
        }

        # Check if command matches any display keyword
        for keyword, display_id in display_keywords.items():
            if keyword in command_lower:
                logger.info(f"[DISPLAY VOICE] Detected display command: '{command}' -> {display_id}")
                return await self.connect_to_display(display_id)

        # Check for disconnection commands
        if any(word in command_lower for word in ["disconnect", "stop mirroring", "stop mirror", "stop sharing"]):
            if "tv" in command_lower or "living room" in command_lower:
                return await self.disconnect_display("living_room_tv")

        return {
            "handled": False,
            "message": "Not a display command"
        }

    async def connect_to_display(self, display_id: str) -> Dict[str, Any]:
        """
        Connect to a specific display

        Args:
            display_id: Display identifier

        Returns:
            Connection result
        """
        try:
            logger.info(f"[DISPLAY VOICE] Connecting to display: {display_id}")

            # Ensure we're initialized
            if not self.display_monitor:
                await self.initialize()

            if not self.display_monitor:
                return {
                    "handled": True,
                    "success": False,
                    "message": "Display monitor not available"
                }

            # Connect to the display
            result = await self.display_monitor.connect_display(display_id)

            if result.get("success"):
                return {
                    "handled": True,
                    "success": True,
                    "message": f"Connected to {display_id.replace('_', ' ').title()}",
                    "result": result
                }
            else:
                return {
                    "handled": True,
                    "success": False,
                    "message": result.get("message", "Connection failed"),
                    "result": result
                }

        except Exception as e:
            logger.error(f"[DISPLAY VOICE] Error connecting to display: {e}")
            return {
                "handled": True,
                "success": False,
                "message": f"Error: {str(e)}"
            }

    async def disconnect_display(self, display_id: str) -> Dict[str, Any]:
        """
        Disconnect from a display

        Args:
            display_id: Display identifier

        Returns:
            Disconnection result
        """
        try:
            logger.info(f"[DISPLAY VOICE] Disconnecting from display: {display_id}")

            # For now, just press Escape to close Control Center/Screen Mirroring
            import pyautogui
            pyautogui.press('escape')
            await asyncio.sleep(0.5)
            pyautogui.press('escape')

            return {
                "handled": True,
                "success": True,
                "message": f"Disconnected from {display_id.replace('_', ' ').title()}"
            }

        except Exception as e:
            logger.error(f"[DISPLAY VOICE] Error disconnecting: {e}")
            return {
                "handled": True,
                "success": False,
                "message": f"Error: {str(e)}"
            }

# Global instance
_display_voice_handler = None

def get_display_voice_handler() -> DisplayVoiceHandler:
    """Get or create the global display voice handler"""
    global _display_voice_handler
    if _display_voice_handler is None:
        _display_voice_handler = DisplayVoiceHandler()
    return _display_voice_handler

async def handle_display_command(command: str) -> Dict[str, Any]:
    """
    Convenience function to handle display commands

    Args:
        command: Voice command text

    Returns:
        Response dictionary
    """
    handler = get_display_voice_handler()
    return await handler.handle_command(command)