#!/usr/bin/env python3
"""
Hybrid display connector for macOS
Uses PyAutoGUI + AppleScript + Direct Vision Click
"""

import subprocess
import asyncio
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import pyautogui
import time

logger = logging.getLogger(__name__)


class AppleScriptDisplayConnector:
    """
    Connect to AirPlay displays using hybrid approach:
    - PyAutoGUI for clicking Control Center coordinates
    - AppleScript for menu bar interaction by index
    - Direct Vision Click for finding display names
    """

    def __init__(self):
        self.logger = logger
        self.direct_clicker = None  # Will be set by caller
        self.vision_analyzer = None  # Will be set by caller

    def set_direct_clicker(self, clicker):
        """Set the Direct Vision Clicker instance"""
        self.direct_clicker = clicker

    def set_vision_analyzer(self, analyzer):
        """Set the Vision Analyzer instance"""
        self.vision_analyzer = analyzer
        # Also initialize direct clicker if we have vision analyzer
        if analyzer and not self.direct_clicker:
            from .direct_vision_clicker import get_direct_clicker
            self.direct_clicker = get_direct_clicker()
            self.direct_clicker.set_vision_analyzer(analyzer)

    async def connect_to_display(self, display_name: str, timeout: float = 10.0) -> Dict[str, Any]:
        """
        Connect to an AirPlay display using AppleScript

        Args:
            display_name: Name of the display (e.g., "Living Room TV")
            timeout: Maximum time to wait for connection

        Returns:
            Dict with success status and message
        """
        self.logger.info(f"AppleScript: Attempting to connect to '{display_name}'")

        try:
            # Strategy 1: Try Control Center approach (macOS 11+)
            result = await self._connect_via_control_center(display_name, timeout)
            if result["success"]:
                return result

            # Strategy 2: Try System Preferences approach (fallback)
            self.logger.info("Control Center approach failed, trying System Preferences...")
            result = await self._connect_via_system_preferences(display_name, timeout)
            return result

        except Exception as e:
            self.logger.error(f"AppleScript connection failed: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"AppleScript error: {str(e)}",
                "error": str(e)
            }

    async def _connect_via_control_center(self, display_name: str, timeout: float) -> Dict[str, Any]:
        """
        Connect using Control Center with hybrid approach:
        1. Use AppleScript to click menu bar items by index to open Control Center
        2. Wait for menu to appear
        3. Use Direct Vision Click to find and click display name
        """
        try:
            # Step 1: Try clicking different menu bar items to find Screen Mirroring
            # From our testing: items 2-7 open Control Center menus
            # We'll try each one and use Direct Vision Click to see if "Living Room TV" appears

            success = False
            for item_index in [3, 4, 5, 2, 6, 7]:  # Try likely candidates first
                self.logger.info(f"Trying menu bar item {item_index}...")

                # Click menu bar item using AppleScript
                applescript = f'''
                tell application "System Events"
                    tell process "ControlCenter"
                        click menu bar item {item_index} of menu bar 1
                        delay 0.3
                    end tell
                end tell
                '''

                result = await self._run_applescript(applescript, timeout=5.0)
                if not result or "error" in result.lower():
                    continue

                # Wait for menu to appear
                await asyncio.sleep(0.5)

                # Use Direct Vision Click to find display name
                if self.direct_clicker:
                    self.logger.info(f"Looking for '{display_name}' in opened menu...")
                    click_result = await self.direct_clicker.find_and_click_text(
                        display_name,
                        max_retries=1
                    )

                    if click_result.get("success"):
                        self.logger.info(f"✓ Successfully clicked '{display_name}'!")
                        success = True
                        break
                    else:
                        # Display not found, close menu and try next item
                        self.logger.debug(f"'{display_name}' not found in item {item_index} menu")
                        await self._close_menu()
                else:
                    self.logger.warning("Direct clicker not available")
                    break

            if success:
                return {
                    "success": True,
                    "message": f"Connected to {display_name} via Control Center",
                    "method": "control_center_hybrid"
                }
            else:
                return {
                    "success": False,
                    "message": f"Could not find '{display_name}' in any Control Center menu",
                    "method": "control_center_hybrid"
                }

        except Exception as e:
            self.logger.error(f"Control Center hybrid approach failed: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "method": "control_center_hybrid"
            }

    async def _close_menu(self):
        """Close any open Control Center menu"""
        applescript = '''
        tell application "System Events"
            key code 53
        end tell
        '''
        await self._run_applescript(applescript, timeout=2.0)

    async def _connect_via_system_preferences(self, display_name: str, timeout: float) -> Dict[str, Any]:
        """
        Connect using System Preferences/Settings (fallback)

        AppleScript approach:
        1. Open System Preferences/Settings
        2. Navigate to Displays pane
        3. Find and select the display
        4. Enable screen mirroring
        """
        # Check macOS version to determine if it's System Preferences or System Settings
        version_check = await self._run_applescript('''
        set osVersion to system version of (system info)
        return osVersion
        ''', timeout=5.0)

        # macOS 13+ uses "System Settings", older uses "System Preferences"
        is_ventura_or_later = version_check and float('.'.join(version_check.split('.')[:2])) >= 13.0 if version_check else False
        app_name = "System Settings" if is_ventura_or_later else "System Preferences"
        pane_id = "com.apple.Displays-Settings.extension" if is_ventura_or_later else "com.apple.preference.displays"

        applescript = f'''
        tell application "{app_name}"
            activate
            delay 0.5

            -- Open Displays pane
            try
                reveal pane id "{pane_id}"
            on error
                reveal anchor "displaysDisplayTab" of pane id "com.apple.preference.displays"
            end try

            delay 1.0
        end tell

        tell application "System Events"
            tell process "{app_name}"
                try
                    -- Look for AirPlay Display dropdown or list
                    set displayPopup to pop up button 1 of window 1
                    click displayPopup
                    delay 0.3

                    -- Click on the display name
                    click menu item "{display_name}" of menu 1 of displayPopup
                    delay 0.5

                    -- Try to enable mirroring if available
                    try
                        set mirrorCheckbox to checkbox "Mirror Displays" of window 1
                        if value of mirrorCheckbox is 0 then
                            click mirrorCheckbox
                        end if
                    end try

                    return "SUCCESS: Connected to {display_name}"
                on error errMsg
                    return "ERROR: " & errMsg
                end try
            end tell
        end tell

        -- Close System Preferences/Settings
        tell application "{app_name}"
            quit
        end tell
        '''

        result = await self._run_applescript(applescript, timeout)

        if result and "SUCCESS" in result:
            self.logger.info(f"✓ Connected to {display_name} via System Preferences")
            return {
                "success": True,
                "message": f"Connected to {display_name}",
                "method": "system_preferences"
            }
        else:
            self.logger.warning(f"System Preferences approach failed: {result}")
            return {
                "success": False,
                "message": f"System Preferences approach failed: {result}",
                "method": "system_preferences"
            }

    async def _run_applescript(self, script: str, timeout: float = 10.0) -> Optional[str]:
        """
        Run an AppleScript and return the output

        Args:
            script: AppleScript code to execute
            timeout: Maximum execution time

        Returns:
            Script output or None if failed
        """
        try:
            # Run AppleScript via osascript
            proc = await asyncio.create_subprocess_exec(
                'osascript',
                '-e', script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Wait for completion with timeout
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout
            )

            # Check result
            if proc.returncode == 0:
                output = stdout.decode('utf-8').strip()
                self.logger.debug(f"AppleScript output: {output}")
                return output
            else:
                error = stderr.decode('utf-8').strip()
                self.logger.error(f"AppleScript error: {error}")
                return None

        except asyncio.TimeoutError:
            self.logger.error(f"AppleScript timed out after {timeout}s")
            try:
                proc.kill()
            except:
                pass
            return None
        except Exception as e:
            self.logger.error(f"Failed to run AppleScript: {e}", exc_info=True)
            return None

    async def disconnect_display(self, display_name: str) -> Dict[str, Any]:
        """
        Disconnect from an AirPlay display

        Args:
            display_name: Name of the display to disconnect

        Returns:
            Dict with success status and message
        """
        self.logger.info(f"AppleScript: Attempting to disconnect from '{display_name}'")

        applescript = f'''
        tell application "System Events"
            tell process "ControlCenter"
                try
                    -- Click Control Center icon
                    set controlCenterItem to menu bar item "Control Center" of menu bar 1
                    click controlCenterItem
                    delay 0.5

                    -- Click Screen Mirroring
                    set screenMirroringButton to button "Screen Mirroring" of window "Control Center"
                    click screenMirroringButton
                    delay 0.5

                    -- Click "Stop Mirroring" or display name to toggle off
                    try
                        set stopButton to button "Stop Mirroring" of window "Control Center"
                        click stopButton
                    on error
                        -- Click the display name again to toggle off
                        set displayButton to button "{display_name}" of window "Control Center"
                        click displayButton
                    end try

                    return "SUCCESS: Disconnected from {display_name}"
                on error errMsg
                    return "ERROR: " & errMsg
                end try
            end tell
        end tell
        '''

        result = await self._run_applescript(applescript, timeout=10.0)

        if result and "SUCCESS" in result:
            self.logger.info(f"✓ Disconnected from {display_name}")
            return {
                "success": True,
                "message": f"Disconnected from {display_name}"
            }
        else:
            return {
                "success": False,
                "message": f"Failed to disconnect: {result}"
            }

    async def list_available_displays(self) -> Dict[str, Any]:
        """
        List available AirPlay displays

        Returns:
            Dict with list of available displays
        """
        # This is tricky with AppleScript - we'd need to open Control Center
        # and parse the UI, which is not reliable
        # Better to use PyATV for discovery
        self.logger.warning("AppleScript cannot reliably list displays - use PyATV instead")
        return {
            "success": False,
            "message": "Use PyATV for display discovery",
            "displays": []
        }


# Singleton instance
_applescript_connector = None

def get_applescript_connector() -> AppleScriptDisplayConnector:
    """Get singleton AppleScript connector instance"""
    global _applescript_connector
    if _applescript_connector is None:
        _applescript_connector = AppleScriptDisplayConnector()
    return _applescript_connector


async def test_applescript_connector():
    """Test the AppleScript connector"""
    connector = get_applescript_connector()

    print("Testing AppleScript Display Connector\n")
    print("=" * 50)

    # Test connection
    print("\n1. Testing connection to 'Living Room TV'...")
    result = await connector.connect_to_display("Living Room TV")
    print(f"Result: {result}")

    if result["success"]:
        print("\n2. Waiting 5 seconds...")
        await asyncio.sleep(5)

        print("\n3. Testing disconnection...")
        result = await connector.disconnect_display("Living Room TV")
        print(f"Result: {result}")


if __name__ == "__main__":
    # Test the connector
    asyncio.run(test_applescript_connector())
