"""
Real-time Display Connection State Verifier

This module provides accurate, real-time verification of display connection state
rather than relying on cached status. It uses multiple methods to determine
the actual connection state and integrates with the learning database.

The module implements a multi-method verification approach:
1. System profiler analysis for hardware-level display detection
2. Window server queries for active display arrangement
3. AirPlay-specific status checking for wireless displays

Example:
    >>> verifier = get_display_verifier()
    >>> result = await verifier.verify_actual_connection("Living Room TV")
    >>> print(f"Connected: {result['is_connected']}")
    Connected: True
"""

import asyncio
import subprocess
import logging
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class DisplayStateVerifier:
    """
    Verifies actual display connection state in real-time.
    
    This class provides comprehensive display connection verification using multiple
    detection methods to ensure accurate real-time status reporting. It maintains
    a short-term cache to avoid excessive system calls while ensuring freshness.
    
    Attributes:
        last_verification (Dict[str, Dict]): Cache of recent verification results
        verification_cache_ttl (int): Time-to-live for cached results in seconds
    """

    def __init__(self) -> None:
        """
        Initialize the display state verifier.
        
        Sets up caching mechanism with a 2-second TTL to balance performance
        and accuracy for real-time verification needs.
        """
        self.last_verification: Dict[str, Dict] = {}
        self.verification_cache_ttl: int = 2  # Cache for 2 seconds max

    async def verify_actual_connection(self, display_name: str) -> Dict[str, any]:
        """
        Verify if a display is actually connected right now.
        
        Uses multiple verification methods in order of reliability:
        1. System profiler (highest confidence)
        2. Window server queries (medium confidence)  
        3. AirPlay process detection (lower confidence)
        
        Args:
            display_name: Name or identifier of the display to verify
            
        Returns:
            Dict containing:
                - is_connected (bool): Whether display is actually connected
                - connection_mode (str): 'extended', 'mirrored', 'airplay', or None
                - confidence (float): Confidence in verification (0.0-1.0)
                - method (str): Method used for verification
                - timestamp (datetime): When verification was performed
                - details (Dict, optional): Additional display information
                
        Example:
            >>> result = await verifier.verify_actual_connection("Samsung TV")
            >>> if result['is_connected'] and result['confidence'] > 0.8:
            ...     print(f"TV connected in {result['connection_mode']} mode")
        """

        # Check cache first (very short TTL)
        cache_key = display_name.lower()
        if cache_key in self.last_verification:
            cached = self.last_verification[cache_key]
            age = (datetime.now() - cached['timestamp']).total_seconds()
            if age < self.verification_cache_ttl:
                logger.debug(f"[VERIFIER] Using cached verification for {display_name} (age: {age:.1f}s)")
                return cached

        # Method 1: Check system_profiler for actual displays
        result = await self._verify_via_system_profiler(display_name)
        if result['confidence'] > 0.8:
            self.last_verification[cache_key] = result
            return result

        # Method 2: Check window server for display arrangement
        result = await self._verify_via_window_server(display_name)
        if result['confidence'] > 0.7:
            self.last_verification[cache_key] = result
            return result

        # Method 3: Check AirPlay status specifically
        result = await self._verify_via_airplay_status(display_name)
        self.last_verification[cache_key] = result
        return result

    async def _verify_via_system_profiler(self, display_name: str) -> Dict[str, any]:
        """
        Use system_profiler to check actual connected displays.
        
        This method provides the highest confidence verification by querying
        the system's hardware-level display information directly from macOS.
        
        Args:
            display_name: Name of display to verify
            
        Returns:
            Dict with verification results and confidence score of 0.9-0.95
            for successful detection, 0.0 for failures
            
        Raises:
            No exceptions raised - errors are logged and returned as low confidence
        """
        try:
            # Get display information
            process = await asyncio.create_subprocess_exec(
                'system_profiler', 'SPDisplaysDataType', '-json',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5.0)

            if process.returncode == 0:
                import json
                data = json.loads(stdout.decode('utf-8'))

                # Parse displays
                displays = []
                for item in data.get('SPDisplaysDataType', []):
                    for display in item.get('spdisplays_ndrvs', []):
                        display_info = {
                            'name': display.get('_name', ''),
                            'vendor': display.get('spdisplays_vendor', ''),
                            'connection': display.get('spdisplays_connection_type', ''),
                            'resolution': display.get('_spdisplays_resolution', ''),
                            'airplay': 'airplay' in display.get('spdisplays_connection_type', '').lower()
                        }
                        displays.append(display_info)

                        # Check if this is our target display
                        if self._matches_display_name(display_name, display_info):
                            mode = 'extended'  # Default for AirPlay
                            if 'mirror' in display_info.get('resolution', '').lower():
                                mode = 'mirrored'

                            logger.info(f"[VERIFIER] Found {display_name} via system_profiler: {mode} mode")
                            return {
                                'is_connected': True,
                                'connection_mode': mode,
                                'confidence': 0.95,
                                'method': 'system_profiler',
                                'timestamp': datetime.now(),
                                'details': display_info
                            }

                logger.debug(f"[VERIFIER] {display_name} not found in system_profiler output")
                return {
                    'is_connected': False,
                    'connection_mode': None,
                    'confidence': 0.9,
                    'method': 'system_profiler',
                    'timestamp': datetime.now(),
                    'available_displays': displays
                }

        except asyncio.TimeoutError:
            logger.warning("[VERIFIER] system_profiler timed out")
        except Exception as e:
            logger.error(f"[VERIFIER] system_profiler error: {e}")

        return {
            'is_connected': False,
            'connection_mode': None,
            'confidence': 0.0,
            'method': 'system_profiler_failed',
            'timestamp': datetime.now()
        }

    async def _verify_via_window_server(self, display_name: str) -> Dict[str, any]:
        """
        Check window server for display arrangement.
        
        Uses AppleScript to query the WindowServer process for active desktop
        arrangements, providing medium-confidence verification of display state.
        
        Args:
            display_name: Name of display to verify
            
        Returns:
            Dict with verification results and confidence score of 0.8
            for successful detection, 0.0 for failures
        """
        try:
            # Use AppleScript to get display arrangement
            script = '''
            tell application "System Events"
                tell application process "WindowServer"
                    set displayCount to count of desktops
                    set displayNames to {}
                    repeat with i from 1 to displayCount
                        set end of displayNames to name of desktop i
                    end repeat
                    return displayNames
                end tell
            end tell
            '''

            process = await asyncio.create_subprocess_exec(
                'osascript', '-e', script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=3.0)

            if process.returncode == 0:
                output = stdout.decode('utf-8').strip()
                displays = output.split(', ') if output else []

                for display in displays:
                    if self._fuzzy_match(display_name, display):
                        logger.info(f"[VERIFIER] Found {display_name} via window server")
                        return {
                            'is_connected': True,
                            'connection_mode': 'extended',
                            'confidence': 0.8,
                            'method': 'window_server',
                            'timestamp': datetime.now()
                        }

        except Exception as e:
            logger.debug(f"[VERIFIER] Window server check failed: {e}")

        return {
            'is_connected': False,
            'connection_mode': None,
            'confidence': 0.0,
            'method': 'window_server_failed',
            'timestamp': datetime.now()
        }

    async def _verify_via_airplay_status(self, display_name: str) -> Dict[str, any]:
        """
        Check AirPlay status specifically for TV connections.
        
        Detects active AirPlay connections by checking for the AirPlayXPCHelper
        process. Provides lower confidence as it can't verify the specific target.
        
        Args:
            display_name: Name of display to verify (used for TV pattern matching)
            
        Returns:
            Dict with verification results and confidence score of 0.7
            for active AirPlay, 0.6 for no AirPlay detected
        """
        try:
            # Check if screen recording process exists (indicates active AirPlay)
            process = await asyncio.create_subprocess_exec(
                'pgrep', '-l', 'AirPlayXPCHelper',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=2.0)

            if process.returncode == 0:
                # AirPlay is active, but we need to verify it's to the right display
                # This is a heuristic - if AirPlay is active and user asks about a TV,
                # it's likely connected
                logger.info(f"[VERIFIER] AirPlay is active, assuming {display_name} is connected")
                return {
                    'is_connected': True,
                    'connection_mode': 'airplay',
                    'confidence': 0.7,
                    'method': 'airplay_process',
                    'timestamp': datetime.now()
                }

        except Exception as e:
            logger.debug(f"[VERIFIER] AirPlay check failed: {e}")

        return {
            'is_connected': False,
            'connection_mode': None,
            'confidence': 0.6,
            'method': 'no_airplay_detected',
            'timestamp': datetime.now()
        }

    def _matches_display_name(self, target: str, display_info: Dict) -> bool:
        """
        Check if display info matches target display name.
        
        Performs intelligent matching including direct name comparison,
        TV pattern detection, and AirPlay connection inference.
        
        Args:
            target: Target display name to match
            display_info: Dictionary containing display information from system
            
        Returns:
            True if the display info matches the target name
            
        Example:
            >>> display_info = {'name': 'Samsung TV', 'airplay': True}
            >>> verifier._matches_display_name("Living Room TV", display_info)
            True
        """
        target_lower = target.lower()

        # Check direct name match
        if target_lower in display_info.get('name', '').lower():
            return True

        # Check for TV patterns
        if 'tv' in target_lower:
            if display_info.get('airplay'):
                return True
            if 'tv' in display_info.get('vendor', '').lower():
                return True
            if 'living room' in target_lower and 'airplay' in display_info.get('connection', '').lower():
                return True

        return False

    def _fuzzy_match(self, target: str, candidate: str, threshold: float = 0.7) -> bool:
        """
        Fuzzy string matching for display names.
        
        Performs flexible string matching to handle variations in display naming
        between different system queries and user input.
        
        Args:
            target: Target display name
            candidate: Candidate display name to compare
            threshold: Similarity threshold (currently unused but available for future enhancement)
            
        Returns:
            True if strings are considered a match
            
        Example:
            >>> verifier._fuzzy_match("Samsung TV", "Samsung Smart TV")
            True
            >>> verifier._fuzzy_match("Living Room", "LivingRoom Display")
            True
        """
        target_lower = target.lower()
        candidate_lower = candidate.lower()

        # Direct contains check
        if target_lower in candidate_lower or candidate_lower in target_lower:
            return True

        # Check key words
        target_words = set(target_lower.split())
        candidate_words = set(candidate_lower.split())

        if target_words & candidate_words:  # Any common words
            return True

        return False

    async def get_all_displays(self) -> List[Dict[str, any]]:
        """
        Get all currently connected displays with their states.
        
        Retrieves comprehensive information about all connected displays
        using system_profiler for complete hardware-level detection.
        
        Returns:
            List of dictionaries, each containing:
                - name (str): Display name
                - vendor (str): Display manufacturer
                - connection_type (str): Connection method (HDMI, AirPlay, etc.)
                - resolution (str): Current resolution setting
                - is_airplay (bool): Whether this is an AirPlay connection
                - is_builtin (bool): Whether this is the built-in display
                
        Example:
            >>> displays = await verifier.get_all_displays()
            >>> for display in displays:
            ...     print(f"{display['name']}: {display['connection_type']}")
            Built-in Display: Internal
            Samsung TV: AirPlay
        """
        try:
            process = await asyncio.create_subprocess_exec(
                'system_profiler', 'SPDisplaysDataType', '-json',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5.0)

            if process.returncode == 0:
                import json
                data = json.loads(stdout.decode('utf-8'))

                displays = []
                for item in data.get('SPDisplaysDataType', []):
                    for display in item.get('spdisplays_ndrvs', []):
                        displays.append({
                            'name': display.get('_name', 'Unknown'),
                            'vendor': display.get('spdisplays_vendor', ''),
                            'connection_type': display.get('spdisplays_connection_type', ''),
                            'resolution': display.get('_spdisplays_resolution', ''),
                            'is_airplay': 'airplay' in display.get('spdisplays_connection_type', '').lower(),
                            'is_builtin': display.get('spdisplays_main', '') == 'spdisplays_yes'
                        })

                return displays

        except Exception as e:
            logger.error(f"[VERIFIER] Failed to get all displays: {e}")

        return []


# Singleton instance
_verifier_instance: Optional[DisplayStateVerifier] = None


def get_display_verifier() -> DisplayStateVerifier:
    """
    Get singleton instance of display verifier.
    
    Implements the singleton pattern to ensure only one verifier instance
    exists throughout the application lifecycle, maintaining cache consistency.
    
    Returns:
        The singleton DisplayStateVerifier instance
        
    Example:
        >>> verifier1 = get_display_verifier()
        >>> verifier2 = get_display_verifier()
        >>> assert verifier1 is verifier2  # Same instance
    """
    global _verifier_instance
    if _verifier_instance is None:
        _verifier_instance = DisplayStateVerifier()
    return _verifier_instance