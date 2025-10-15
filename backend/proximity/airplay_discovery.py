"""
AirPlay Discovery Service
==========================

Discovers available AirPlay displays (like Sony TV) that appear in
macOS Screen Sharing menu but aren't yet connected.

This bridges the gap between proximity detection and display availability.
Core Graphics only detects ACTIVE displays; this discovers AVAILABLE displays.

Features:
- Discover AirPlay-capable devices on network
- Query Screen Sharing available targets
- Match discovered devices with registered displays
- Trigger AirPlay connection via AppleScript

Author: Derek Russell
Date: 2025-10-15
"""

import asyncio
import logging
import subprocess
import json
import re
from typing import Dict, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AirPlayDevice:
    """Represents an available AirPlay device"""
    device_name: str
    device_id: str
    device_type: str  # "tv", "monitor", "apple_tv", etc.
    is_available: bool
    network_address: Optional[str] = None
    model: Optional[str] = None
    discovered_at: Optional[datetime] = None


class AirPlayDiscoveryService:
    """
    Discovers available AirPlay displays that aren't yet connected
    
    Why This Matters:
    - Core Graphics API (CGGetActiveDisplayList) only sees ACTIVE displays
    - If Sony TV is on but not connected, CG API doesn't see it
    - This service discovers AVAILABLE displays for connection
    
    Discovery Methods:
    1. system_profiler SPAirPlayDataType
    2. AppleScript queries (Screen Sharing menu)
    3. Network scanning (Bonjour/mDNS for _airplay._tcp)
    """
    
    def __init__(self, scan_interval_seconds: float = 30.0):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.scan_interval_seconds = scan_interval_seconds
        
        # State
        self.available_devices: Dict[str, AirPlayDevice] = {}
        self.last_scan_time: Optional[datetime] = None
        self.discovery_cache_seconds = 60.0  # Cache for 60s
        
        # Statistics
        self.total_scans = 0
        self.total_devices_discovered = 0
        
        self.logger.info("[AIRPLAY DISCOVERY] Service initialized")
    
    async def discover_airplay_devices(self) -> List[AirPlayDevice]:
        """
        Discover all available AirPlay devices on network
        
        Returns:
            List of discovered AirPlay devices
        """
        try:
            self.total_scans += 1
            
            # Check cache
            if self._is_cache_valid():
                self.logger.debug("[AIRPLAY DISCOVERY] Using cached results")
                return list(self.available_devices.values())
            
            self.logger.info("[AIRPLAY DISCOVERY] Starting discovery scan...")
            
            # Method 1: system_profiler (macOS built-in)
            devices_from_profiler = await self._discover_via_system_profiler()
            
            # Method 2: AppleScript (Screen Sharing menu)
            devices_from_applescript = await self._discover_via_applescript()
            
            # Method 3: Bonjour/mDNS (network scan)
            devices_from_bonjour = await self._discover_via_bonjour()
            
            # Merge results
            all_devices = self._merge_discovery_results(
                devices_from_profiler,
                devices_from_applescript,
                devices_from_bonjour
            )
            
            # Update cache
            self.available_devices = {d.device_id: d for d in all_devices}
            self.last_scan_time = datetime.now()
            self.total_devices_discovered = len(all_devices)
            
            self.logger.info(f"[AIRPLAY DISCOVERY] Found {len(all_devices)} devices")
            return all_devices
            
        except Exception as e:
            self.logger.error(f"[AIRPLAY DISCOVERY] Error during discovery: {e}")
            return []
    
    async def _discover_via_system_profiler(self) -> List[AirPlayDevice]:
        """
        Discover AirPlay devices via system_profiler
        
        Command: system_profiler SPAirPlayDataType -json
        """
        try:
            self.logger.debug("[AIRPLAY] Querying system_profiler...")
            
            result = await asyncio.create_subprocess_exec(
                "system_profiler", "SPAirPlayDataType", "-json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=10.0)
            
            if result.returncode != 0:
                self.logger.warning(f"[AIRPLAY] system_profiler failed: {stderr.decode()}")
                return []
            
            data = json.loads(stdout.decode())
            devices = []
            
            # Parse AirPlay data
            airplay_data = data.get("SPAirPlayDataType", [])
            for entry in airplay_data:
                # Extract device info
                device_name = entry.get("_name", "Unknown")
                device_id = entry.get("_unique_identifier", f"airplay_{len(devices)}")
                
                device = AirPlayDevice(
                    device_name=device_name,
                    device_id=device_id,
                    device_type="airplay",
                    is_available=True,
                    discovered_at=datetime.now()
                )
                devices.append(device)
            
            self.logger.debug(f"[AIRPLAY] system_profiler found {len(devices)} devices")
            return devices
            
        except asyncio.TimeoutError:
            self.logger.warning("[AIRPLAY] system_profiler timeout")
            return []
        except Exception as e:
            self.logger.error(f"[AIRPLAY] system_profiler error: {e}")
            return []
    
    async def _discover_via_applescript(self) -> List[AirPlayDevice]:
        """
        Discover displays via AppleScript (Screen Sharing menu query)
        
        Uses AppleScript to query what's available in Screen Sharing menu
        """
        try:
            self.logger.debug("[AIRPLAY] Querying via AppleScript...")
            
            # AppleScript to get available AirPlay displays from menu
            # This checks the actual Screen Mirroring/AirPlay menu
            script = """
            tell application "System Events"
                try
                    tell process "ControlCenter"
                        -- Try to get Display menu items
                        set displayMenu to menu bar item "Display" of menu bar 1
                        click displayMenu
                        delay 0.5
                        
                        set menuItems to name of every menu item of menu 1 of displayMenu
                        click displayMenu -- close menu
                        
                        return menuItems as text
                    end tell
                on error errMsg
                    -- Fallback: Try Screen Mirroring in menu bar extras
                    try
                        tell process "SystemUIServer"
                            set menuItems to name of every menu bar item of menu bar 1
                            return menuItems as text
                        end tell
                    on error errMsg2
                        return "Error: " & errMsg2
                    end try
                end try
            end tell
            """
            
            result = await asyncio.create_subprocess_exec(
                "osascript", "-e", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=10.0)
            
            if result.returncode != 0:
                self.logger.debug(f"[AIRPLAY] AppleScript query returned error: {stderr.decode()}")
                return []
            
            output = stdout.decode().strip()
            self.logger.debug(f"[AIRPLAY] AppleScript output: {output}")
            
            # Parse output for display names
            devices = []
            
            # Look for common TV/display patterns
            if output and "Error" not in output:
                # Split by comma (AppleScript list separator)
                items = output.split(", ")
                
                for item in items:
                    # Look for TV-related keywords
                    if any(keyword in item for keyword in ["TV", "Television", "Display", "Monitor", "Living Room", "Sony", "Samsung", "LG"]):
                        device = AirPlayDevice(
                            device_name=item.strip(),
                            device_id=f"airplay_{item.strip().lower().replace(' ', '_')}",
                            device_type="tv",
                            is_available=True,
                            discovered_at=datetime.now()
                        )
                        devices.append(device)
                        self.logger.info(f"[AIRPLAY] Found display via AppleScript: {item}")
            
            self.logger.debug(f"[AIRPLAY] AppleScript found {len(devices)} devices")
            return devices
            
        except asyncio.TimeoutError:
            self.logger.warning("[AIRPLAY] AppleScript timeout")
            return []
        except Exception as e:
            self.logger.error(f"[AIRPLAY] AppleScript error: {e}")
            return []
    
    async def _discover_via_bonjour(self) -> List[AirPlayDevice]:
        """
        Discover AirPlay devices via Bonjour/mDNS network scanning
        
        Scans for _airplay._tcp service on local network
        """
        try:
            self.logger.debug("[AIRPLAY] Scanning via Bonjour/mDNS...")
            
            # Use dns-sd command (macOS built-in)
            result = await asyncio.create_subprocess_exec(
                "dns-sd", "-B", "_airplay._tcp", "local.",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait for results (5 second timeout)
            try:
                stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=5.0)
            except asyncio.TimeoutError:
                # dns-sd runs continuously, so timeout is expected
                result.terminate()
                await result.wait()
                stdout = b""
            
            output = stdout.decode()
            devices = []
            
            # Parse dns-sd output
            # Format: Timestamp Add/Rmv Instance Name Service Type Domain
            for line in output.split('\n'):
                if 'Add' in line and '_airplay._tcp' in line:
                    parts = line.split()
                    if len(parts) >= 7:
                        instance_name = ' '.join(parts[3:-3])
                        
                        device = AirPlayDevice(
                            device_name=instance_name,
                            device_id=f"bonjour_{instance_name.replace(' ', '_').lower()}",
                            device_type="airplay",
                            is_available=True,
                            discovered_at=datetime.now()
                        )
                        devices.append(device)
            
            self.logger.debug(f"[AIRPLAY] Bonjour found {len(devices)} devices")
            return devices
            
        except Exception as e:
            self.logger.error(f"[AIRPLAY] Bonjour scan error: {e}")
            return []
    
    def _merge_discovery_results(
        self,
        *device_lists: List[AirPlayDevice]
    ) -> List[AirPlayDevice]:
        """
        Merge results from multiple discovery methods
        
        Deduplicates by device name and prefers most recent data
        """
        merged: Dict[str, AirPlayDevice] = {}
        
        for device_list in device_lists:
            for device in device_list:
                # Use device name as key for deduplication
                key = device.device_name.lower().strip()
                
                if key not in merged:
                    merged[key] = device
                else:
                    # Prefer device with more information
                    existing = merged[key]
                    if device.network_address and not existing.network_address:
                        merged[key] = device
        
        return list(merged.values())
    
    def _is_cache_valid(self) -> bool:
        """Check if discovery cache is still valid"""
        if not self.last_scan_time:
            return False
        
        elapsed = (datetime.now() - self.last_scan_time).total_seconds()
        return elapsed < self.discovery_cache_seconds
    
    async def find_device_by_name(self, device_name: str) -> Optional[AirPlayDevice]:
        """
        Find a specific device by name
        
        Args:
            device_name: Device name to search for (e.g., "Sony TV")
            
        Returns:
            AirPlayDevice if found, None otherwise
        """
        # Refresh if cache is stale
        if not self._is_cache_valid():
            await self.discover_airplay_devices()
        
        # Search by name (case-insensitive, partial match)
        search_term = device_name.lower()
        
        for device in self.available_devices.values():
            if search_term in device.device_name.lower():
                return device
        
        return None
    
    async def is_device_available(self, device_name: str) -> bool:
        """
        Check if a specific device is currently available
        
        Args:
            device_name: Device name (e.g., "Sony Living Room TV")
            
        Returns:
            True if device is available for connection
        """
        device = await self.find_device_by_name(device_name)
        return device is not None and device.is_available
    
    async def connect_to_airplay_device(
        self,
        device_name: str,
        mode: str = "extend"
    ) -> Dict:
        """
        Connect to AirPlay device via AppleScript
        
        Args:
            device_name: Device name to connect to
            mode: "extend" or "mirror"
            
        Returns:
            Connection result
        """
        try:
            self.logger.info(f"[AIRPLAY] Connecting to {device_name} (mode: {mode})")
            
            # First, verify device is available
            device = await self.find_device_by_name(device_name)
            if not device:
                return {
                    "success": False,
                    "error": f"Device '{device_name}' not found"
                }
            
            # AppleScript to connect to AirPlay display
            script = f"""
            tell application "System Events"
                try
                    -- Click Screen Mirroring menu bar item
                    tell process "SystemUIServer"
                        set menuItems to menu bar items of menu bar 1
                        repeat with menuItem in menuItems
                            if description of menuItem contains "Displays" or description of menuItem contains "AirPlay" then
                                click menuItem
                                delay 0.5
                                
                                -- Find and click device
                                set menuItemsList to menu items of menu 1 of menuItem
                                repeat with item in menuItemsList
                                    if title of item contains "{device_name}" then
                                        click item
                                        delay 2
                                        
                                        -- Set mirror/extend mode
                                        {self._generate_mode_script(mode)}
                                        
                                        return "Connected"
                                    end if
                                end repeat
                            end if
                        end repeat
                    end tell
                    return "Device not found in menu"
                on error errMsg
                    return "Error: " & errMsg
                end try
            end tell
            """
            
            result = await asyncio.create_subprocess_exec(
                "osascript", "-e", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=15.0)
            output = stdout.decode().strip()
            
            if "Connected" in output:
                self.logger.info(f"[AIRPLAY] Successfully connected to {device_name}")
                return {
                    "success": True,
                    "device_name": device_name,
                    "mode": mode,
                    "message": f"Connected to {device_name}"
                }
            else:
                self.logger.warning(f"[AIRPLAY] Connection failed: {output}")
                return {
                    "success": False,
                    "error": output
                }
                
        except asyncio.TimeoutError:
            self.logger.error(f"[AIRPLAY] Connection timeout for {device_name}")
            return {
                "success": False,
                "error": "Connection timeout"
            }
        except Exception as e:
            self.logger.error(f"[AIRPLAY] Connection error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_mode_script(self, mode: str) -> str:
        """Generate AppleScript fragment for mirror/extend mode"""
        if mode == "mirror":
            return """
            -- Enable mirror displays
            click checkbox "Mirror Displays"
            """
        else:
            return """
            -- Extend (uncheck mirror)
            try
                click checkbox "Mirror Displays"
            end try
            """
    
    def get_discovery_stats(self) -> Dict:
        """Get discovery statistics"""
        return {
            "total_scans": self.total_scans,
            "total_devices_discovered": self.total_devices_discovered,
            "current_available_devices": len(self.available_devices),
            "available_device_names": [d.device_name for d in self.available_devices.values()],
            "last_scan": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "cache_valid": self._is_cache_valid()
        }


# Singleton instance
_airplay_discovery: Optional[AirPlayDiscoveryService] = None

def get_airplay_discovery(scan_interval_seconds: float = 30.0) -> AirPlayDiscoveryService:
    """Get singleton AirPlayDiscoveryService instance"""
    global _airplay_discovery
    if _airplay_discovery is None:
        _airplay_discovery = AirPlayDiscoveryService(scan_interval_seconds)
    return _airplay_discovery
