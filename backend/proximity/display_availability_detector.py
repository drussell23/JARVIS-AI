"""
Display Availability Detector
==============================

Detects if displays (like Sony TV) are currently available/powered on.
Checks macOS display list to determine if external displays are connected.

Features:
- Real-time display availability checking
- Display power state inference
- Connection status tracking
- Periodic availability polling

Author: Derek Russell
Date: 2025-10-14
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)


# Import statement moved here for better organization
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from proximity.airplay_discovery import AirPlayDiscoveryService


class DisplayAvailabilityDetector:
    """
    Detects if displays are currently available (powered on and connected)
    
    Note: macOS doesn't provide direct "is TV powered on" detection.
    Instead, we check if the display appears in the active display list.
    If a TV is off or unplugged, it won't appear.
    """
    
    def __init__(self, poll_interval_seconds: float = 10.0):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.poll_interval_seconds = poll_interval_seconds
        
        # State tracking
        self.available_displays: Set[int] = set()
        self.last_check_time: Optional[datetime] = None
        self.availability_history: Dict[int, List[bool]] = {}
        
        # Performance
        self.total_checks = 0
        
        self.logger.info("[DISPLAY DETECTOR] Initialized")
    
    async def is_display_available(
        self,
        display_id: int = None,
        device_name: str = None
    ) -> bool:
        """
        Check if a specific display is currently available
        
        Args:
            display_id: Display ID to check (for active displays)
            device_name: Device name to check (for AirPlay displays)
            
        Returns:
            True if display is available (powered on and connected OR available via AirPlay)
        """
        try:
            # Check active displays by ID
            if display_id is not None:
                await self.refresh_available_displays()
                available = display_id in self.available_displays
                
                # Update history
                if display_id not in self.availability_history:
                    self.availability_history[display_id] = []
                self.availability_history[display_id].append(available)
                
                # Keep only last 100 checks
                if len(self.availability_history[display_id]) > 100:
                    self.availability_history[display_id] = self.availability_history[display_id][-100:]
                
                return available
            
            # Check AirPlay displays by name
            if device_name is not None:
                from proximity.airplay_discovery import get_airplay_discovery
                
                discovery = get_airplay_discovery()
                available = await discovery.is_device_available(device_name)
                
                self.logger.debug(f"[DISPLAY DETECTOR] AirPlay '{device_name}' available: {available}")
                return available
            
            return False
            
        except Exception as e:
            self.logger.error(f"[DISPLAY DETECTOR] Error checking display: {e}")
            return False
    
    async def refresh_available_displays(self, include_airplay: bool = True) -> Set[int]:
        """
        Refresh list of currently available displays
        
        Args:
            include_airplay: Also discover AirPlay-capable displays
        
        Returns:
            Set of available display IDs
        """
        try:
            self.total_checks += 1
            
            # Get displays from multi-monitor detector (active displays)
            from vision.multi_monitor_detector import MultiMonitorDetector
            
            detector = MultiMonitorDetector()
            displays = await detector.detect_displays()
            
            # Extract display IDs
            self.available_displays = {d.display_id for d in displays}
            self.last_check_time = datetime.now()
            
            self.logger.debug(f"[DISPLAY DETECTOR] Active displays: {self.available_displays}")
            
            # ENHANCED: Also discover AirPlay-capable displays (not yet connected)
            if include_airplay:
                airplay_displays = await self._discover_airplay_displays()
                self.logger.debug(f"[DISPLAY DETECTOR] AirPlay-capable displays: {airplay_displays}")
            
            return self.available_displays
            
        except Exception as e:
            self.logger.error(f"[DISPLAY DETECTOR] Error refreshing displays: {e}")
            return set()
    
    async def _discover_airplay_displays(self) -> List[str]:
        """
        Discover AirPlay-capable displays that aren't yet connected
        
        Returns:
            List of AirPlay device names
        """
        try:
            from proximity.airplay_discovery import get_airplay_discovery
            
            discovery = get_airplay_discovery()
            devices = await discovery.discover_airplay_devices()
            
            device_names = [d.device_name for d in devices if d.is_available]
            
            self.logger.info(f"[DISPLAY DETECTOR] Found {len(device_names)} AirPlay devices: {device_names}")
            return device_names
            
        except Exception as e:
            self.logger.error(f"[DISPLAY DETECTOR] AirPlay discovery error: {e}")
            return []
    
    async def get_availability_status(self, display_id: int) -> Dict:
        """
        Get detailed availability status for a display
        
        Args:
            display_id: Display ID
            
        Returns:
            Status dict with availability and history
        """
        available = await self.is_display_available(display_id)
        history = self.availability_history.get(display_id, [])
        
        # Calculate uptime percentage
        if history:
            uptime_pct = sum(history) / len(history)
        else:
            uptime_pct = 1.0 if available else 0.0
        
        return {
            "display_id": display_id,
            "available": available,
            "last_check": self.last_check_time.isoformat() if self.last_check_time else None,
            "check_count": len(history),
            "uptime_percentage": round(uptime_pct, 3),
            "status": "online" if available else "offline"
        }
    
    async def get_all_available_displays(self) -> List[int]:
        """
        Get list of all currently available display IDs
        
        Returns:
            List of available display IDs
        """
        await self.refresh_available_displays()
        return list(self.available_displays)
    
    def get_detector_stats(self) -> Dict:
        """Get detector statistics"""
        return {
            "total_checks": self.total_checks,
            "available_display_count": len(self.available_displays),
            "available_displays": list(self.available_displays),
            "last_check": self.last_check_time.isoformat() if self.last_check_time else None,
            "tracked_displays": len(self.availability_history)
        }


# Singleton instance
_availability_detector: Optional[DisplayAvailabilityDetector] = None

def get_availability_detector(poll_interval_seconds: float = 10.0) -> DisplayAvailabilityDetector:
    """Get singleton DisplayAvailabilityDetector instance"""
    global _availability_detector
    if _availability_detector is None:
        _availability_detector = DisplayAvailabilityDetector(poll_interval_seconds)
    return _availability_detector
