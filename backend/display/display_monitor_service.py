"""
Display Monitor Service
=======================

Monitors available displays in Screen Mirroring menu and prompts user
to connect when registered displays become available.

This is a SIMPLE display availability monitor - no proximity detection needed.

Features:
- Polls Screen Mirroring menu for available displays
- Detects when "Living Room TV" becomes available
- Prompts user: "Would you like to extend to Living Room TV?"
- Handles yes/no responses
- User override (don't ask again)

Author: Derek Russell
Date: 2025-10-15
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DisplayMonitorConfig:
    """Configuration for a monitored display"""
    display_name: str  # e.g., "Living Room TV"
    auto_prompt: bool = True  # Automatically prompt when available
    default_mode: str = "extend"  # "extend" or "mirror"
    enabled: bool = True


class DisplayMonitorService:
    """
    Simple display availability monitor
    
    Polls Screen Mirroring menu and prompts when registered displays
    become available. No proximity detection needed.
    """
    
    def __init__(self, poll_interval_seconds: float = 10.0):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.poll_interval_seconds = poll_interval_seconds
        
        # Monitored displays (user-configured)
        self.monitored_displays: Dict[str, DisplayMonitorConfig] = {}
        
        # State tracking
        self.available_displays: Set[str] = set()
        self.previously_available: Set[str] = set()
        self.user_overrides: Dict[str, datetime] = {}  # Display -> override timestamp
        self.override_duration_minutes = 60  # Don't ask again for 60 min
        
        # Pending prompts
        self.pending_prompt: Optional[str] = None
        self.prompt_timestamp: Optional[datetime] = None
        
        # Statistics
        self.total_polls = 0
        self.total_prompts = 0
        self.total_connections = 0
        
        # Monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        self.logger.info("[DISPLAY MONITOR] Service initialized")
    
    def register_display(
        self,
        display_name: str,
        auto_prompt: bool = True,
        default_mode: str = "extend"
    ):
        """
        Register a display to monitor
        
        Args:
            display_name: Display name (e.g., "Living Room TV")
            auto_prompt: Automatically prompt when available
            default_mode: "extend" or "mirror"
        """
        config = DisplayMonitorConfig(
            display_name=display_name,
            auto_prompt=auto_prompt,
            default_mode=default_mode,
            enabled=True
        )
        
        self.monitored_displays[display_name] = config
        self.logger.info(f"[DISPLAY MONITOR] Registered: {display_name}")
    
    def unregister_display(self, display_name: str):
        """Unregister a monitored display"""
        if display_name in self.monitored_displays:
            del self.monitored_displays[display_name]
            self.logger.info(f"[DISPLAY MONITOR] Unregistered: {display_name}")
    
    async def start_monitoring(self):
        """Start monitoring for available displays"""
        if self.is_monitoring:
            self.logger.warning("[DISPLAY MONITOR] Already monitoring")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("[DISPLAY MONITOR] Started monitoring")
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("[DISPLAY MONITOR] Stopped monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self.is_monitoring:
                await self._poll_available_displays()
                await asyncio.sleep(self.poll_interval_seconds)
        except asyncio.CancelledError:
            self.logger.info("[DISPLAY MONITOR] Monitoring cancelled")
        except Exception as e:
            self.logger.error(f"[DISPLAY MONITOR] Error in monitoring loop: {e}")
    
    async def _poll_available_displays(self):
        """
        Poll for connected displays (HDMI, USB-C, AirPlay)
        
        Detects when new displays become available and generates prompts
        """
        try:
            self.total_polls += 1
            
            # Method 1: Check CONNECTED displays via Core Graphics (HDMI, USB-C, already-connected AirPlay)
            current_available = set()
            
            try:
                from vision.multi_monitor_detector import MultiMonitorDetector
                
                detector = MultiMonitorDetector()
                displays = await detector.detect_displays()
                
                # Add display names (skip primary/built-in display)
                for display in displays:
                    if not display.is_primary:  # Skip built-in MacBook display
                        # Try to get a friendly name
                        display_name = display.name
                        if display_name and display_name != "Primary Display":
                            current_available.add(display_name)
                            self.logger.debug(f"[DISPLAY MONITOR] Found connected display: {display_name}")
                
            except Exception as e:
                self.logger.error(f"[DISPLAY MONITOR] Core Graphics detection failed: {e}")
            
            # Method 2: Also check for AirPlay devices (newly available, not yet connected)
            try:
                from proximity.airplay_discovery import get_airplay_discovery
                
                discovery = get_airplay_discovery()
                devices = await discovery.discover_airplay_devices()
                
                # Extract device names
                for d in devices:
                    if d.is_available:
                        current_available.add(d.device_name)
                        self.logger.debug(f"[DISPLAY MONITOR] Found AirPlay device: {d.device_name}")
                        
            except Exception as e:
                self.logger.error(f"[DISPLAY MONITOR] AirPlay discovery failed: {e}")
            
            # Check for newly available monitored displays
            newly_available = current_available - self.previously_available
            
            for display_name in newly_available:
                if display_name in self.monitored_displays:
                    config = self.monitored_displays[display_name]
                    
                    if config.enabled and config.auto_prompt:
                        # Check user override
                        if not self._is_override_active(display_name):
                            # Generate prompt
                            await self._generate_prompt(display_name, config)
            
            # Update state
            self.previously_available = current_available
            self.available_displays = current_available
            
        except Exception as e:
            self.logger.error(f"[DISPLAY MONITOR] Error polling displays: {e}")
    
    async def _generate_prompt(self, display_name: str, config: DisplayMonitorConfig):
        """
        Generate prompt for display connection
        
        Args:
            display_name: Display name
            config: Display configuration
        """
        try:
            # Skip if already have pending prompt
            if self.pending_prompt:
                return
            
            self.total_prompts += 1
            self.pending_prompt = display_name
            self.prompt_timestamp = datetime.now()
            
            # Generate natural language prompt
            mode = config.default_mode
            prompt = f"Sir, I see {display_name} is now available. Would you like to {mode} your display to it?"
            
            self.logger.info(f"[DISPLAY MONITOR] Generated prompt: {prompt}")
            
            # Return prompt (will be picked up by voice handler)
            return prompt
            
        except Exception as e:
            self.logger.error(f"[DISPLAY MONITOR] Error generating prompt: {e}")
    
    def has_pending_prompt(self) -> bool:
        """Check if there's a pending display prompt"""
        return self.pending_prompt is not None
    
    def get_pending_prompt(self) -> Optional[Dict]:
        """Get current pending prompt"""
        if not self.pending_prompt:
            return None
        
        config = self.monitored_displays.get(self.pending_prompt)
        if not config:
            return None
        
        return {
            "display_name": self.pending_prompt,
            "mode": config.default_mode,
            "prompt": f"Sir, I see {self.pending_prompt} is now available. Would you like to {config.default_mode} your display to it?",
            "timestamp": self.prompt_timestamp.isoformat() if self.prompt_timestamp else None
        }
    
    async def handle_user_response(self, response: str) -> Dict:
        """
        Handle user response to display prompt
        
        Args:
            response: User's voice command (e.g., "yes", "no")
            
        Returns:
            Response result
        """
        try:
            if not self.pending_prompt:
                return {
                    "handled": False,
                    "reason": "No pending prompt"
                }
            
            display_name = self.pending_prompt
            config = self.monitored_displays.get(display_name)
            
            if not config:
                return {
                    "handled": False,
                    "reason": f"Display {display_name} not configured"
                }
            
            # Parse response
            response_lower = response.lower().strip()
            
            # Affirmative responses
            if any(word in response_lower for word in ["yes", "yeah", "yep", "sure", "connect", "extend", "mirror"]):
                # Determine mode (check if user said "mirror" explicitly)
                if "mirror" in response_lower:
                    mode = "mirror"
                else:
                    mode = config.default_mode
                
                result = await self._connect_to_display(display_name, mode)
                self._clear_pending_prompt()
                
                return {
                    "handled": True,
                    "action": "connect",
                    "display_name": display_name,
                    "mode": mode,
                    "result": result
                }
            
            # Negative responses
            elif any(word in response_lower for word in ["no", "nope", "don't", "skip", "not now"]):
                # Register user override
                self._set_user_override(display_name)
                self._clear_pending_prompt()
                
                return {
                    "handled": True,
                    "action": "skip",
                    "display_name": display_name,
                    "response": f"Understood, sir. I won't ask about {display_name} for the next hour."
                }
            
            else:
                # Unclear response
                return {
                    "handled": True,
                    "action": "clarify",
                    "response": "Sir, I didn't quite catch that. Would you like to extend the display? Please say 'yes' or 'no'."
                }
                
        except Exception as e:
            self.logger.error(f"[DISPLAY MONITOR] Error handling response: {e}")
            self._clear_pending_prompt()
            return {
                "handled": False,
                "error": str(e)
            }
    
    async def _connect_to_display(self, display_name: str, mode: str) -> Dict:
        """
        Connect to display via AirPlay
        
        Args:
            display_name: Display name
            mode: "extend" or "mirror"
            
        Returns:
            Connection result
        """
        try:
            self.logger.info(f"[DISPLAY MONITOR] Connecting to {display_name} (mode: {mode})")
            
            from proximity.airplay_discovery import get_airplay_discovery
            
            discovery = get_airplay_discovery()
            result = await discovery.connect_to_airplay_device(display_name, mode)
            
            if result.get("success"):
                self.total_connections += 1
                self.logger.info(f"[DISPLAY MONITOR] Connected to {display_name}")
                
                return {
                    "success": True,
                    "response": f"Extending to {display_name}... Done, sir."
                }
            else:
                return {
                    "success": False,
                    "response": f"I encountered an issue connecting to {display_name}. Please try manually."
                }
                
        except Exception as e:
            self.logger.error(f"[DISPLAY MONITOR] Connection error: {e}")
            return {
                "success": False,
                "response": f"Error connecting: {str(e)}"
            }
    
    def _set_user_override(self, display_name: str):
        """Set user override (don't ask again for a while)"""
        self.user_overrides[display_name] = datetime.now()
        self.logger.info(f"[DISPLAY MONITOR] User override set for {display_name}")
    
    def _is_override_active(self, display_name: str) -> bool:
        """Check if user override is still active"""
        if display_name not in self.user_overrides:
            return False
        
        override_time = self.user_overrides[display_name]
        elapsed = (datetime.now() - override_time).total_seconds() / 60
        
        if elapsed > self.override_duration_minutes:
            # Override expired
            del self.user_overrides[display_name]
            return False
        
        return True
    
    def _clear_pending_prompt(self):
        """Clear pending prompt"""
        self.pending_prompt = None
        self.prompt_timestamp = None
    
    def get_stats(self) -> Dict:
        """Get service statistics"""
        return {
            "total_polls": self.total_polls,
            "total_prompts": self.total_prompts,
            "total_connections": self.total_connections,
            "monitored_displays": len(self.monitored_displays),
            "available_displays": len(self.available_displays),
            "available_display_names": list(self.available_displays),
            "active_overrides": len(self.user_overrides),
            "is_monitoring": self.is_monitoring,
            "has_pending_prompt": self.has_pending_prompt()
        }


# Singleton instance
_display_monitor: Optional[DisplayMonitorService] = None

def get_display_monitor(poll_interval_seconds: float = 10.0) -> DisplayMonitorService:
    """Get singleton DisplayMonitorService instance"""
    global _display_monitor
    if _display_monitor is None:
        _display_monitor = DisplayMonitorService(poll_interval_seconds)
    return _display_monitor
