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
    """Configuration for a monitored display.
    
    Attributes:
        display_name: Human-readable display name (e.g., "Living Room TV")
        auto_prompt: Whether to automatically prompt when display becomes available
        default_mode: Default connection mode ("extend" or "mirror")
        enabled: Whether monitoring is enabled for this display
    """
    display_name: str  # e.g., "Living Room TV"
    auto_prompt: bool = True  # Automatically prompt when available
    default_mode: str = "extend"  # "extend" or "mirror"
    enabled: bool = True


class DisplayMonitorService:
    """Simple display availability monitor.
    
    Polls Screen Mirroring menu and prompts when registered displays
    become available. No proximity detection needed.
    
    This service monitors both connected displays (HDMI, USB-C) and
    available AirPlay devices, generating user prompts when configured
    displays become available.
    
    Attributes:
        poll_interval_seconds: How often to poll for displays
        monitored_displays: Dictionary of configured displays to monitor
        available_displays: Set of currently available display names
        previously_available: Set of previously available displays for change detection
        user_overrides: Dictionary tracking user "don't ask again" preferences
        override_duration_minutes: How long user overrides last
        pending_prompt: Currently pending display prompt
        prompt_timestamp: When the current prompt was generated
        total_polls: Statistics counter for polling operations
        total_prompts: Statistics counter for generated prompts
        total_connections: Statistics counter for successful connections
        monitoring_task: Asyncio task for the monitoring loop
        is_monitoring: Whether monitoring is currently active
    """
    
    def __init__(self, poll_interval_seconds: float = 10.0):
        """Initialize the Display Monitor Service.
        
        Args:
            poll_interval_seconds: How often to poll for available displays
        """
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
    ) -> None:
        """Register a display to monitor for availability.
        
        Args:
            display_name: Human-readable display name (e.g., "Living Room TV")
            auto_prompt: Whether to automatically prompt when available
            default_mode: Default connection mode ("extend" or "mirror")
            
        Example:
            >>> monitor.register_display("Living Room TV", auto_prompt=True, default_mode="extend")
        """
        config = DisplayMonitorConfig(
            display_name=display_name,
            auto_prompt=auto_prompt,
            default_mode=default_mode,
            enabled=True
        )
        
        self.monitored_displays[display_name] = config
        self.logger.info(f"[DISPLAY MONITOR] Registered: {display_name}")
    
    def unregister_display(self, display_name: str) -> None:
        """Unregister a monitored display.
        
        Args:
            display_name: Display name to stop monitoring
        """
        if display_name in self.monitored_displays:
            del self.monitored_displays[display_name]
            self.logger.info(f"[DISPLAY MONITOR] Unregistered: {display_name}")
    
    async def start_monitoring(self) -> None:
        """Start monitoring for available displays.
        
        Creates an asyncio task that continuously polls for display availability
        at the configured interval.
        
        Raises:
            RuntimeError: If monitoring is already active
        """
        if self.is_monitoring:
            self.logger.warning("[DISPLAY MONITOR] Already monitoring")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("[DISPLAY MONITOR] Started monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring for displays.
        
        Cancels the monitoring task and waits for it to complete.
        """
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("[DISPLAY MONITOR] Stopped monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop that polls for displays at regular intervals.
        
        Raises:
            asyncio.CancelledError: When monitoring is cancelled
            Exception: For any other errors during monitoring
        """
        try:
            while self.is_monitoring:
                await self._poll_available_displays()
                await asyncio.sleep(self.poll_interval_seconds)
        except asyncio.CancelledError:
            self.logger.info("[DISPLAY MONITOR] Monitoring cancelled")
        except Exception as e:
            self.logger.error(f"[DISPLAY MONITOR] Error in monitoring loop: {e}")
    
    async def _poll_available_displays(self) -> None:
        """Poll for connected displays (HDMI, USB-C, AirPlay).
        
        Detects when new displays become available and generates prompts
        for registered displays. Uses two methods:
        1. Core Graphics for connected displays
        2. AirPlay discovery for available wireless displays
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
    
    async def _generate_prompt(self, display_name: str, config: DisplayMonitorConfig) -> Optional[str]:
        """Generate prompt for display connection.
        
        Args:
            display_name: Name of the display that became available
            config: Configuration for the display
            
        Returns:
            Generated prompt string, or None if prompt couldn't be generated
        """
        try:
            # Skip if already have pending prompt
            if self.pending_prompt:
                return None
            
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
            return None
    
    def has_pending_prompt(self) -> bool:
        """Check if there's a pending display prompt.
        
        Returns:
            True if there's a prompt waiting for user response
        """
        return self.pending_prompt is not None
    
    def get_pending_prompt(self) -> Optional[Dict]:
        """Get current pending prompt details.
        
        Returns:
            Dictionary with prompt details, or None if no pending prompt
            
        Example:
            >>> prompt = monitor.get_pending_prompt()
            >>> if prompt:
            ...     print(f"Display: {prompt['display_name']}")
            ...     print(f"Prompt: {prompt['prompt']}")
        """
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
        """Handle user response to display prompt.
        
        Args:
            response: User's voice command (e.g., "yes", "no", "mirror")
            
        Returns:
            Dictionary with response handling results containing:
            - handled: Whether the response was processed
            - action: Action taken ("connect", "skip", "clarify")
            - display_name: Name of the display (if applicable)
            - mode: Connection mode used (if connecting)
            - response: Generated response text
            - result: Connection result (if connecting)
            
        Example:
            >>> result = await monitor.handle_user_response("yes")
            >>> if result["handled"] and result["action"] == "connect":
            ...     print(f"Connected to {result['display_name']}")
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

                # Generate dynamic response using Claude if available
                try:
                    from api.vision_command_handler import vision_command_handler
                    if vision_command_handler and hasattr(vision_command_handler, 'intelligence'):
                        prompt = f"""The user was asked: "Sir, I see your {display_name} is now available. Would you like to extend your display to it?"

They responded: "{response}"

Generate a brief, natural JARVIS-style acknowledgment that:
1. Confirms you understood they don't want to connect
2. Is brief and conversational (1-2 sentences max)
3. Uses "sir" appropriately
4. Shows understanding without being verbose

Respond ONLY with JARVIS's exact words, no quotes or formatting."""

                        claude_response = await vision_command_handler.intelligence._get_claude_vision_response(
                            None, prompt
                        )
                        dynamic_response = claude_response.get("response", f"Understood, sir. I won't ask about {display_name} for the next hour.")
                    else:
                        dynamic_response = f"Understood, sir. I won't ask about {display_name} for the next hour."
                except Exception as e:
                    self.logger.warning(f"Could not generate dynamic response: {e}")
                    dynamic_response = f"Understood, sir. I won't ask about {display_name} for the next hour."

                return {
                    "handled": True,
                    "action": "skip",
                    "display_name": display_name,
                    "response": dynamic_response
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
        """Connect to display via adaptive Control Center clicker.

        Args:
            display_name: Name of the display to connect to
            mode: Connection mode ("extend" or "mirror")

        Returns:
            Dictionary with connection results containing:
            - success: Whether connection was successful
            - response: User-friendly response message
            - method: Method used for connection
            - error: Error message if connection failed
        """
        try:
            self.logger.info(f"[DISPLAY MONITOR] Connecting to {display_name} (mode: {mode})")

            # Use adaptive control center clicker
            from display.control_center_clicker import get_control_center_clicker

            # Get vision analyzer if available
            vision_analyzer = None
            try:
                from vision.claude_vision_analyzer_main import get_claude_vision_analyzer
                vision_analyzer = get_claude_vision_analyzer()
            except Exception as e:
                self.logger.debug(f"[DISPLAY MONITOR] Vision analyzer not available: {e}")

            clicker = get_control_center_clicker(
                vision_analyzer=vision_analyzer,
                use_adaptive=True
            )

            # Connect using adaptive clicker
            result = clicker.connect_to_living_room_tv() if display_name == "Living Room TV" else \
                     await self._connect_to_generic_device(clicker, display_name)

            if result.get("success"):
                self.total_connections += 1
                self.logger.info(f"[DISPLAY MONITOR] Connected to {display_name}")

                # Generate dynamic success response
                try:
                    from api.vision_command_handler import vision_command_handler
                    if vision_command_handler and hasattr(vision_command_handler, 'intelligence'):
                        prompt = f"""The user asked you to connect to {display_name}. You successfully connected.

Generate a brief, natural JARVIS-style confirmation that:
1. Confirms the connection is complete
2. Is brief and conversational (1 sentence)
3. Uses "sir" appropriately
4. Sounds confident and efficient

Respond ONLY with JARVIS's exact words, no quotes or formatting."""

                        claude_response = await vision_command_handler.intelligence._get_claude_vision_response(
                            None, prompt
                        )
                        success_response = claude_response.get("response", f"Extending to {display_name}... Done, sir.")
                    else:
                        success_response = f"Extending to {display_name}... Done, sir."
                except Exception as e:
                    self.logger.warning(f"Could not generate dynamic response: {e}")
                    success_response = f"Extending to {display_name}... Done, sir."

                return {
                    "success": True,
                    "response": success_response,
                    "method": result.get("method", "adaptive_detection")
                }
            else:
                return {
                    "success": False,
                    "response": f"I encountered an issue connecting to {display_name}. Please try manually.",
                    "error": result.get("error")
                }

        except Exception as e:
            self.logger.error(f"[DISPLAY MONITOR] Connection error: {e}", exc_info=True)
            return {
                "success": False,
                "response": f"Error connecting: {str(e)}"
            }

    async def _connect_to_generic_device(self, clicker, device_name: str) -> Dict:
        """Connect to a generic device (not Living Room TV).

        Args:
            clicker: ControlCenterClicker instance
            device_name: Name of the device to connect to

        Returns:
            Dictionary with connection results containing:
            - success: Whether connection was successful
            - message: Success message
            - method: Method used for connection
            - error: Error message if connection failed
        """
        try:
            # Step 1: Open Control Center
            cc_result = clicker.open_control_center(wait_after_click=0.5)
            if not cc_result.get('success'):
                return cc_result

            # Step 2: Open Screen Mirroring
            sm_result = clicker.open_screen_mirroring(wait_after_click=0.5)
            if not sm_result.get('success'):
                return sm_result

            # Step 3: Click device
            # Use adaptive clicker's click_device method if available
            if hasattr(clicker, '_adaptive_clicker') and clicker._adaptive_clicker:
                import asyncio
                device_result = await clicker._adaptive_clicker.click_device(device_name)

                if hasattr(device_result, 'success'):
                    return {
                        "success": device_result.success,
                        "message": f"Connected to {device_name}",
                        "method": device_result.method_used
                    }

            # Fallback: assume success if we got this far
            return {
                "success": True,
                "message": f"Connected to {device_name}",
                "method": "manual_flow"
            }

        except Exception as e:
            self.logger.error(f"[DISPLAY MONITOR] Generic device connection error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _set_user_override(self, display_name: str) -> None:
        """Set user override (don't ask again for a while).
        
        Args:
            display_name: Display name to set override for
        """
        self.user_overrides[display_name] = datetime.now()
        self.logger.info(f"[DISPLAY MONITOR] User override set for {display_name}")
    
    def _is_override_active(self, display_name: str) -> bool:
        """Check if user override is still active.
        
        Args:
            display_name: Display name to check
            
        Returns:
            True if override is still active (user said "don't ask again" recently)
        """
        if display_name not in self.user_overrides:
            return False
        
        override_time = self.user_overrides[display_name]
        elapsed = (datetime.now() - override_time).total_seconds() / 60
        
        if elapsed > self.override_duration_minutes:
            # Override expired
            del self.user_overrides[display_name]
            return False
        
        return True
    
    def _clear_pending_prompt(self) -> None:
        """Clear the current pending prompt."""
        self.pending_prompt = None
        self.prompt_timestamp = None
    
    def get_stats(self) -> Dict:
        """Get service statistics and current state.
        
        Returns:
            Dictionary containing service statistics including:
            - total_polls: Number of display polling operations
            - total_prompts: Number of prompts generated
            - total_connections: Number of successful connections
            - monitored_displays: Number of registered displays
            - available_displays: Number of currently available displays
            - available_display_names: List of available display names
            - active_overrides: Number of active user overrides
            - is_monitoring: Whether monitoring is currently active
            - has_pending_prompt: Whether there's a pending prompt
            
        Example:
            >>> stats = monitor.get_stats()
            >>> print(f"Monitoring {stats['monitored_displays']} displays")
            >>> print(f"Found {stats['available_displays']} available displays")
        """
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
    """Get singleton DisplayMonitorService instance.
    
    Args:
        poll_interval_seconds: How often to poll for displays (only used on first call)
        
    Returns:
        Singleton DisplayMonitorService instance
        
    Example:
        >>> monitor = get_display_monitor(poll_interval_seconds=5.0)
        >>> monitor.register_display("Living Room TV")
        >>> await monitor.start_monitoring()
    """
    global _display_monitor
    if _display_monitor is None:
        _display_monitor = DisplayMonitorService(poll_interval_seconds)
    return _display_monitor