#!/usr/bin/env python3
"""
Advanced Display Monitor for JARVIS
====================================

Production-ready, fully async, dynamic display monitoring system with:
- Zero hardcoding (all configuration-driven)
- Multi-method detection (AppleScript, Core Graphics, Yabai)
- Voice integration
- Smart caching
- Robust error handling
- Multi-monitor support
- Event-driven architecture

Author: Derek Russell
Date: 2025-10-15
Version: 2.0
"""

import asyncio
import json
import logging
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class DetectionMethod(Enum):
    """Available detection methods for display discovery.
    
    Attributes:
        APPLESCRIPT: Uses AppleScript to query Control Center menu
        DNSSD: Uses DNS-SD (Bonjour) to discover AirPlay devices
        COREGRAPHICS: Uses Core Graphics API for display enumeration
        YABAI: Uses Yabai window manager for display information
    """

    APPLESCRIPT = "applescript"
    DNSSD = "dnssd"
    COREGRAPHICS = "coregraphics"
    YABAI = "yabai"


class DisplayType(Enum):
    """Types of displays that can be detected.
    
    Attributes:
        AIRPLAY: AirPlay-enabled displays (Apple TV, etc.)
        HDMI: HDMI-connected displays
        THUNDERBOLT: Thunderbolt/DisplayPort displays
        USB_C: USB-C connected displays
        WIRELESS: Wireless displays (Miracast, etc.)
        UNKNOWN: Unidentified display type
    """

    AIRPLAY = "airplay"
    HDMI = "hdmi"
    THUNDERBOLT = "thunderbolt"
    USB_C = "usb_c"
    WIRELESS = "wireless"
    UNKNOWN = "unknown"


class ConnectionMode(Enum):
    """Display connection modes.
    
    Attributes:
        EXTEND: Extend desktop to display
        MIRROR: Mirror primary display
    """

    EXTEND = "extend"
    MIRROR = "mirror"


@dataclass
class DisplayInfo:
    """Information about a detected display.
    
    Attributes:
        id: Unique identifier for the display
        name: Human-readable display name
        display_type: Type of display connection
        is_available: Whether display is currently available
        is_connected: Whether display is currently connected
        detection_method: Method used to detect this display
        detected_at: Timestamp when display was detected
        metadata: Additional display metadata
    """

    id: str
    name: str
    display_type: DisplayType
    is_available: bool
    is_connected: bool
    detection_method: DetectionMethod
    detected_at: datetime
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict:
        """Convert DisplayInfo to dictionary format.
        
        Returns:
            Dictionary representation with serialized enums and datetime
        """
        data = asdict(self)
        data["display_type"] = self.display_type.value
        data["detection_method"] = self.detection_method.value
        data["detected_at"] = self.detected_at.isoformat()
        return data


@dataclass
class MonitoredDisplay:
    """Configuration for a monitored display.
    
    Attributes:
        id: Unique identifier for the display
        name: Display name to match against
        display_type: Type of display connection
        aliases: Alternative names that match this display
        auto_connect: Whether to automatically connect when detected
        auto_prompt: Whether to prompt user when detected
        connection_mode: Default connection mode (extend/mirror)
        priority: Connection priority (higher = more important)
        enabled: Whether monitoring is enabled for this display
    """

    id: str
    name: str
    display_type: str
    aliases: List[str]
    auto_connect: bool
    auto_prompt: bool
    connection_mode: str
    priority: int
    enabled: bool

    def matches(self, display_name: str) -> bool:
        """Check if display name matches this configuration.
        
        Args:
            display_name: Name of detected display to match
            
        Returns:
            True if display name matches this configuration
        """
        if display_name.lower() == self.name.lower():
            return True
        return any(alias.lower() == display_name.lower() for alias in self.aliases)


class DisplayCache:
    """Cache for display detection results to reduce API calls.
    
    Attributes:
        ttl_seconds: Time-to-live for cached entries in seconds
    """

    def __init__(self, ttl_seconds: int = 5):
        """Initialize display cache.
        
        Args:
            ttl_seconds: Cache expiration time in seconds
        """
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, tuple[List[str], datetime]] = {}

    def get(self, key: str) -> Optional[List[str]]:
        """Get cached value if not expired.
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            Cached value if valid, None if expired or not found
        """
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: List[str]):
        """Set cache value with current timestamp.
        
        Args:
            key: Cache key to store
            value: List of display names to cache
        """
        self._cache[key] = (value, datetime.now())

    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()


class AppleScriptDetector:
    """AppleScript-based display detection using Control Center menu.
    
    This detector queries the Screen Mirroring menu in Control Center
    to find available AirPlay displays.
    
    Attributes:
        config: Configuration dictionary for this detector
        timeout: Script execution timeout in seconds
        retry_attempts: Number of retry attempts on failure
        retry_delay: Delay between retry attempts in seconds
        filter_items: System items to filter out from results
    """

    def __init__(self, config: Dict):
        """Initialize AppleScript detector.
        
        Args:
            config: Configuration dictionary containing timeout, retry settings
        """
        self.config = config
        self.timeout = config.get("timeout_seconds", 5.0)
        self.retry_attempts = config.get("retry_attempts", 3)
        self.retry_delay = config.get("retry_delay_seconds", 0.5)
        self.filter_items = config.get("filter_system_items", [])

    async def detect_displays(self) -> List[str]:
        """Detect available displays using AppleScript.
        
        Queries the Screen Mirroring menu in Control Center to find
        available AirPlay displays.
        
        Returns:
            List of available display names
            
        Raises:
            asyncio.TimeoutError: If script execution times out
        """
        for attempt in range(self.retry_attempts):
            try:
                script = """
                tell application "System Events"
                    tell process "ControlCenter"
                        try
                            set mirroringMenu to menu 1 of menu bar item "Screen Mirroring" of menu bar 1
                            set menuItems to name of menu items of mirroringMenu

                            set deviceNames to {}
                            repeat with itemName in menuItems
                                set end of deviceNames to itemName as text
                            end repeat

                            return deviceNames
                        on error errMsg
                            return "ERROR:" & errMsg
                        end try
                    end tell
                end tell
                """

                result = await asyncio.create_subprocess_exec(
                    "osascript",
                    "-e",
                    script,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=self.timeout)

                if result.returncode == 0:
                    output = stdout.decode("utf-8").strip()

                    if output.startswith("ERROR:"):
                        logger.debug(f"AppleScript error: {output}")
                        continue

                    # Parse the output
                    devices = []
                    if output:
                        # AppleScript returns comma-separated values
                        raw_devices = [d.strip() for d in output.split(", ")]
                        devices = [d for d in raw_devices if d and d not in self.filter_items]

                    logger.debug(f"AppleScript detected: {devices}")
                    return devices

            except asyncio.TimeoutError:
                logger.warning(f"AppleScript timeout (attempt {attempt + 1}/{self.retry_attempts})")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"AppleScript error: {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay)

        return []

    async def connect_display(self, display_name: str) -> Dict[str, Any]:
        """Connect to a display using AppleScript.
        
        Uses AppleScript to click through the Control Center menu
        to connect to the specified display.
        
        Args:
            display_name: Name of display to connect to
            
        Returns:
            Connection result with success status and message
            
        Raises:
            asyncio.TimeoutError: If connection times out
        """
        try:
            script = f"""
            tell application "System Events"
                tell process "ControlCenter"
                    try
                        click menu bar item "Screen Mirroring" of menu bar 1
                        delay 0.3

                        click menu item "{display_name}" of menu 1 of menu bar item "Screen Mirroring" of menu bar 1
                        delay 0.2

                        return "SUCCESS"
                    on error errMsg
                        return "ERROR:" & errMsg
                    end try
                end tell
            end tell
            """

            result = await asyncio.create_subprocess_exec(
                "osascript",
                "-e",
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                result.communicate(), timeout=self.timeout * 2  # Connection takes longer
            )

            output = stdout.decode("utf-8").strip()

            if "SUCCESS" in output:
                return {"success": True, "message": f"Connected to {display_name}"}
            else:
                error_detail = output.replace("ERROR:", "").strip()
                return {
                    "success": False,
                    "message": f"Failed to connect to {display_name}",
                    "error": error_detail,
                }

        except asyncio.TimeoutError:
            return {"success": False, "message": "Connection timeout"}
        except Exception as e:
            return {"success": False, "message": str(e)}


class DNSSDDetector:
    """DNS-SD (Bonjour) based AirPlay display detection for macOS Sequoia+.
    
    Uses the dns-sd command to discover AirPlay devices on the network
    via Bonjour service discovery.
    
    Attributes:
        config: Configuration dictionary for this detector
        timeout: Discovery timeout in seconds
        service_type: Bonjour service type to search for
        exclude_local: Whether to exclude local device from results
    """

    def __init__(self, config: Dict):
        """Initialize DNS-SD detector.
        
        Args:
            config: Configuration dictionary with timeout and service settings
        """
        self.config = config
        self.timeout = config.get("timeout_seconds", 5.0)
        self.service_type = config.get("service_type", "_airplay._tcp")
        self.exclude_local = config.get("exclude_local_device", True)

    async def detect_displays(self) -> List[str]:
        """Detect AirPlay displays using dns-sd command.
        
        Runs dns-sd browsing for the configured service type and
        parses the output to extract device names.
        
        Returns:
            List of discovered AirPlay device names
            
        Raises:
            FileNotFoundError: If dns-sd command is not available
        """
        try:
            # Start dns-sd browsing in background
            process = await asyncio.create_subprocess_exec(
                "dns-sd",
                "-B",
                self.service_type,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Let it run for timeout seconds to collect results
            await asyncio.sleep(self.timeout)

            # Kill the process
            process.terminate()
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=2.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                stdout, stderr = b"", b""

            if stdout:
                output = stdout.decode("utf-8", errors="ignore")

                # Parse dns-sd output
                # Format: "Timestamp A/R Flags if Domain Service Type Instance Name"
                # We want the Instance Name column
                devices = []
                for line in output.split("\n"):
                    if "Add" in line and self.service_type in line:
                        # Split by multiple spaces and get last part (instance name)
                        parts = [p for p in line.split("  ") if p.strip()]
                        if len(parts) >= 3:
                            instance_name = parts[-1].strip()

                            # Filter out local device if configured
                            if self.exclude_local and "MacBook" in instance_name:
                                continue

                            if instance_name and instance_name not in devices:
                                devices.append(instance_name)

                logger.debug(f"DNS-SD detected: {devices}")
                return devices

            return []

        except FileNotFoundError:
            logger.warning("dns-sd command not found")
            return []
        except Exception as e:
            logger.error(f"DNS-SD detection error: {e}")
            return []


class CoreGraphicsDetector:
    """Core Graphics-based display detection using Quartz framework.
    
    Uses the macOS Core Graphics API to enumerate connected displays.
    Requires the Quartz Python bindings to be available.
    
    Attributes:
        config: Configuration dictionary for this detector
        max_displays: Maximum number of displays to query
        exclude_builtin: Whether to exclude built-in displays
    """

    def __init__(self, config: Dict):
        """Initialize Core Graphics detector.
        
        Args:
            config: Configuration dictionary with display limits and filters
        """
        self.config = config
        self.max_displays = config.get("max_displays", 32)
        self.exclude_builtin = config.get("exclude_builtin", True)

    async def detect_displays(self) -> List[str]:
        """Detect displays using Core Graphics API.
        
        Queries the Core Graphics display list to find connected displays.
        
        Returns:
            List of display names (generated from display IDs)
            
        Raises:
            ImportError: If Quartz framework is not available
        """
        try:
            import Quartz

            # Get online displays
            result = Quartz.CGGetOnlineDisplayList(self.max_displays, None, None)

            if result[0] != 0:  # Error
                logger.error(f"CoreGraphics error: {result[0]}")
                return []

            display_ids = result[1]
            display_count = result[2]

            displays = []
            for display_id in display_ids[:display_count]:
                is_builtin = Quartz.CGDisplayIsBuiltin(display_id)

                if self.exclude_builtin and is_builtin:
                    continue

                # Try to get display name (if available)
                display_name = f"External Display {display_id}"
                displays.append(display_name)

            logger.debug(f"CoreGraphics detected: {displays}")
            return displays

        except ImportError:
            logger.warning("CoreGraphics (Quartz) not available")
            return []
        except Exception as e:
            logger.error(f"CoreGraphics error: {e}")
            return []


class YabaiDetector:
    """Yabai-based display detection using window manager queries.
    
    Uses the Yabai tiling window manager to query display information.
    Requires Yabai to be installed and running.
    
    Attributes:
        config: Configuration dictionary for this detector
        timeout: Command execution timeout in seconds
    """

    def __init__(self, config: Dict):
        """Initialize Yabai detector.
        
        Args:
            config: Configuration dictionary with timeout settings
        """
        self.config = config
        self.timeout = config.get("command_timeout", 3.0)

    async def detect_displays(self) -> List[str]:
        """Detect displays using Yabai window manager.
        
        Executes 'yabai -m query --displays' to get display information.
        
        Returns:
            List of display names (generated from display indices/IDs)
            
        Raises:
            FileNotFoundError: If Yabai is not installed
            asyncio.TimeoutError: If command execution times out
        """
        try:
            result = await asyncio.create_subprocess_exec(
                "yabai",
                "-m",
                "query",
                "--displays",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=self.timeout)

            if result.returncode == 0:
                displays_data = json.loads(stdout.decode("utf-8"))
                displays = [f"Display {d.get('index', d.get('id'))}" for d in displays_data]
                logger.debug(f"Yabai detected: {displays}")
                return displays

            return []

        except FileNotFoundError:
            logger.debug("Yabai not installed")
            return []
        except asyncio.TimeoutError:
            logger.warning("Yabai timeout")
            return []
        except Exception as e:
            logger.error(f"Yabai error: {e}")
            return []


class AdvancedDisplayMonitor:
    """
    Advanced display monitoring system with multi-method detection and voice integration.

    Features:
    - Multi-method detection (AppleScript, CoreGraphics, Yabai, DNS-SD)
    - Smart caching with configurable TTL
    - Voice integration for announcements and prompts
    - Event-driven callbacks for display state changes
    - Graceful degradation when detection methods fail
    - Zero hardcoding - all configuration-driven
    - Circuit breaker pattern for connection management
    - Real-time display state verification
    - Learning pattern storage for AI improvements

    Attributes:
        config: Configuration dictionary loaded from JSON
        voice_handler: Voice integration handler for TTS
        vision_analyzer: Vision analyzer for AI-powered UI detection
        websocket_manager: WebSocket manager for UI notifications
        cache: Display detection result cache
        detectors: Dictionary of detection method instances
        monitored_displays: List of configured displays to monitor
        is_monitoring: Whether monitoring is currently active
        monitoring_task: Async task for the monitoring loop
        available_displays: Set of currently available display IDs
        connected_displays: Set of currently connected display IDs
        connecting_displays: Set of displays currently being connected (circuit breaker)
        initial_scan_complete: Whether initial startup scan is complete
        pending_prompt_display: Display ID with pending user prompt
        callbacks: Event callback registry
    """

    def __init__(self, config_path: Optional[str] = None, voice_handler=None, vision_analyzer=None):
        """
        Initialize advanced display monitor.

        Args:
            config_path: Path to configuration JSON file. If None, uses default path
            voice_handler: Voice integration handler for TTS announcements
            vision_analyzer: Vision analyzer for AI-powered UI detection
            
        Raises:
            FileNotFoundError: If configuration file is not found
            json.JSONDecodeError: If configuration file contains invalid JSON
        """
        self.config = self._load_config(config_path)
        self.voice_handler = voice_handler
        self.vision_analyzer = vision_analyzer  # Store vision analyzer for UAE integration
        self.websocket_manager = None  # Will be set by main.py

        # Initialize components
        self.cache = DisplayCache(ttl_seconds=self.config["caching"]["display_list_ttl_seconds"])

        # Initialize detectors
        self.detectors = {}
        self._init_detectors()

        # Monitored displays configuration
        self.monitored_displays = self._load_monitored_displays()

        # State tracking
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.available_displays: Set[str] = set()
        self.connected_displays: Set[str] = set()
        self.connecting_displays: Set[str] = (
            set()
        )  # Circuit breaker: displays currently being connected
        self.initial_scan_complete = False  # Track if initial scan is done
        self.pending_prompt_display: Optional[str] = (
            None  # Track which display has a pending prompt
        )

        # Event callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            "display_detected": [],
            "display_lost": [],
            "display_connected": [],
            "display_disconnected": [],
            "error": [],
        }

        logger.info(
            f"[DISPLAY MONITOR] Initialized with {len(self.monitored_displays)} monitored displays"
        )

    def set_websocket_manager(self, ws_manager):
        """Set WebSocket manager for UI notifications.
        
        Args:
            ws_manager: WebSocket manager instance for broadcasting events
        """
        self.websocket_manager = ws_manager
        logger.info("[DISPLAY MONITOR] WebSocket manager set")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file, or None for default
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            json.JSONDecodeError: If configuration file contains invalid JSON
        """
        if config_path is None:
            # Default config path
            config_path = Path(__file__).parent.parent / "config" / "display_monitor_config.json"

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info(f"[DISPLAY MONITOR] Loaded config from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"[DISPLAY MONITOR] Config file not found: {config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"[DISPLAY MONITOR] Invalid JSON in config: {e}")
            raise

    def _init_detectors(self):
        """Initialize detection method instances based on configuration.
        
        Creates detector instances for each enabled detection method
        specified in the configuration.
        """
        methods = self.config["display_monitoring"]["detection_methods"]

        if "applescript" in methods and self.config.get("applescript", {}).get("enabled", False):
            self.detectors[DetectionMethod.APPLESCRIPT] = AppleScriptDetector(
                self.config["applescript"]
            )

        if "dnssd" in methods and self.config.get("dnssd", {}).get("enabled", False):
            self.detectors[DetectionMethod.DNSSD] = DNSSDDetector(self.config["dnssd"])

        if "coregraphics" in methods and self.config.get("coregraphics", {}).get("enabled", False):
            self.detectors[DetectionMethod.COREGRAPHICS] = CoreGraphicsDetector(
                self.config["coregraphics"]
            )

        if "yabai" in methods and self.config.get("yabai", {}).get("enabled", False):
            self.detectors[DetectionMethod.YABAI] = YabaiDetector(self.config["yabai"])

        logger.info(f"[DISPLAY MONITOR] Initialized {len(self.detectors)} detection methods")

    def _load_monitored_displays(self) -> List[MonitoredDisplay]:
        """Load monitored displays from configuration.
        
        Returns:
            List of MonitoredDisplay instances for enabled displays
        """
        displays = []
        for display_config in self.config["displays"]["monitored_displays"]:
            if display_config.get("enabled", True):
                displays.append(MonitoredDisplay(**display_config))
        return displays

    def register_callback(self, event: str, callback: Callable):
        """
        Register event callback for display state changes.

        Args:
            event: Event name (display_detected, display_lost, display_connected, 
                  display_disconnected, error)
            callback: Async callback function to execute on event
            
        Example:
            >>> async def on_display_detected(display, detected_name):
            ...     print(f"Found {display.name}")
            >>> monitor.register_callback("display_detected", on_display_detected)
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            logger.debug(f"[DISPLAY MONITOR] Registered callback for {event}")

    async def _emit_event(self, event: str, **kwargs):
        """Emit event to all registered callbacks.
        
        Args:
            event: Event name to emit
            **kwargs: Event data to pass to callbacks
        """
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(**kwargs)
                    else:
                        callback(**kwargs)
                except Exception as e:
                    logger.error(f"[DISPLAY MONITOR] Callback error for {event}: {e}")

    async def start(self):
        """Start display monitoring.
        
        Begins the monitoring loop with optional startup delay.
        Does nothing if monitoring is already active or disabled in config.
        """
        if self.is_monitoring:
            logger.warning("[DISPLAY MONITOR] Already monitoring")
            return

        if not self.config["display_monitoring"]["enabled"]:
            logger.warning("[DISPLAY MONITOR] Monitoring disabled in config")
            return

        # Startup delay
        startup_delay = self.config["display_monitoring"]["startup_delay_seconds"]
        if startup_delay > 0:
            logger.info(f"[DISPLAY MONITOR] Starting in {startup_delay}s...")
            await asyncio.sleep(startup_delay)

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitor_loop())
        logger.info("[DISPLAY MONITOR] Started monitoring")

    async def stop(self):
        """Stop display monitoring.
        
        Cancels the monitoring task and clears cache and state.
        """
        self.is_monitoring = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        self.cache.clear()
        self.initial_scan_complete = False  # Reset for next start
        logger.info("[DISPLAY MONITOR] Stopped monitoring")

    async def _monitor_loop(self):
        """Main monitoring loop that periodically checks for display changes.
        
        Runs continuously while monitoring is enabled, checking for display
        state changes at the configured interval.
        
        Raises:
            asyncio.CancelledError: When monitoring is cancelled
        """
        check_interval = self.config["display_monitoring"]["check_interval_seconds"]
        logger.info(f"[DISPLAY MONITOR] Monitor loop starting (interval: {check_interval}s)")

        try:
            while self.is_monitoring:
                await self._check_displays()
                logger.debug(f"[DISPLAY MONITOR] Check complete, sleeping {check_interval}s...")
                await asyncio.sleep(check_interval)
        except asyncio.CancelledError:
            logger.info("[DISPLAY MONITOR] Monitoring cancelled")
        except Exception as e:
            logger.error(f"[DISPLAY MONITOR] Error in monitoring loop: {e}")
            await self._emit_event("error", error=e)

    async def _check_displays(self):
        """Check for available displays and handle state changes.
        
        Detects currently available displays, matches them against monitored
        displays, and handles new detections and losses. Manages initial scan
        completion and startup announcements.
        """
        try:
            # Detect all available displays
            detected_displays = await self._detect_all_displays()
            logger.debug(
                f"[DISPLAY MONITOR] Check: detected {len(detected_displays)} displays: {detected_displays}"
            )

            # Match against monitored displays
            current_available = set()
            for display_name in detected_displays:
                for monitored in self.monitored_displays:
                    if monitored.matches(display_name):
                        current_available.add(monitored.id)
                        logger.info(
                            f"[DISPLAY MONITOR] MATCH: '{display_name}' → '{monitored.name}' (id: {monitored.id}), in_available={monitored.id in self.available_displays}, initial_complete={self.initial_scan_complete}"
                        )

                        # New display detected - announce and set pending prompt
                        if monitored.id not in self.available_displays:
                            logger.info(f"[DISPLAY MONITOR] NEW DISPLAY DETECTED: {monitored.name}")
                            if self.initial_scan_complete:
                                # Display became available after initial scan - announce it!
                                logger.info(
                                    f"[DISPLAY MONITOR] STATE CHANGE: {monitored.name} became AVAILABLE"
                                )
                                await self._handle_display_detected(monitored, display_name)
                            else:
                                # Initial scan - STILL announce it so user can respond!
                                logger.info(
                                    f"[DISPLAY MONITOR] Initial scan found: {monitored.name} - will prompt user"
                                )
                                await self._handle_display_detected(monitored, display_name)

            # Check for lost displays (only after initial scan)
            if self.initial_scan_complete:
                for display_id in self.available_displays - current_available:
                    monitored = next(
                        (d for d in self.monitored_displays if d.id == display_id), None
                    )
                    if monitored:
                        logger.info(
                            f"[DISPLAY MONITOR] STATE CHANGE: {monitored.name} became UNAVAILABLE"
                        )
                        await self._handle_display_lost(monitored)

            logger.debug(
                f"[DISPLAY MONITOR] Check: available={list(current_available)}, previous={list(self.available_displays)}"
            )
            self.available_displays = current_available

            # Mark initial scan as complete after first run
            if not self.initial_scan_complete:
                self.initial_scan_complete = True
                logger.info(
                    f"[DISPLAY MONITOR] Initial scan complete. Currently monitoring {len(self.active_connections)} connections"
                )