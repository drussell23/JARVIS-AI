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
import logging
import subprocess
import json
import hashlib
from typing import List, Dict, Optional, Set, Callable, Any
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import os

logger = logging.getLogger(__name__)


class DetectionMethod(Enum):
    """Available detection methods"""
    APPLESCRIPT = "applescript"
    COREGRAPHICS = "coregraphics"
    YABAI = "yabai"


class DisplayType(Enum):
    """Types of displays"""
    AIRPLAY = "airplay"
    HDMI = "hdmi"
    THUNDERBOLT = "thunderbolt"
    USB_C = "usb_c"
    WIRELESS = "wireless"
    UNKNOWN = "unknown"


class ConnectionMode(Enum):
    """Display connection modes"""
    EXTEND = "extend"
    MIRROR = "mirror"


@dataclass
class DisplayInfo:
    """Information about a detected display"""
    id: str
    name: str
    display_type: DisplayType
    is_available: bool
    is_connected: bool
    detection_method: DetectionMethod
    detected_at: datetime
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['display_type'] = self.display_type.value
        data['detection_method'] = self.detection_method.value
        data['detected_at'] = self.detected_at.isoformat()
        return data


@dataclass
class MonitoredDisplay:
    """Configuration for a monitored display"""
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
        """Check if display name matches this configuration"""
        if display_name.lower() == self.name.lower():
            return True
        return any(alias.lower() == display_name.lower() for alias in self.aliases)


class DisplayCache:
    """Cache for display detection results"""

    def __init__(self, ttl_seconds: int = 5):
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, tuple[List[str], datetime]] = {}

    def get(self, key: str) -> Optional[List[str]]:
        """Get cached value if not expired"""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: List[str]):
        """Set cache value"""
        self._cache[key] = (value, datetime.now())

    def clear(self):
        """Clear all cache"""
        self._cache.clear()


class AppleScriptDetector:
    """AppleScript-based display detection"""

    def __init__(self, config: Dict):
        self.config = config
        self.timeout = config.get('timeout_seconds', 5.0)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.retry_delay = config.get('retry_delay_seconds', 0.5)
        self.filter_items = config.get('filter_system_items', [])

    async def detect_displays(self) -> List[str]:
        """Detect available displays using AppleScript"""
        for attempt in range(self.retry_attempts):
            try:
                script = '''
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
                '''

                result = await asyncio.create_subprocess_exec(
                    'osascript', '-e', script,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await asyncio.wait_for(
                    result.communicate(),
                    timeout=self.timeout
                )

                if result.returncode == 0:
                    output = stdout.decode('utf-8').strip()

                    if output.startswith('ERROR:'):
                        logger.debug(f"AppleScript error: {output}")
                        continue

                    # Parse the output
                    devices = []
                    if output:
                        # AppleScript returns comma-separated values
                        raw_devices = [d.strip() for d in output.split(', ')]
                        devices = [
                            d for d in raw_devices
                            if d and d not in self.filter_items
                        ]

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
        """Connect to a display using AppleScript"""
        try:
            script = f'''
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
            '''

            result = await asyncio.create_subprocess_exec(
                'osascript', '-e', script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                result.communicate(),
                timeout=self.timeout * 2  # Connection takes longer
            )

            output = stdout.decode('utf-8').strip()

            if "SUCCESS" in output:
                return {"success": True, "message": f"Connected to {display_name}"}
            else:
                error_detail = output.replace("ERROR:", "").strip()
                return {
                    "success": False,
                    "message": f"Failed to connect to {display_name}",
                    "error": error_detail
                }

        except asyncio.TimeoutError:
            return {"success": False, "message": "Connection timeout"}
        except Exception as e:
            return {"success": False, "message": str(e)}


class CoreGraphicsDetector:
    """Core Graphics-based display detection"""

    def __init__(self, config: Dict):
        self.config = config
        self.max_displays = config.get('max_displays', 32)
        self.exclude_builtin = config.get('exclude_builtin', True)

    async def detect_displays(self) -> List[str]:
        """Detect displays using Core Graphics"""
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
    """Yabai-based display detection"""

    def __init__(self, config: Dict):
        self.config = config
        self.timeout = config.get('command_timeout', 3.0)

    async def detect_displays(self) -> List[str]:
        """Detect displays using Yabai"""
        try:
            result = await asyncio.create_subprocess_exec(
                'yabai', '-m', 'query', '--displays',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                result.communicate(),
                timeout=self.timeout
            )

            if result.returncode == 0:
                displays_data = json.loads(stdout.decode('utf-8'))
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
    Advanced display monitoring system

    Features:
    - Multi-method detection (AppleScript, CoreGraphics, Yabai)
    - Smart caching
    - Voice integration
    - Event-driven callbacks
    - Graceful degradation
    - Zero hardcoding
    """

    def __init__(self, config_path: Optional[str] = None, voice_handler = None):
        """
        Initialize advanced display monitor

        Args:
            config_path: Path to configuration JSON file
            voice_handler: Voice integration handler for TTS
        """
        self.config = self._load_config(config_path)
        self.voice_handler = voice_handler

        # Initialize components
        self.cache = DisplayCache(
            ttl_seconds=self.config['caching']['display_list_ttl_seconds']
        )

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

        # Event callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'display_detected': [],
            'display_lost': [],
            'display_connected': [],
            'display_disconnected': [],
            'error': []
        }

        logger.info(f"[DISPLAY MONITOR] Initialized with {len(self.monitored_displays)} monitored displays")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file"""
        if config_path is None:
            # Default config path
            config_path = Path(__file__).parent.parent / 'config' / 'display_monitor_config.json'

        try:
            with open(config_path, 'r') as f:
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
        """Initialize detection method instances"""
        methods = self.config['display_monitoring']['detection_methods']

        if 'applescript' in methods and self.config['applescript']['enabled']:
            self.detectors[DetectionMethod.APPLESCRIPT] = AppleScriptDetector(
                self.config['applescript']
            )

        if 'coregraphics' in methods and self.config['coregraphics']['enabled']:
            self.detectors[DetectionMethod.COREGRAPHICS] = CoreGraphicsDetector(
                self.config['coregraphics']
            )

        if 'yabai' in methods and self.config['yabai']['enabled']:
            self.detectors[DetectionMethod.YABAI] = YabaiDetector(
                self.config['yabai']
            )

        logger.info(f"[DISPLAY MONITOR] Initialized {len(self.detectors)} detection methods")

    def _load_monitored_displays(self) -> List[MonitoredDisplay]:
        """Load monitored displays from configuration"""
        displays = []
        for display_config in self.config['displays']['monitored_displays']:
            if display_config.get('enabled', True):
                displays.append(MonitoredDisplay(**display_config))
        return displays

    def register_callback(self, event: str, callback: Callable):
        """
        Register event callback

        Args:
            event: Event name (display_detected, display_lost, etc.)
            callback: Async callback function
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            logger.debug(f"[DISPLAY MONITOR] Registered callback for {event}")

    async def _emit_event(self, event: str, **kwargs):
        """Emit event to all registered callbacks"""
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
        """Start display monitoring"""
        if self.is_monitoring:
            logger.warning("[DISPLAY MONITOR] Already monitoring")
            return

        if not self.config['display_monitoring']['enabled']:
            logger.warning("[DISPLAY MONITOR] Monitoring disabled in config")
            return

        # Startup delay
        startup_delay = self.config['display_monitoring']['startup_delay_seconds']
        if startup_delay > 0:
            logger.info(f"[DISPLAY MONITOR] Starting in {startup_delay}s...")
            await asyncio.sleep(startup_delay)

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitor_loop())
        logger.info("[DISPLAY MONITOR] Started monitoring")

    async def stop(self):
        """Stop display monitoring"""
        self.is_monitoring = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        self.cache.clear()
        logger.info("[DISPLAY MONITOR] Stopped monitoring")

    async def _monitor_loop(self):
        """Main monitoring loop"""
        check_interval = self.config['display_monitoring']['check_interval_seconds']

        try:
            while self.is_monitoring:
                await self._check_displays()
                await asyncio.sleep(check_interval)
        except asyncio.CancelledError:
            logger.info("[DISPLAY MONITOR] Monitoring cancelled")
        except Exception as e:
            logger.error(f"[DISPLAY MONITOR] Error in monitoring loop: {e}")
            await self._emit_event('error', error=e)

    async def _check_displays(self):
        """Check for available displays"""
        try:
            # Detect all available displays
            detected_displays = await self._detect_all_displays()

            # Match against monitored displays
            current_available = set()
            for display_name in detected_displays:
                for monitored in self.monitored_displays:
                    if monitored.matches(display_name):
                        current_available.add(monitored.id)

                        # New display detected
                        if monitored.id not in self.available_displays:
                            await self._handle_display_detected(monitored, display_name)

            # Check for lost displays
            for display_id in self.available_displays - current_available:
                monitored = next((d for d in self.monitored_displays if d.id == display_id), None)
                if monitored:
                    await self._handle_display_lost(monitored)

            self.available_displays = current_available

        except Exception as e:
            logger.error(f"[DISPLAY MONITOR] Error checking displays: {e}")

    async def _detect_all_displays(self) -> List[str]:
        """Detect displays using all available methods"""
        # Check cache first
        cache_key = "all_displays"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        all_displays = set()
        preferred_method = self.config['display_monitoring'].get('preferred_detection_method')

        # Try preferred method first
        if preferred_method:
            method_enum = DetectionMethod(preferred_method)
            if method_enum in self.detectors:
                displays = await self.detectors[method_enum].detect_displays()
                all_displays.update(displays)

                # If we got results, cache and return
                if displays:
                    result = list(all_displays)
                    self.cache.set(cache_key, result)
                    return result

        # Fallback: Try all methods (parallel if enabled)
        if self.config['performance']['parallel_detection']:
            tasks = [
                detector.detect_displays()
                for detector in self.detectors.values()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, list):
                    all_displays.update(result)
        else:
            for detector in self.detectors.values():
                displays = await detector.detect_displays()
                all_displays.update(displays)

        result = list(all_displays)
        self.cache.set(cache_key, result)
        return result

    async def _handle_display_detected(self, monitored: MonitoredDisplay, detected_name: str):
        """Handle newly detected display"""
        logger.info(f"[DISPLAY MONITOR] Detected: {monitored.name} ({detected_name})")

        # Emit event
        await self._emit_event('display_detected', display=monitored, detected_name=detected_name)

        # Voice prompt if enabled
        if monitored.auto_prompt and self.config['voice_integration']['speak_on_detection']:
            await self._speak_detection_prompt(monitored)

        # Auto-connect if enabled
        if monitored.auto_connect:
            await self.connect_display(monitored.id)

    async def _handle_display_lost(self, monitored: MonitoredDisplay):
        """Handle lost display"""
        logger.info(f"[DISPLAY MONITOR] Lost: {monitored.name}")

        # Emit event
        await self._emit_event('display_lost', display=monitored)

        if monitored.id in self.connected_displays:
            self.connected_displays.remove(monitored.id)
            await self._emit_event('display_disconnected', display=monitored)

    async def _speak_detection_prompt(self, monitored: MonitoredDisplay):
        """Speak detection prompt"""
        if not self.config['voice_integration']['enabled']:
            return

        template = self.config['voice_integration']['prompt_template']
        message = template.format(display_name=monitored.name)

        logger.info(f"[DISPLAY MONITOR] Voice: {message}")

        # Use voice handler if available
        if self.voice_handler:
            try:
                await self.voice_handler.speak(message)
            except Exception as e:
                logger.error(f"[DISPLAY MONITOR] Voice handler error: {e}")
        else:
            # Fallback to macOS say command
            try:
                subprocess.Popen(['say', message])
            except Exception as e:
                logger.error(f"[DISPLAY MONITOR] Say command error: {e}")

    async def connect_display(self, display_id: str) -> Dict[str, Any]:
        """
        Connect to a display

        Args:
            display_id: Display ID from configuration

        Returns:
            Connection result dictionary
        """
        # Find monitored display
        monitored = next((d for d in self.monitored_displays if d.id == display_id), None)
        if not monitored:
            return {"success": False, "message": f"Display {display_id} not found in configuration"}

        logger.info(f"[DISPLAY MONITOR] Connecting to {monitored.name}...")

        # Use AppleScript detector for connection
        if DetectionMethod.APPLESCRIPT in self.detectors:
            result = await self.detectors[DetectionMethod.APPLESCRIPT].connect_display(monitored.name)

            if result['success']:
                self.connected_displays.add(display_id)
                await self._emit_event('display_connected', display=monitored)

                # Speak success message
                if self.config['voice_integration']['speak_on_connection']:
                    template = self.config['voice_integration']['connection_success_message']
                    message = template.format(display_name=monitored.name)

                    if self.voice_handler:
                        try:
                            await self.voice_handler.speak(message)
                        except:
                            subprocess.Popen(['say', message])
                    else:
                        subprocess.Popen(['say', message])

            return result

        return {"success": False, "message": "AppleScript detector not available"}

    def get_status(self) -> Dict[str, Any]:
        """Get current monitor status"""
        return {
            "is_monitoring": self.is_monitoring,
            "available_displays": list(self.available_displays),
            "connected_displays": list(self.connected_displays),
            "monitored_count": len(self.monitored_displays),
            "detection_methods": [m.value for m in self.detectors.keys()],
            "cache_enabled": self.config['caching']['enabled']
        }


# Singleton instance
_monitor_instance: Optional[AdvancedDisplayMonitor] = None


def get_display_monitor(config_path: Optional[str] = None, voice_handler = None) -> AdvancedDisplayMonitor:
    """Get singleton display monitor instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = AdvancedDisplayMonitor(config_path, voice_handler)
    return _monitor_instance


if __name__ == "__main__":
    # Test the monitor
    async def test():
        logging.basicConfig(level=logging.INFO)

        monitor = get_display_monitor()

        # Register test callbacks
        async def on_detected(display, detected_name):
            print(f"✅ Detected: {display.name} ({detected_name})")

        async def on_lost(display):
            print(f"❌ Lost: {display.name}")

        monitor.register_callback('display_detected', on_detected)
        monitor.register_callback('display_lost', on_lost)

        await monitor.start()

        print("Monitoring for displays... Press Ctrl+C to stop")
        try:
            while True:
                status = monitor.get_status()
                print(f"Status: {status}")
                await asyncio.sleep(5)
        except KeyboardInterrupt:
            print("\nStopping...")
            await monitor.stop()

    asyncio.run(test())
