"""
JARVIS macOS Helper - System Event Monitor

Real-time monitoring of macOS system events using NSWorkspace notifications.
Provides async event emission for app launches, focus changes, space transitions,
and system state changes.

Features:
- App lifecycle monitoring (launch, terminate, activate, hide)
- Window focus tracking via Accessibility API
- Space/desktop change detection via Yabai or direct monitoring
- System state monitoring (sleep, wake, screen lock/unlock)
- User activity detection (idle, active)
- Async event emission with batching
- Integration with existing Yabai Spatial Intelligence

Apple Compliance:
- Uses NSWorkspace notifications (public API)
- Accessibility API requires user permission
- No private API usage
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Monitor Configuration
# =============================================================================

@dataclass
class MonitorConfig:
    """Configuration for the system event monitor."""
    # Polling intervals
    window_poll_interval_ms: int = 100  # Window focus polling
    space_poll_interval_ms: int = 250  # Space change polling
    idle_poll_interval_ms: int = 5000  # User idle polling

    # Feature flags
    enable_app_monitoring: bool = True
    enable_window_monitoring: bool = True
    enable_space_monitoring: bool = True
    enable_system_state_monitoring: bool = True
    enable_idle_monitoring: bool = True

    # Yabai integration
    use_yabai_if_available: bool = True
    yabai_path: str = "/opt/homebrew/bin/yabai"

    # Idle thresholds
    idle_threshold_seconds: float = 300.0  # 5 minutes
    short_idle_threshold_seconds: float = 30.0  # 30 seconds

    # Batching
    enable_event_batching: bool = True
    batch_window_ms: int = 50  # Batch events within 50ms


# =============================================================================
# System Event Monitor
# =============================================================================

class SystemEventMonitor:
    """
    Monitors macOS system events and emits them to the event bus.

    Uses multiple sources:
    - NSWorkspace notifications for app lifecycle
    - Accessibility API for window focus
    - Yabai for space management (if available)
    - IOKit for idle time
    """

    def __init__(self, config: Optional[MonitorConfig] = None):
        """
        Initialize the system event monitor.

        Args:
            config: Monitor configuration (uses defaults if None)
        """
        self.config = config or MonitorConfig()

        # State tracking
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # App state
        self._running_apps: Dict[str, Dict[str, Any]] = {}  # bundle_id -> app info
        self._frontmost_app: Optional[str] = None  # bundle_id

        # Window state
        self._focused_window: Optional[Dict[str, Any]] = None
        self._windows: Dict[int, Dict[str, Any]] = {}  # window_id -> window info

        # Space state
        self._current_space: int = 1
        self._spaces: Dict[int, Dict[str, Any]] = {}

        # System state
        self._is_screen_locked: bool = False
        self._is_system_sleeping: bool = False
        self._last_user_activity: datetime = datetime.now()
        self._is_idle: bool = False

        # Yabai integration
        self._yabai_available: bool = False
        self._yabai_si = None  # YabaiSpatialIntelligence instance

        # Event bus (lazy loaded)
        self._event_bus = None

        # Event batching
        self._pending_events: List[Any] = []
        self._batch_lock = asyncio.Lock()

        logger.info("SystemEventMonitor initialized")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """Start all monitoring tasks."""
        if self._running:
            logger.warning("SystemEventMonitor already running")
            return

        self._running = True

        # Initialize event bus
        try:
            from .event_bus import get_macos_event_bus
            self._event_bus = await get_macos_event_bus()
        except Exception as e:
            logger.error(f"Failed to initialize event bus: {e}")
            return

        # Check Yabai availability
        await self._check_yabai()

        # Initialize existing spatial intelligence if available
        if self.config.use_yabai_if_available and self._yabai_available:
            await self._init_yabai_integration()

        # Start monitoring tasks
        await self._start_monitoring_tasks()

        # Initial state capture
        await self._capture_initial_state()

        logger.info("SystemEventMonitor started")

    async def stop(self) -> None:
        """Stop all monitoring tasks."""
        if not self._running:
            return

        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

        # Stop Yabai integration
        if self._yabai_si:
            try:
                await self._yabai_si.stop_monitoring()
            except Exception as e:
                logger.warning(f"Error stopping Yabai SI: {e}")

        logger.info("SystemEventMonitor stopped")

    async def _start_monitoring_tasks(self) -> None:
        """Start individual monitoring tasks."""
        if self.config.enable_app_monitoring:
            self._tasks.append(
                asyncio.create_task(
                    self._app_monitoring_loop(),
                    name="app_monitor"
                )
            )

        if self.config.enable_window_monitoring:
            self._tasks.append(
                asyncio.create_task(
                    self._window_monitoring_loop(),
                    name="window_monitor"
                )
            )

        if self.config.enable_space_monitoring and not self._yabai_si:
            # Only run our own space monitoring if not using Yabai SI
            self._tasks.append(
                asyncio.create_task(
                    self._space_monitoring_loop(),
                    name="space_monitor"
                )
            )

        if self.config.enable_system_state_monitoring:
            self._tasks.append(
                asyncio.create_task(
                    self._system_state_monitoring_loop(),
                    name="system_state_monitor"
                )
            )

        if self.config.enable_idle_monitoring:
            self._tasks.append(
                asyncio.create_task(
                    self._idle_monitoring_loop(),
                    name="idle_monitor"
                )
            )

        if self.config.enable_event_batching:
            self._tasks.append(
                asyncio.create_task(
                    self._event_batch_processor(),
                    name="event_batch_processor"
                )
            )

    async def _capture_initial_state(self) -> None:
        """Capture initial system state on startup."""
        try:
            # Get running apps
            await self._update_running_apps()

            # Get current space
            await self._update_current_space()

            # Check screen lock status
            await self._update_screen_lock_status()

            logger.debug(
                f"Initial state: apps={len(self._running_apps)}, "
                f"space={self._current_space}, locked={self._is_screen_locked}"
            )

        except Exception as e:
            logger.error(f"Failed to capture initial state: {e}")

    # =========================================================================
    # Yabai Integration
    # =========================================================================

    async def _check_yabai(self) -> None:
        """Check if Yabai is available."""
        try:
            result = await asyncio.create_subprocess_exec(
                self.config.yabai_path, "-v",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()

            if result.returncode == 0:
                self._yabai_available = True
                logger.info(f"Yabai available: {stdout.decode().strip()}")
            else:
                self._yabai_available = False
                logger.info("Yabai not available")

        except FileNotFoundError:
            self._yabai_available = False
            logger.info("Yabai not installed")
        except Exception as e:
            self._yabai_available = False
            logger.warning(f"Yabai check failed: {e}")

    async def _init_yabai_integration(self) -> None:
        """Initialize integration with existing Yabai Spatial Intelligence."""
        try:
            from intelligence.yabai_spatial_intelligence import YabaiSpatialIntelligence

            self._yabai_si = YabaiSpatialIntelligence()

            # Register event listener to bridge Yabai events
            self._yabai_si.register_event_listener(
                "space_changed",
                self._on_yabai_space_changed
            )
            self._yabai_si.register_event_listener(
                "window_focused",
                self._on_yabai_window_focused
            )
            self._yabai_si.register_event_listener(
                "app_launched",
                self._on_yabai_app_launched
            )

            # Start Yabai monitoring
            await self._yabai_si.start_monitoring()
            logger.info("Yabai Spatial Intelligence integration active")

        except ImportError:
            logger.info("Yabai Spatial Intelligence not available, using direct monitoring")
            self._yabai_si = None
        except Exception as e:
            logger.warning(f"Failed to initialize Yabai SI: {e}")
            self._yabai_si = None

    async def _on_yabai_space_changed(self, yabai_event: Any) -> None:
        """Handle space change from Yabai SI."""
        from .event_types import MacOSEventFactory
        event = MacOSEventFactory.create_space_changed(
            new_space_id=yabai_event.space_id,
            new_space_index=yabai_event.space_index,
            previous_space_id=self._current_space,
        )
        self._current_space = yabai_event.space_id
        await self._emit_event(event)

    async def _on_yabai_window_focused(self, yabai_event: Any) -> None:
        """Handle window focus from Yabai SI."""
        from .event_types import MacOSEventFactory
        event = MacOSEventFactory.create_window_focused(
            app_name=yabai_event.app_name,
            window_id=yabai_event.window_id,
            window_title=yabai_event.title,
        )
        await self._emit_event(event)

    async def _on_yabai_app_launched(self, yabai_event: Any) -> None:
        """Handle app launch from Yabai SI."""
        from .event_types import MacOSEventFactory
        event = MacOSEventFactory.create_app_launched(
            app_name=yabai_event.app_name,
            bundle_id=getattr(yabai_event, 'bundle_id', ''),
        )
        await self._emit_event(event)

    # =========================================================================
    # App Monitoring
    # =========================================================================

    async def _app_monitoring_loop(self) -> None:
        """Monitor app launches and terminations via NSWorkspace."""
        while self._running:
            try:
                await asyncio.sleep(0.5)  # Check every 500ms
                await self._update_running_apps()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"App monitoring error: {e}")
                await asyncio.sleep(1)

    async def _update_running_apps(self) -> None:
        """Update running apps and emit events for changes."""
        try:
            # Get running apps via AppleScript
            script = """
            tell application "System Events"
                set appList to {}
                repeat with p in (processes whose background only is false)
                    set end of appList to {name of p, bundle identifier of p, frontmost of p}
                end repeat
                return appList
            end tell
            """
            result = await asyncio.create_subprocess_exec(
                "osascript", "-e", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()

            if result.returncode != 0:
                return

            # Parse output
            current_apps: Dict[str, Dict[str, Any]] = {}
            output = stdout.decode().strip()

            # Parse AppleScript list format
            # Format: {{name, bundle_id, frontmost}, ...}
            if output.startswith("{") and output.endswith("}"):
                # Simple parsing - could be more robust
                import re
                app_pattern = r'\{([^,]+), ([^,]+), (true|false)\}'
                matches = re.findall(app_pattern, output.replace("missing value", ""))

                for name, bundle_id, frontmost in matches:
                    name = name.strip()
                    bundle_id = bundle_id.strip()
                    is_frontmost = frontmost.strip() == "true"

                    current_apps[bundle_id] = {
                        "name": name,
                        "bundle_id": bundle_id,
                        "is_frontmost": is_frontmost,
                    }

                    # Track frontmost app
                    if is_frontmost and self._frontmost_app != bundle_id:
                        old_frontmost = self._frontmost_app
                        self._frontmost_app = bundle_id
                        await self._emit_app_activated(current_apps[bundle_id])

            # Detect launches and terminations
            from .event_types import MacOSEventFactory

            # New apps
            for bundle_id, info in current_apps.items():
                if bundle_id not in self._running_apps:
                    event = MacOSEventFactory.create_app_launched(
                        app_name=info["name"],
                        bundle_id=bundle_id,
                    )
                    await self._emit_event(event)

            # Terminated apps
            for bundle_id, info in self._running_apps.items():
                if bundle_id not in current_apps:
                    from .event_types import MacOSEventType, AppEvent
                    event = AppEvent(
                        event_type=MacOSEventType.APP_TERMINATED,
                        source="system_event_monitor",
                        app_name=info["name"],
                        bundle_id=bundle_id,
                        requires_agi_processing=True,
                    )
                    await self._emit_event(event)

            self._running_apps = current_apps

        except Exception as e:
            logger.debug(f"Error updating running apps: {e}")

    async def _emit_app_activated(self, app_info: Dict[str, Any]) -> None:
        """Emit app activated event."""
        from .event_types import MacOSEventFactory
        event = MacOSEventFactory.create_app_activated(
            app_name=app_info["name"],
            bundle_id=app_info["bundle_id"],
        )
        await self._emit_event(event)

    # =========================================================================
    # Window Monitoring
    # =========================================================================

    async def _window_monitoring_loop(self) -> None:
        """Monitor window focus changes via Accessibility API."""
        while self._running:
            try:
                await asyncio.sleep(self.config.window_poll_interval_ms / 1000)
                await self._update_focused_window()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Window monitoring error: {e}")
                await asyncio.sleep(0.5)

    async def _update_focused_window(self) -> None:
        """Update focused window information."""
        try:
            # Get focused window via AppleScript
            script = """
            tell application "System Events"
                set frontApp to first application process whose frontmost is true
                set appName to name of frontApp
                try
                    set frontWindow to first window of frontApp
                    set winTitle to name of frontWindow
                    return {appName, winTitle}
                on error
                    return {appName, ""}
                end try
            end tell
            """
            result = await asyncio.create_subprocess_exec(
                "osascript", "-e", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()

            if result.returncode != 0:
                return

            output = stdout.decode().strip()
            # Parse: {appName, windowTitle}
            if output.startswith("{") and output.endswith("}"):
                parts = output[1:-1].split(", ", 1)
                if len(parts) >= 1:
                    app_name = parts[0].strip()
                    window_title = parts[1].strip() if len(parts) > 1 else ""

                    new_window = {
                        "app_name": app_name,
                        "title": window_title,
                    }

                    # Check if window changed
                    if (not self._focused_window or
                        self._focused_window.get("app_name") != app_name or
                        self._focused_window.get("title") != window_title):

                        old_window = self._focused_window
                        self._focused_window = new_window

                        # Emit window focused event
                        from .event_types import MacOSEventFactory
                        event = MacOSEventFactory.create_window_focused(
                            app_name=app_name,
                            window_id=0,  # Not available via AppleScript
                            window_title=window_title,
                        )
                        await self._emit_event(event)

        except Exception as e:
            logger.debug(f"Error updating focused window: {e}")

    # =========================================================================
    # Space Monitoring
    # =========================================================================

    async def _space_monitoring_loop(self) -> None:
        """Monitor space/desktop changes (fallback when Yabai SI not used)."""
        while self._running:
            try:
                await asyncio.sleep(self.config.space_poll_interval_ms / 1000)
                await self._update_current_space()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Space monitoring error: {e}")
                await asyncio.sleep(1)

    async def _update_current_space(self) -> None:
        """Update current space information."""
        try:
            if self._yabai_available:
                # Use Yabai for space info
                result = await asyncio.create_subprocess_exec(
                    self.config.yabai_path, "-m", "query", "--spaces", "--space",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await result.communicate()

                if result.returncode == 0:
                    import json
                    space_info = json.loads(stdout.decode())
                    new_space = space_info.get("index", 1)

                    if new_space != self._current_space:
                        old_space = self._current_space
                        self._current_space = new_space

                        from .event_types import MacOSEventFactory
                        event = MacOSEventFactory.create_space_changed(
                            new_space_id=new_space,
                            new_space_index=new_space,
                            previous_space_id=old_space,
                        )
                        await self._emit_event(event)
            else:
                # Fallback: Can't reliably detect space changes without Yabai
                pass

        except Exception as e:
            logger.debug(f"Error updating space: {e}")

    # =========================================================================
    # System State Monitoring
    # =========================================================================

    async def _system_state_monitoring_loop(self) -> None:
        """Monitor system state (sleep, wake, screen lock)."""
        while self._running:
            try:
                await asyncio.sleep(1)  # Check every second
                await self._update_screen_lock_status()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"System state monitoring error: {e}")
                await asyncio.sleep(2)

    async def _update_screen_lock_status(self) -> None:
        """Update screen lock status."""
        try:
            # Try using the existing screen lock detector
            try:
                from voice_unlock.objc.server.screen_lock_detector import is_screen_locked
                is_locked = is_screen_locked()
            except ImportError:
                # Fallback: Check via IOKit/CGSession
                script = """
                tell application "System Events"
                    get running of screen saver preferences
                end tell
                """
                result = await asyncio.create_subprocess_exec(
                    "osascript", "-e", script,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await result.communicate()
                is_locked = b"true" in stdout.lower()

            # Detect change
            if is_locked != self._is_screen_locked:
                old_status = self._is_screen_locked
                self._is_screen_locked = is_locked

                from .event_types import MacOSEventFactory
                if is_locked:
                    event = MacOSEventFactory.create_screen_locked()
                else:
                    event = MacOSEventFactory.create_screen_unlocked()

                await self._emit_event(event)

        except Exception as e:
            logger.debug(f"Error checking screen lock: {e}")

    # =========================================================================
    # Idle Monitoring
    # =========================================================================

    async def _idle_monitoring_loop(self) -> None:
        """Monitor user idle state."""
        while self._running:
            try:
                await asyncio.sleep(self.config.idle_poll_interval_ms / 1000)
                await self._update_idle_state()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Idle monitoring error: {e}")
                await asyncio.sleep(5)

    async def _update_idle_state(self) -> None:
        """Update user idle state."""
        try:
            # Get idle time via ioreg
            result = await asyncio.create_subprocess_exec(
                "ioreg", "-c", "IOHIDSystem", "-d", "4",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()

            if result.returncode != 0:
                return

            # Parse HIDIdleTime
            output = stdout.decode()
            import re
            match = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', output)

            if match:
                # HIDIdleTime is in nanoseconds
                idle_ns = int(match.group(1))
                idle_seconds = idle_ns / 1_000_000_000

                was_idle = self._is_idle

                # Check if user became idle
                if idle_seconds >= self.config.idle_threshold_seconds and not self._is_idle:
                    self._is_idle = True
                    from .event_types import MacOSEventType, UserActivityEvent
                    event = UserActivityEvent(
                        event_type=MacOSEventType.USER_IDLE_STARTED,
                        source="system_event_monitor",
                        activity_type="idle_started",
                        idle_duration_seconds=idle_seconds,
                    )
                    await self._emit_event(event)

                # Check if user became active
                elif idle_seconds < self.config.short_idle_threshold_seconds and self._is_idle:
                    self._is_idle = False
                    self._last_user_activity = datetime.now()

                    from .event_types import MacOSEventType, UserActivityEvent
                    event = UserActivityEvent(
                        event_type=MacOSEventType.USER_IDLE_ENDED,
                        source="system_event_monitor",
                        activity_type="idle_ended",
                        idle_duration_seconds=0,
                    )
                    await self._emit_event(event)

        except Exception as e:
            logger.debug(f"Error checking idle state: {e}")

    # =========================================================================
    # Event Emission
    # =========================================================================

    async def _emit_event(self, event: Any) -> None:
        """Emit an event, optionally batching."""
        if not self._event_bus:
            return

        if self.config.enable_event_batching:
            async with self._batch_lock:
                self._pending_events.append(event)
        else:
            await self._event_bus.emit(event)

    async def _event_batch_processor(self) -> None:
        """Process batched events."""
        while self._running:
            try:
                await asyncio.sleep(self.config.batch_window_ms / 1000)

                async with self._batch_lock:
                    if self._pending_events:
                        events = self._pending_events.copy()
                        self._pending_events.clear()

                # Emit events outside lock
                for event in events:
                    await self._event_bus.emit(event)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Event batch processor error: {e}")

    # =========================================================================
    # Status and Stats
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get current monitor status."""
        return {
            "running": self._running,
            "yabai_available": self._yabai_available,
            "yabai_si_active": self._yabai_si is not None,
            "running_apps": len(self._running_apps),
            "frontmost_app": self._frontmost_app,
            "current_space": self._current_space,
            "is_screen_locked": self._is_screen_locked,
            "is_idle": self._is_idle,
            "last_user_activity": self._last_user_activity.isoformat(),
            "active_tasks": len(self._tasks),
        }

    def get_running_apps(self) -> Dict[str, Dict[str, Any]]:
        """Get currently running apps."""
        return self._running_apps.copy()

    def get_focused_window(self) -> Optional[Dict[str, Any]]:
        """Get currently focused window."""
        return self._focused_window.copy() if self._focused_window else None


# =============================================================================
# Singleton Pattern
# =============================================================================

_system_event_monitor: Optional[SystemEventMonitor] = None


async def get_system_event_monitor(
    config: Optional[MonitorConfig] = None,
    auto_start: bool = True,
) -> SystemEventMonitor:
    """
    Get the global system event monitor instance.

    Args:
        config: Monitor configuration
        auto_start: Automatically start monitoring

    Returns:
        The SystemEventMonitor singleton
    """
    global _system_event_monitor

    if _system_event_monitor is None:
        _system_event_monitor = SystemEventMonitor(config)

    if auto_start and not _system_event_monitor._running:
        await _system_event_monitor.start()

    return _system_event_monitor


async def stop_system_event_monitor() -> None:
    """Stop the global system event monitor."""
    global _system_event_monitor

    if _system_event_monitor is not None:
        await _system_event_monitor.stop()
        _system_event_monitor = None
