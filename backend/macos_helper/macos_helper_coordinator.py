"""
JARVIS macOS Helper - Main Coordinator

Central coordinator for the macOS helper layer.
Integrates all monitoring components and provides a unified interface
for the AGI OS coordinator.

Features:
- Unified lifecycle management for all monitors
- Permission onboarding workflow
- Health monitoring and self-healing
- Integration with AGI OS coordinator
- Menu bar status indicator (via Swift bridge)
- LaunchAgent support for background operation
- Graceful degradation when permissions missing

Architecture:
    MacOSHelperCoordinator
           │
    ┌──────┴──────┐
    │ Components  │
    │ ┌─────────┐ │
    │ │ Event   │ │
    │ │  Bus    │ │
    │ └─────────┘ │
    │ ┌─────────┐ │
    │ │ System  │ │
    │ │ Monitor │ │
    │ └─────────┘ │
    │ ┌─────────┐ │
    │ │ Notif   │ │
    │ │ Monitor │ │
    │ └─────────┘ │
    │ ┌─────────┐ │
    │ │ Perms   │ │
    │ │ Manager │ │
    │ └─────────┘ │
    └─────────────┘
           │
    ┌──────┴──────┐
    │ AGI OS      │
    │ Coordinator │
    └─────────────┘
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .menu_bar import MenuBarIndicator
    from .agi_integration import AGIBridge

logger = logging.getLogger(__name__)


# =============================================================================
# State and Configuration
# =============================================================================

class MacOSHelperState(str, Enum):
    """State of the macOS helper."""
    OFFLINE = "offline"
    INITIALIZING = "initializing"
    ONBOARDING = "onboarding"  # Permission onboarding in progress
    ONLINE = "online"
    DEGRADED = "degraded"  # Some permissions missing
    PAUSED = "paused"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class MacOSHelperConfig:
    """Configuration for the macOS helper coordinator."""
    # Components to enable
    enable_system_monitor: bool = True
    enable_notification_monitor: bool = True
    enable_permission_monitor: bool = True
    enable_menu_bar: bool = True  # Menu bar status indicator

    # AGI OS integration
    enable_agi_bridge: bool = True
    enable_voice_feedback: bool = True

    # Health monitoring
    health_check_interval_seconds: float = 30.0
    auto_restart_on_failure: bool = True
    max_restart_attempts: int = 3

    # Permission handling
    require_all_permissions: bool = False  # If True, won't start without all perms
    run_permission_onboarding: bool = True

    # Startup
    announce_startup: bool = True
    startup_delay_seconds: float = 1.0  # Delay before starting monitors

    # Logging
    log_level: str = "INFO"


@dataclass
class ComponentStatus:
    """Status of a component."""
    name: str
    running: bool = False
    healthy: bool = True
    error: Optional[str] = None
    last_check: datetime = field(default_factory=datetime.now)
    restart_count: int = 0


@dataclass
class HelperStats:
    """Statistics for the helper."""
    started_at: Optional[datetime] = None
    uptime_seconds: float = 0.0
    events_processed: int = 0
    permissions_granted: int = 0
    permissions_denied: int = 0
    component_restarts: int = 0
    agi_events_bridged: int = 0


# =============================================================================
# Main Coordinator
# =============================================================================

class MacOSHelperCoordinator:
    """
    Central coordinator for the JARVIS macOS helper layer.

    Manages:
    - Event bus
    - System event monitor
    - Notification monitor
    - Permission manager
    - AGI OS integration
    - Health monitoring
    """

    def __init__(self, config: Optional[MacOSHelperConfig] = None):
        """
        Initialize the macOS helper coordinator.

        Args:
            config: Helper configuration
        """
        self.config = config or MacOSHelperConfig()

        # State
        self._state = MacOSHelperState.OFFLINE
        self._started_at: Optional[datetime] = None

        # Components (lazy loaded)
        self._event_bus = None
        self._system_monitor = None
        self._notification_monitor = None
        self._permission_manager = None
        self._menu_bar: Optional['MenuBarIndicator'] = None

        # AGI OS integration
        self._agi_coordinator = None
        self._voice_communicator = None
        self._agi_bridge: Optional['AGIBridge'] = None

        # Component status tracking
        self._component_status: Dict[str, ComponentStatus] = {}

        # Health monitoring
        self._health_task: Optional[asyncio.Task] = None

        # Statistics
        self._stats = HelperStats()

        # Callbacks
        self._on_state_changed: List[Callable[[MacOSHelperState], Coroutine]] = []

        logger.info("MacOSHelperCoordinator initialized")

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def state(self) -> MacOSHelperState:
        """Get current helper state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if helper is running."""
        return self._state in [MacOSHelperState.ONLINE, MacOSHelperState.DEGRADED]

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> bool:
        """
        Start the macOS helper.

        Returns:
            True if started successfully
        """
        if self._state in [MacOSHelperState.ONLINE, MacOSHelperState.DEGRADED]:
            logger.warning("MacOSHelper already running")
            return True

        logger.info("Starting MacOSHelper...")
        await self._set_state(MacOSHelperState.INITIALIZING)
        self._started_at = datetime.now()
        self._stats.started_at = self._started_at

        try:
            # Step 1: Initialize event bus
            await self._init_event_bus()

            # Step 2: Initialize permission manager and check permissions
            await self._init_permission_manager()
            permissions_ok = await self._check_permissions()

            if not permissions_ok:
                if self.config.require_all_permissions:
                    logger.error("Required permissions not granted")
                    await self._set_state(MacOSHelperState.ERROR)
                    return False

                if self.config.run_permission_onboarding:
                    await self._set_state(MacOSHelperState.ONBOARDING)
                    await self._run_onboarding()

            # Step 3: Initialize monitors
            await asyncio.sleep(self.config.startup_delay_seconds)
            await self._init_monitors()

            # Step 4: Initialize AGI OS integration
            if self.config.enable_agi_bridge:
                await self._init_agi_integration()
                await self._init_agi_bridge()

            # Step 5: Initialize menu bar
            if self.config.enable_menu_bar:
                await self._init_menu_bar()

            # Step 6: Start health monitoring
            self._health_task = asyncio.create_task(
                self._health_monitor_loop(),
                name="macos_helper_health"
            )

            # Step 7: Determine final state
            final_state = self._determine_health_state()
            await self._set_state(final_state)

            # Step 8: Announce startup
            if self.config.announce_startup and self._voice_communicator:
                await self._announce_startup()

            logger.info(f"MacOSHelper started (state={self._state.value})")
            return True

        except Exception as e:
            logger.exception(f"Failed to start MacOSHelper: {e}")
            await self._set_state(MacOSHelperState.ERROR)
            return False

    async def stop(self) -> None:
        """Stop the macOS helper."""
        if self._state == MacOSHelperState.OFFLINE:
            return

        logger.info("Stopping MacOSHelper...")
        await self._set_state(MacOSHelperState.SHUTTING_DOWN)

        # Announce shutdown
        if self._voice_communicator:
            try:
                from agi_os.realtime_voice_communicator import VoiceMode
                await self._voice_communicator.speak(
                    "macOS helper shutting down.",
                    mode=VoiceMode.QUIET
                )
            except Exception:
                pass

        # Cancel health monitor
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Stop components in reverse order
        await self._stop_monitors()
        await self._stop_event_bus()

        await self._set_state(MacOSHelperState.OFFLINE)
        logger.info("MacOSHelper stopped")

    async def pause(self) -> None:
        """Pause monitoring (still responds to direct queries)."""
        if self._state in [MacOSHelperState.ONLINE, MacOSHelperState.DEGRADED]:
            await self._set_state(MacOSHelperState.PAUSED)
            logger.info("MacOSHelper paused")

    async def resume(self) -> None:
        """Resume monitoring."""
        if self._state == MacOSHelperState.PAUSED:
            final_state = self._determine_health_state()
            await self._set_state(final_state)
            logger.info("MacOSHelper resumed")

    # =========================================================================
    # Component Initialization
    # =========================================================================

    async def _init_event_bus(self) -> None:
        """Initialize the event bus."""
        try:
            from .event_bus import get_macos_event_bus
            self._event_bus = await get_macos_event_bus(
                enable_agi_bridge=self.config.enable_agi_bridge
            )
            self._component_status["event_bus"] = ComponentStatus(
                name="event_bus",
                running=True,
                healthy=True,
            )
            logger.info("Event bus initialized")
        except Exception as e:
            logger.error(f"Failed to initialize event bus: {e}")
            self._component_status["event_bus"] = ComponentStatus(
                name="event_bus",
                running=False,
                healthy=False,
                error=str(e),
            )
            raise

    async def _init_permission_manager(self) -> None:
        """Initialize the permission manager."""
        try:
            from .permission_manager import get_permission_manager
            self._permission_manager = await get_permission_manager(
                auto_start_monitoring=self.config.enable_permission_monitor
            )
            self._component_status["permission_manager"] = ComponentStatus(
                name="permission_manager",
                running=True,
                healthy=True,
            )
            logger.info("Permission manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize permission manager: {e}")
            self._component_status["permission_manager"] = ComponentStatus(
                name="permission_manager",
                running=False,
                healthy=False,
                error=str(e),
            )

    async def _init_monitors(self) -> None:
        """Initialize monitoring components."""
        # System event monitor
        if self.config.enable_system_monitor:
            try:
                from .system_event_monitor import get_system_event_monitor
                self._system_monitor = await get_system_event_monitor()
                self._component_status["system_monitor"] = ComponentStatus(
                    name="system_monitor",
                    running=True,
                    healthy=True,
                )
                logger.info("System event monitor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize system monitor: {e}")
                self._component_status["system_monitor"] = ComponentStatus(
                    name="system_monitor",
                    running=False,
                    healthy=False,
                    error=str(e),
                )

        # Notification monitor
        if self.config.enable_notification_monitor:
            try:
                from .notification_monitor import get_notification_monitor
                self._notification_monitor = await get_notification_monitor()
                self._component_status["notification_monitor"] = ComponentStatus(
                    name="notification_monitor",
                    running=True,
                    healthy=True,
                )
                logger.info("Notification monitor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize notification monitor: {e}")
                self._component_status["notification_monitor"] = ComponentStatus(
                    name="notification_monitor",
                    running=False,
                    healthy=False,
                    error=str(e),
                )

    async def _init_agi_integration(self) -> None:
        """Initialize AGI OS integration."""
        # Connect to AGI OS coordinator
        try:
            from agi_os import get_agi_os
            self._agi_coordinator = await get_agi_os()
            logger.info("Connected to AGI OS coordinator")
        except ImportError:
            logger.info("AGI OS coordinator not available")
        except Exception as e:
            logger.warning(f"Could not connect to AGI OS: {e}")

        # Initialize voice communicator
        if self.config.enable_voice_feedback:
            try:
                from agi_os.realtime_voice_communicator import get_voice_communicator
                self._voice_communicator = await get_voice_communicator()
                logger.info("Voice communicator initialized")
            except Exception as e:
                logger.warning(f"Voice communicator not available: {e}")

    async def _init_agi_bridge(self) -> None:
        """Initialize the AGI OS integration bridge."""
        try:
            from .agi_integration import start_agi_bridge, AGIBridgeConfig

            # Configure the bridge based on helper config
            bridge_config = AGIBridgeConfig(
                enable_voice_feedback=self.config.enable_voice_feedback,
                send_context_to_uae=True,
                send_events_to_cai=True,
            )

            self._agi_bridge = await start_agi_bridge(bridge_config)

            if self._agi_bridge:
                self._component_status["agi_bridge"] = ComponentStatus(
                    name="agi_bridge",
                    running=True,
                    healthy=True,
                )
                logger.info("AGI Bridge initialized")
            else:
                self._component_status["agi_bridge"] = ComponentStatus(
                    name="agi_bridge",
                    running=False,
                    healthy=False,
                    error="Failed to start AGI bridge",
                )

        except Exception as e:
            logger.warning(f"AGI Bridge not available: {e}")
            self._component_status["agi_bridge"] = ComponentStatus(
                name="agi_bridge",
                running=False,
                healthy=False,
                error=str(e),
            )

    async def _stop_agi_bridge(self) -> None:
        """Stop the AGI bridge."""
        if self._agi_bridge:
            try:
                from .agi_integration import stop_agi_bridge
                await stop_agi_bridge()
                self._agi_bridge = None
            except Exception as e:
                logger.warning(f"Error stopping AGI bridge: {e}")

    async def _init_menu_bar(self) -> None:
        """Initialize the menu bar status indicator."""
        try:
            from .menu_bar import start_menu_bar, MenuBarState
            from .menu_bar.status_indicator import MenuBarStats

            self._menu_bar = await start_menu_bar()

            if self._menu_bar:
                # Register action handlers
                self._menu_bar.register_action_handler("pause", self._handle_menu_pause)
                self._menu_bar.register_action_handler("resume", self._handle_menu_resume)
                self._menu_bar.register_action_handler("restart", self._handle_menu_restart)
                self._menu_bar.register_action_handler("quit", self._handle_menu_quit)
                self._menu_bar.register_action_handler("settings", self._handle_menu_settings)

                # Update initial permission status
                await self._update_menu_bar_permissions()

                self._component_status["menu_bar"] = ComponentStatus(
                    name="menu_bar",
                    running=True,
                    healthy=True,
                )
                logger.info("Menu bar indicator initialized")
            else:
                self._component_status["menu_bar"] = ComponentStatus(
                    name="menu_bar",
                    running=False,
                    healthy=False,
                    error="Failed to start menu bar",
                )

        except Exception as e:
            logger.error(f"Failed to initialize menu bar: {e}")
            self._component_status["menu_bar"] = ComponentStatus(
                name="menu_bar",
                running=False,
                healthy=False,
                error=str(e),
            )

    async def _update_menu_bar_permissions(self) -> None:
        """Update menu bar with current permission status."""
        if not self._menu_bar or not self._permission_manager:
            return

        try:
            overview = await self._permission_manager.check_all_permissions()
            for perm_type, result in overview.results.items():
                self._menu_bar.set_permission_status(
                    perm_type.value,
                    result.status.value
                )
        except Exception as e:
            logger.warning(f"Failed to update menu bar permissions: {e}")

    async def _update_menu_bar_stats(self) -> None:
        """Update menu bar with current statistics."""
        if not self._menu_bar:
            return

        try:
            from .menu_bar.status_indicator import MenuBarStats

            stats = MenuBarStats(
                active_apps=self._system_monitor.get_status().get("app_count", 0) if self._system_monitor else 0,
                tracked_windows=self._system_monitor.get_status().get("window_count", 0) if self._system_monitor else 0,
                pending_notifications=self._notification_monitor.get_stats().get("pending_count", 0) if self._notification_monitor else 0,
                events_processed=self._stats.events_processed,
                uptime_seconds=self._stats.uptime_seconds,
                voice_active=self._voice_communicator is not None,
            )

            self._menu_bar.update_stats(stats)
        except Exception as e:
            logger.warning(f"Failed to update menu bar stats: {e}")

    # -------------------------------------------------------------------------
    # Menu Bar Action Handlers
    # -------------------------------------------------------------------------

    async def _handle_menu_pause(self, action_id: str) -> None:
        """Handle pause from menu bar."""
        await self.pause()
        if self._menu_bar:
            from .menu_bar import MenuBarState
            self._menu_bar.set_state(MenuBarState.PAUSED)

    async def _handle_menu_resume(self, action_id: str) -> None:
        """Handle resume from menu bar."""
        await self.resume()
        if self._menu_bar:
            from .menu_bar import MenuBarState
            state_map = {
                MacOSHelperState.ONLINE: MenuBarState.ONLINE,
                MacOSHelperState.DEGRADED: MenuBarState.DEGRADED,
                MacOSHelperState.ERROR: MenuBarState.ERROR,
            }
            self._menu_bar.set_state(state_map.get(self._state, MenuBarState.ONLINE))

    async def _handle_menu_restart(self, action_id: str) -> None:
        """Handle restart from menu bar."""
        if self._menu_bar:
            self._menu_bar.set_activity("Restarting...")
        await self.stop()
        await asyncio.sleep(1)
        await self.start()

    async def _handle_menu_quit(self, action_id: str) -> None:
        """Handle quit from menu bar."""
        await stop_macos_helper()

    async def _handle_menu_settings(self, action_id: str) -> None:
        """Handle settings from menu bar."""
        import subprocess
        # Open System Preferences to Security & Privacy
        subprocess.run([
            "open", "x-apple.systempreferences:com.apple.preference.security?Privacy"
        ], capture_output=True)

    async def _stop_menu_bar(self) -> None:
        """Stop the menu bar indicator."""
        if self._menu_bar:
            try:
                from .menu_bar import stop_menu_bar
                await stop_menu_bar()
                self._menu_bar = None
            except Exception as e:
                logger.warning(f"Error stopping menu bar: {e}")

    async def _stop_monitors(self) -> None:
        """Stop all monitors."""
        # Stop AGI bridge first (to ensure clean event forwarding shutdown)
        await self._stop_agi_bridge()

        # Stop menu bar (UI component)
        await self._stop_menu_bar()

        if self._system_monitor:
            try:
                await self._system_monitor.stop()
            except Exception as e:
                logger.warning(f"Error stopping system monitor: {e}")

        if self._notification_monitor:
            try:
                await self._notification_monitor.stop()
            except Exception as e:
                logger.warning(f"Error stopping notification monitor: {e}")

        if self._permission_manager:
            try:
                await self._permission_manager.stop_monitoring()
            except Exception as e:
                logger.warning(f"Error stopping permission manager: {e}")

    async def _stop_event_bus(self) -> None:
        """Stop the event bus."""
        if self._event_bus:
            try:
                from .event_bus import stop_macos_event_bus
                await stop_macos_event_bus()
            except Exception as e:
                logger.warning(f"Error stopping event bus: {e}")

    # =========================================================================
    # Permission Handling
    # =========================================================================

    async def _check_permissions(self) -> bool:
        """
        Check required permissions.

        Returns:
            True if all required permissions are granted
        """
        if not self._permission_manager:
            return False

        overview = await self._permission_manager.check_all_permissions()

        self._stats.permissions_granted = sum(
            1 for r in overview.results.values()
            if r.status.value == "granted"
        )
        self._stats.permissions_denied = sum(
            1 for r in overview.results.values()
            if r.status.value == "denied"
        )

        return overview.all_required_granted

    async def _run_onboarding(self) -> None:
        """Run permission onboarding workflow."""
        if not self._permission_manager:
            return

        logger.info("Running permission onboarding...")

        steps = await self._permission_manager.generate_onboarding_steps()

        for step in steps:
            # Announce each step via voice if available
            if self._voice_communicator:
                from agi_os.realtime_voice_communicator import VoiceMode
                await self._voice_communicator.speak(
                    f"Please grant {step['name']} permission. {step['description'][:100]}",
                    mode=VoiceMode.NORMAL
                )

            # Open settings for the permission
            await self._permission_manager.open_permission_settings(
                step["permission_type"]
            )

            # Wait for user to grant permission
            await asyncio.sleep(5)

        logger.info("Onboarding complete")

    # =========================================================================
    # Health Monitoring
    # =========================================================================

    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        while self._state not in [MacOSHelperState.OFFLINE, MacOSHelperState.SHUTTING_DOWN]:
            try:
                await asyncio.sleep(self.config.health_check_interval_seconds)

                # Update uptime
                if self._started_at:
                    self._stats.uptime_seconds = (datetime.now() - self._started_at).total_seconds()

                # Check component health
                await self._check_component_health()

                # Update menu bar stats
                await self._update_menu_bar_stats()

                # Update state based on health
                new_state = self._determine_health_state()
                if new_state != self._state and self._state != MacOSHelperState.PAUSED:
                    await self._set_state(new_state)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    async def _check_component_health(self) -> None:
        """Check health of all components."""
        # Event bus
        if self._event_bus:
            stats = self._event_bus.get_stats()
            status = self._component_status.get("event_bus")
            if status:
                status.healthy = stats.get("running", False)
                status.last_check = datetime.now()
                self._stats.events_processed = stats.get("events_processed", 0)
                self._stats.agi_events_bridged = stats.get("events_bridged_to_agi", 0)

        # System monitor
        if self._system_monitor:
            status_data = self._system_monitor.get_status()
            status = self._component_status.get("system_monitor")
            if status:
                status.healthy = status_data.get("running", False)
                status.last_check = datetime.now()

        # Notification monitor
        if self._notification_monitor:
            stats = self._notification_monitor.get_stats()
            status = self._component_status.get("notification_monitor")
            if status:
                status.healthy = stats.get("running", False)
                status.last_check = datetime.now()

        # Permission manager
        if self._permission_manager:
            stats = self._permission_manager.get_stats()
            status = self._component_status.get("permission_manager")
            if status:
                status.healthy = stats.get("monitoring", False) or True  # Always healthy if exists
                status.last_check = datetime.now()

        # Try to restart unhealthy components
        if self.config.auto_restart_on_failure:
            await self._restart_unhealthy_components()

    async def _restart_unhealthy_components(self) -> None:
        """Attempt to restart unhealthy components."""
        for name, status in self._component_status.items():
            if not status.healthy and status.restart_count < self.config.max_restart_attempts:
                logger.info(f"Attempting to restart {name}...")
                status.restart_count += 1
                self._stats.component_restarts += 1

                try:
                    if name == "system_monitor" and self._system_monitor:
                        await self._system_monitor.stop()
                        from .system_event_monitor import get_system_event_monitor
                        self._system_monitor = await get_system_event_monitor()
                        status.healthy = True
                        status.error = None
                        logger.info(f"Successfully restarted {name}")

                    elif name == "notification_monitor" and self._notification_monitor:
                        await self._notification_monitor.stop()
                        from .notification_monitor import get_notification_monitor
                        self._notification_monitor = await get_notification_monitor()
                        status.healthy = True
                        status.error = None
                        logger.info(f"Successfully restarted {name}")

                except Exception as e:
                    logger.error(f"Failed to restart {name}: {e}")
                    status.error = str(e)

    def _determine_health_state(self) -> MacOSHelperState:
        """Determine overall health state."""
        healthy_count = sum(
            1 for s in self._component_status.values()
            if s.healthy
        )
        total_count = len(self._component_status)

        # Event bus is critical
        event_bus_status = self._component_status.get("event_bus")
        if event_bus_status and not event_bus_status.healthy:
            return MacOSHelperState.ERROR

        if healthy_count == total_count:
            return MacOSHelperState.ONLINE
        elif healthy_count > 0:
            return MacOSHelperState.DEGRADED
        else:
            return MacOSHelperState.ERROR

    # =========================================================================
    # State Management
    # =========================================================================

    async def _set_state(self, new_state: MacOSHelperState) -> None:
        """Set helper state and notify callbacks."""
        old_state = self._state
        self._state = new_state

        if old_state != new_state:
            logger.info(f"State changed: {old_state.value} -> {new_state.value}")

            # Sync menu bar state
            if self._menu_bar:
                try:
                    from .menu_bar import MenuBarState
                    state_map = {
                        MacOSHelperState.OFFLINE: MenuBarState.OFFLINE,
                        MacOSHelperState.INITIALIZING: MenuBarState.INITIALIZING,
                        MacOSHelperState.ONBOARDING: MenuBarState.INITIALIZING,
                        MacOSHelperState.ONLINE: MenuBarState.ONLINE,
                        MacOSHelperState.DEGRADED: MenuBarState.DEGRADED,
                        MacOSHelperState.PAUSED: MenuBarState.PAUSED,
                        MacOSHelperState.ERROR: MenuBarState.ERROR,
                        MacOSHelperState.SHUTTING_DOWN: MenuBarState.OFFLINE,
                    }
                    menu_state = state_map.get(new_state, MenuBarState.OFFLINE)
                    self._menu_bar.set_state(menu_state)
                except Exception as e:
                    logger.debug(f"Failed to sync menu bar state: {e}")

            # Call registered callbacks
            for callback in self._on_state_changed:
                try:
                    await callback(new_state)
                except Exception as e:
                    logger.error(f"State change callback error: {e}")

    def on_state_changed(
        self,
        callback: Callable[[MacOSHelperState], Coroutine]
    ) -> None:
        """Register a callback for state changes."""
        self._on_state_changed.append(callback)

    # =========================================================================
    # Voice Feedback
    # =========================================================================

    async def _announce_startup(self) -> None:
        """Announce startup via voice."""
        if not self._voice_communicator:
            return

        try:
            from agi_os.realtime_voice_communicator import VoiceMode

            # Count healthy components
            healthy = sum(1 for s in self._component_status.values() if s.healthy)
            total = len(self._component_status)

            if self._state == MacOSHelperState.ONLINE:
                message = "macOS helper online. All systems operational."
            elif self._state == MacOSHelperState.DEGRADED:
                message = f"macOS helper online in degraded mode. {healthy} of {total} components active."
            else:
                message = "macOS helper starting with limited functionality."

            await self._voice_communicator.speak(message, mode=VoiceMode.NOTIFICATION)

        except Exception as e:
            logger.debug(f"Startup announcement error: {e}")

    # =========================================================================
    # Public API
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive helper status."""
        return {
            "state": self._state.value,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "uptime_seconds": self._stats.uptime_seconds,
            "components": {
                name: {
                    "running": status.running,
                    "healthy": status.healthy,
                    "error": status.error,
                    "restart_count": status.restart_count,
                }
                for name, status in self._component_status.items()
            },
            "stats": {
                "events_processed": self._stats.events_processed,
                "permissions_granted": self._stats.permissions_granted,
                "permissions_denied": self._stats.permissions_denied,
                "component_restarts": self._stats.component_restarts,
                "agi_events_bridged": self._stats.agi_events_bridged,
            },
        }

    def get_event_bus(self):
        """Get the event bus instance."""
        return self._event_bus

    def get_permission_manager(self):
        """Get the permission manager instance."""
        return self._permission_manager

    def get_system_monitor(self):
        """Get the system monitor instance."""
        return self._system_monitor

    def get_notification_monitor(self):
        """Get the notification monitor instance."""
        return self._notification_monitor

    def get_menu_bar(self):
        """Get the menu bar indicator instance."""
        return self._menu_bar

    def get_agi_bridge(self):
        """Get the AGI integration bridge instance."""
        return self._agi_bridge

    async def check_permissions(self):
        """Check all permissions (public API)."""
        if self._permission_manager:
            return await self._permission_manager.check_all_permissions()
        return None


# =============================================================================
# Singleton Pattern
# =============================================================================

_macos_helper: Optional[MacOSHelperCoordinator] = None


async def get_macos_helper(
    config: Optional[MacOSHelperConfig] = None
) -> MacOSHelperCoordinator:
    """
    Get the global macOS helper instance.

    Args:
        config: Helper configuration

    Returns:
        The MacOSHelperCoordinator singleton
    """
    global _macos_helper

    if _macos_helper is None:
        _macos_helper = MacOSHelperCoordinator(config)

    return _macos_helper


async def start_macos_helper(
    config: Optional[MacOSHelperConfig] = None
) -> MacOSHelperCoordinator:
    """
    Get and start the global macOS helper.

    Args:
        config: Helper configuration

    Returns:
        The started MacOSHelperCoordinator instance
    """
    helper = await get_macos_helper(config)
    if not helper.is_running:
        await helper.start()
    return helper


async def stop_macos_helper() -> None:
    """Stop the global macOS helper."""
    global _macos_helper

    if _macos_helper is not None:
        await _macos_helper.stop()
        _macos_helper = None


# =============================================================================
# CLI Entry Point (for LaunchAgent)
# =============================================================================

async def main():
    """Main entry point for running as a background service."""
    import argparse

    parser = argparse.ArgumentParser(description="JARVIS macOS Helper Service")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice feedback")
    parser.add_argument("--no-agi", action="store_true", help="Disable AGI OS integration")
    parser.add_argument("--no-menu-bar", action="store_true", help="Disable menu bar indicator")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create config
    config = MacOSHelperConfig(
        enable_voice_feedback=not args.no_voice,
        enable_agi_bridge=not args.no_agi,
        enable_menu_bar=not args.no_menu_bar,
        log_level=args.log_level,
    )

    # Start helper
    helper = await start_macos_helper(config)

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        asyncio.create_task(stop_macos_helper())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    # Keep running
    try:
        while helper.is_running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await stop_macos_helper()


if __name__ == "__main__":
    asyncio.run(main())
