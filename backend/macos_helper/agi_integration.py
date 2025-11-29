"""
JARVIS macOS Helper - AGI OS Integration Bridge

Deep integration between the macOS helper layer and the AGI OS coordinator.
Enables seamless bidirectional communication:

- macOS events flow into AGI OS event stream
- AGI OS commands trigger macOS actions
- Intelligence systems receive real-time macOS context
- Voice feedback routed through AGI OS voice communicator

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    AGI Integration Bridge                            │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  macOS Helper Layer          ←→          AGI OS Coordinator         │
    │  ─────────────────────                   ──────────────────────     │
    │  • Event Bus                  →          • ProactiveEventStream     │
    │  • Permission Manager         →          • Voice Communicator       │
    │  • System Monitor             →          • Action Orchestrator      │
    │  • Notification Monitor       →          • UAE/SAI/CAI Systems      │
    │  • Menu Bar                   ←          • Neural Mesh              │
    │                                                                      │
    │  Event Translation:                                                  │
    │  MacOSEvent → AGIEvent                                              │
    │  - APP_LAUNCHED → CONTEXT_CHANGED                                   │
    │  - NOTIFICATION_RECEIVED → OPPORTUNITY_DETECTED                     │
    │  - PERMISSION_DENIED → ISSUE_DETECTED                               │
    │  - IDLE_START → CONTEXT_CHANGED                                     │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘

Usage:
    from macos_helper.agi_integration import AGIBridge, get_agi_bridge

    # Start the bridge
    bridge = await start_agi_bridge()

    # Events now flow automatically between systems
    # AGI OS can request macOS actions:
    await bridge.request_focus_app("Cursor")
    await bridge.request_notification("Build complete!")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agi_os.agi_os_coordinator import AGIOSCoordinator
    from agi_os.proactive_event_stream import ProactiveEventStream
    from agi_os.realtime_voice_communicator import RealTimeVoiceCommunicator

logger = logging.getLogger(__name__)


# =============================================================================
# Event Translation Mappings
# =============================================================================

class AGIEventMapping:
    """Maps macOS events to AGI OS event types."""

    # Import lazily to avoid circular imports
    @staticmethod
    def get_agi_event_type(macos_event_type: str) -> str:
        """
        Map macOS event type to AGI OS event type.

        Args:
            macos_event_type: MacOSEventType value

        Returns:
            AGI EventType value
        """
        # Context change events
        context_events = {
            "app_launched",
            "app_activated",
            "app_deactivated",
            "app_terminated",
            "window_created",
            "window_closed",
            "window_focused",
            "window_moved",
            "window_resized",
            "space_changed",
            "display_changed",
            "idle_start",
            "idle_end",
            "screen_lock",
            "screen_unlock",
            "system_wake",
            "system_sleep",
        }

        # Opportunity events
        opportunity_events = {
            "notification_received",
            "notification_action_available",
            "file_created",
            "file_modified",
            "calendar_event",
            "reminder_due",
        }

        # Issue events
        issue_events = {
            "permission_denied",
            "permission_revoked",
            "app_crashed",
            "monitor_error",
        }

        # User action events
        user_events = {
            "notification_dismissed",
            "notification_clicked",
        }

        if macos_event_type in context_events:
            return "CONTEXT_CHANGED"
        elif macos_event_type in opportunity_events:
            return "OPPORTUNITY_DETECTED"
        elif macos_event_type in issue_events:
            return "ISSUE_DETECTED"
        elif macos_event_type in user_events:
            return "USER_ACTION"
        else:
            return "SYSTEM_STATUS"

    @staticmethod
    def get_agi_priority(macos_priority: str) -> str:
        """
        Map macOS event priority to AGI OS event priority.

        Args:
            macos_priority: MacOSEventPriority value

        Returns:
            AGI EventPriority value
        """
        priority_map = {
            "debug": "LOW",
            "low": "LOW",
            "normal": "MEDIUM",
            "high": "HIGH",
            "critical": "CRITICAL",
        }
        return priority_map.get(macos_priority, "MEDIUM")


# =============================================================================
# Bridge Configuration
# =============================================================================

@dataclass
class AGIBridgeConfig:
    """Configuration for the AGI integration bridge."""
    # Event bridging
    bridge_all_events: bool = True
    min_priority_to_bridge: str = "low"  # Minimum priority to forward to AGI

    # Voice integration
    enable_voice_feedback: bool = True
    voice_for_important_events: bool = True  # Announce high/critical events

    # Action forwarding
    enable_action_requests: bool = True
    auto_approve_safe_actions: bool = True

    # Intelligence integration
    send_context_to_uae: bool = True  # Send context to Unified Awareness Engine
    send_events_to_cai: bool = True   # Send events to Context Awareness Intelligence

    # Debouncing
    event_debounce_seconds: float = 0.5


@dataclass
class BridgeStats:
    """Statistics for the AGI bridge."""
    started_at: Optional[datetime] = None
    events_sent_to_agi: int = 0
    events_received_from_agi: int = 0
    actions_requested: int = 0
    actions_completed: int = 0
    voice_announcements: int = 0


# =============================================================================
# AGI Integration Bridge
# =============================================================================

class AGIBridge:
    """
    Bridge between macOS Helper and AGI OS.

    Handles:
    - Event translation and forwarding
    - Bidirectional command flow
    - Intelligence system integration
    - Voice feedback coordination
    """

    def __init__(self, config: Optional[AGIBridgeConfig] = None):
        """
        Initialize the AGI bridge.

        Args:
            config: Bridge configuration
        """
        self.config = config or AGIBridgeConfig()

        # AGI OS components (connected after start)
        self._agi_coordinator: Optional['AGIOSCoordinator'] = None
        self._event_stream: Optional['ProactiveEventStream'] = None
        self._voice: Optional['RealTimeVoiceCommunicator'] = None

        # macOS helper reference
        self._macos_helper = None

        # State
        self._running = False
        self._stats = BridgeStats()

        # Event debouncing
        self._last_events: Dict[str, datetime] = {}

        # Subscription cleanup
        self._subscriptions: List[Any] = []

        logger.debug("AGIBridge initialized")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> bool:
        """
        Start the AGI bridge.

        Connects to both macOS helper and AGI OS, setting up event forwarding.

        Returns:
            True if started successfully
        """
        if self._running:
            return True

        try:
            # Connect to AGI OS
            await self._connect_to_agi_os()

            # Connect to macOS helper
            await self._connect_to_macos_helper()

            # Set up event forwarding
            await self._setup_event_forwarding()

            # Set up action handling
            await self._setup_action_handling()

            self._running = True
            self._stats.started_at = datetime.now()

            logger.info("AGI Bridge started")
            return True

        except Exception as e:
            logger.error(f"Failed to start AGI Bridge: {e}")
            return False

    async def stop(self) -> None:
        """Stop the AGI bridge."""
        if not self._running:
            return

        # Clean up subscriptions
        for sub in self._subscriptions:
            try:
                if hasattr(sub, 'cancel'):
                    sub.cancel()
                elif hasattr(sub, 'unsubscribe'):
                    await sub.unsubscribe()
            except Exception as e:
                logger.warning(f"Error cleaning up subscription: {e}")

        self._subscriptions.clear()
        self._running = False

        logger.info("AGI Bridge stopped")

    # =========================================================================
    # Connection Setup
    # =========================================================================

    async def _connect_to_agi_os(self) -> None:
        """Connect to AGI OS coordinator and components."""
        try:
            from agi_os import get_agi_os
            self._agi_coordinator = await get_agi_os()

            # Get event stream
            self._event_stream = self._agi_coordinator.get_component('events')

            # Get voice communicator
            if self.config.enable_voice_feedback:
                self._voice = self._agi_coordinator.get_component('voice')

            logger.info("Connected to AGI OS")

        except ImportError:
            logger.info("AGI OS not available - running in standalone mode")
        except Exception as e:
            logger.warning(f"Could not connect to AGI OS: {e}")

    async def _connect_to_macos_helper(self) -> None:
        """Connect to macOS helper coordinator."""
        try:
            from .macos_helper_coordinator import get_macos_helper
            self._macos_helper = await get_macos_helper()
            logger.info("Connected to macOS Helper")

        except Exception as e:
            logger.warning(f"Could not connect to macOS Helper: {e}")

    async def _setup_event_forwarding(self) -> None:
        """Set up event forwarding from macOS helper to AGI OS."""
        if not self._macos_helper or not self._event_stream:
            return

        event_bus = self._macos_helper.get_event_bus()
        if not event_bus:
            return

        # Subscribe to all macOS events
        from .event_types import MacOSEventType

        async def forward_event(event):
            await self._forward_event_to_agi(event)

        # Subscribe to all event types
        for event_type in MacOSEventType:
            try:
                sub = event_bus.subscribe(
                    event_type,
                    forward_event,
                    priority=0  # High priority to ensure forwarding
                )
                self._subscriptions.append(sub)
            except Exception as e:
                logger.warning(f"Failed to subscribe to {event_type}: {e}")

        logger.info("Event forwarding configured")

    async def _setup_action_handling(self) -> None:
        """Set up action request handling from AGI OS."""
        if not self._agi_coordinator or not self._macos_helper:
            return

        # Register macOS action handler with AGI OS
        orchestrator = self._agi_coordinator.get_component('orchestrator')
        if orchestrator:
            # Register handlers for macOS-specific actions
            orchestrator.register_action_handler(
                'focus_app',
                self._handle_focus_app_request
            )
            orchestrator.register_action_handler(
                'show_notification',
                self._handle_notification_request
            )
            orchestrator.register_action_handler(
                'open_url',
                self._handle_open_url_request
            )

        logger.info("Action handling configured")

    # =========================================================================
    # Event Forwarding
    # =========================================================================

    async def _forward_event_to_agi(self, event) -> None:
        """
        Forward a macOS event to AGI OS.

        Args:
            event: MacOSEvent to forward
        """
        if not self._event_stream:
            return

        # Check debouncing
        event_key = f"{event.event_type.value}:{event.source}"
        now = datetime.now()
        last_time = self._last_events.get(event_key)

        if last_time:
            elapsed = (now - last_time).total_seconds()
            if elapsed < self.config.event_debounce_seconds:
                return  # Skip debounced event

        self._last_events[event_key] = now

        # Check priority threshold
        if self.config.min_priority_to_bridge:
            priority_order = ["debug", "low", "normal", "high", "critical"]
            event_priority = event.priority.value if hasattr(event.priority, 'value') else event.priority
            min_priority = self.config.min_priority_to_bridge

            if priority_order.index(event_priority) < priority_order.index(min_priority):
                return  # Below threshold

        try:
            # Translate to AGI event
            from agi_os.proactive_event_stream import AGIEvent, EventType, EventPriority

            agi_event_type = AGIEventMapping.get_agi_event_type(event.event_type.value)
            agi_priority = AGIEventMapping.get_agi_priority(
                event.priority.value if hasattr(event.priority, 'value') else event.priority
            )

            agi_event = AGIEvent(
                event_type=EventType[agi_event_type],
                source=f"macos_helper:{event.source}",
                data={
                    'macos_event_type': event.event_type.value,
                    'original_data': event.data,
                    'timestamp': event.timestamp.isoformat() if event.timestamp else None,
                },
                priority=EventPriority[agi_priority],
                description=self._generate_event_description(event),
            )

            # Emit to AGI OS event stream
            await self._event_stream.emit(agi_event)
            self._stats.events_sent_to_agi += 1

            # Voice announcement for important events
            if (
                self.config.voice_for_important_events
                and self._voice
                and agi_priority in ["HIGH", "CRITICAL"]
            ):
                await self._announce_event(event)

            # Send context to intelligence systems
            if self.config.send_context_to_uae:
                await self._update_uae_context(event)

        except Exception as e:
            logger.warning(f"Failed to forward event to AGI: {e}")

    def _generate_event_description(self, event) -> str:
        """Generate a human-readable description for an event."""
        event_type = event.event_type.value
        data = event.data or {}

        descriptions = {
            "app_launched": f"App launched: {data.get('app_name', 'Unknown')}",
            "app_activated": f"Switched to: {data.get('app_name', 'Unknown')}",
            "notification_received": f"Notification from {data.get('app_name', 'Unknown')}: {data.get('title', '')}",
            "permission_denied": f"Permission denied: {data.get('permission_type', 'Unknown')}",
            "idle_start": "User went idle",
            "idle_end": "User returned from idle",
            "space_changed": f"Moved to space {data.get('space_id', '?')}",
        }

        return descriptions.get(event_type, f"macOS event: {event_type}")

    async def _announce_event(self, event) -> None:
        """Announce an important event via voice."""
        if not self._voice:
            return

        try:
            from agi_os.realtime_voice_communicator import VoiceMode

            description = self._generate_event_description(event)
            await self._voice.speak(description, mode=VoiceMode.NOTIFICATION)
            self._stats.voice_announcements += 1

        except Exception as e:
            logger.warning(f"Failed to announce event: {e}")

    async def _update_uae_context(self, event) -> None:
        """Update UAE (Unified Awareness Engine) with new context."""
        if not self._agi_coordinator:
            return

        uae = self._agi_coordinator.get_component('uae')
        if not uae:
            return

        try:
            # Build context update based on event type
            event_type = event.event_type.value
            data = event.data or {}

            context_updates = {}

            if event_type in ["app_launched", "app_activated"]:
                context_updates['active_app'] = data.get('app_name')
                context_updates['active_bundle_id'] = data.get('bundle_id')

            elif event_type == "space_changed":
                context_updates['current_space'] = data.get('space_id')

            elif event_type in ["idle_start", "idle_end"]:
                context_updates['user_idle'] = (event_type == "idle_start")

            elif event_type == "notification_received":
                # Add to notification context
                if 'recent_notifications' not in context_updates:
                    context_updates['recent_notifications'] = []
                context_updates['recent_notifications'].append({
                    'app': data.get('app_name'),
                    'title': data.get('title'),
                    'timestamp': datetime.now().isoformat(),
                })

            if context_updates and hasattr(uae, 'update_context'):
                await uae.update_context(context_updates)

        except Exception as e:
            logger.debug(f"Failed to update UAE context: {e}")

    # =========================================================================
    # Action Request Handling
    # =========================================================================

    async def _handle_focus_app_request(self, action_data: Dict) -> bool:
        """
        Handle request to focus an app.

        Args:
            action_data: Action parameters including app_name or bundle_id

        Returns:
            True if successful
        """
        if not self._macos_helper:
            return False

        system_monitor = self._macos_helper.get_system_monitor()
        if not system_monitor:
            return False

        try:
            app_name = action_data.get('app_name')
            bundle_id = action_data.get('bundle_id')

            # Use system control to focus the app
            from system_control.macos_controller import MacOSController
            controller = MacOSController()

            if bundle_id:
                success = await controller.activate_app(bundle_id)
            elif app_name:
                success = await controller.activate_app_by_name(app_name)
            else:
                return False

            self._stats.actions_completed += 1
            return success

        except Exception as e:
            logger.error(f"Failed to focus app: {e}")
            return False

    async def _handle_notification_request(self, action_data: Dict) -> bool:
        """
        Handle request to show a notification.

        Args:
            action_data: Action parameters including title, message

        Returns:
            True if successful
        """
        try:
            title = action_data.get('title', 'JARVIS')
            message = action_data.get('message', '')
            subtitle = action_data.get('subtitle')

            # Use menu bar if available
            if self._macos_helper:
                menu_bar = self._macos_helper.get_menu_bar()
                if menu_bar:
                    menu_bar.show_notification(title, message, subtitle)
                    self._stats.actions_completed += 1
                    return True

            # Fallback to AppleScript
            import subprocess
            script = f'''display notification "{message}" with title "{title}"'''
            if subtitle:
                script = f'''display notification "{message}" with title "{title}" subtitle "{subtitle}"'''

            subprocess.run(['osascript', '-e', script], capture_output=True)
            self._stats.actions_completed += 1
            return True

        except Exception as e:
            logger.error(f"Failed to show notification: {e}")
            return False

    async def _handle_open_url_request(self, action_data: Dict) -> bool:
        """
        Handle request to open a URL.

        Args:
            action_data: Action parameters including url

        Returns:
            True if successful
        """
        try:
            url = action_data.get('url')
            if not url:
                return False

            import subprocess
            subprocess.run(['open', url], capture_output=True)
            self._stats.actions_completed += 1
            return True

        except Exception as e:
            logger.error(f"Failed to open URL: {e}")
            return False

    # =========================================================================
    # Public API
    # =========================================================================

    async def request_focus_app(self, app_name: str) -> bool:
        """
        Request to focus an application.

        Args:
            app_name: Name of the app to focus

        Returns:
            True if successful
        """
        self._stats.actions_requested += 1
        return await self._handle_focus_app_request({'app_name': app_name})

    async def request_notification(
        self,
        title: str,
        message: str,
        subtitle: Optional[str] = None
    ) -> bool:
        """
        Request to show a notification.

        Args:
            title: Notification title
            message: Notification message
            subtitle: Optional subtitle

        Returns:
            True if successful
        """
        self._stats.actions_requested += 1
        return await self._handle_notification_request({
            'title': title,
            'message': message,
            'subtitle': subtitle,
        })

    async def request_open_url(self, url: str) -> bool:
        """
        Request to open a URL.

        Args:
            url: URL to open

        Returns:
            True if successful
        """
        self._stats.actions_requested += 1
        return await self._handle_open_url_request({'url': url})

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            'started_at': self._stats.started_at.isoformat() if self._stats.started_at else None,
            'running': self._running,
            'events_sent_to_agi': self._stats.events_sent_to_agi,
            'events_received_from_agi': self._stats.events_received_from_agi,
            'actions_requested': self._stats.actions_requested,
            'actions_completed': self._stats.actions_completed,
            'voice_announcements': self._stats.voice_announcements,
            'config': {
                'bridge_all_events': self.config.bridge_all_events,
                'voice_feedback': self.config.enable_voice_feedback,
                'action_requests': self.config.enable_action_requests,
            },
        }


# =============================================================================
# Singleton Management
# =============================================================================

_agi_bridge: Optional[AGIBridge] = None


async def get_agi_bridge(
    config: Optional[AGIBridgeConfig] = None
) -> AGIBridge:
    """
    Get the global AGI bridge instance.

    Args:
        config: Bridge configuration

    Returns:
        The AGIBridge singleton
    """
    global _agi_bridge

    if _agi_bridge is None:
        _agi_bridge = AGIBridge(config)

    return _agi_bridge


async def start_agi_bridge(
    config: Optional[AGIBridgeConfig] = None
) -> AGIBridge:
    """
    Get and start the global AGI bridge.

    Args:
        config: Bridge configuration

    Returns:
        The started AGIBridge instance
    """
    bridge = await get_agi_bridge(config)
    if not bridge._running:
        await bridge.start()
    return bridge


async def stop_agi_bridge() -> None:
    """Stop the global AGI bridge."""
    global _agi_bridge

    if _agi_bridge is not None:
        await _agi_bridge.stop()
        _agi_bridge = None
