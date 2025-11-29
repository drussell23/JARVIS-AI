"""
JARVIS macOS Helper - Intelligence Coordinator

Central orchestrator that coordinates all Phase 2 intelligence components,
managing data flow, cross-component communication, and unified APIs.

Features:
- Unified startup/shutdown of all intelligence components
- Cross-component event routing
- Data fusion from multiple sources
- UAE (Unified Awareness Engine) integration
- AGI OS coordination
- Health monitoring and error recovery
- Configuration management

Architecture:
    IntelligenceCoordinator
    ├── ScreenContextAnalyzer (real-time screen understanding)
    ├── ProactiveSuggestionEngine (ML-based suggestions)
    ├── NotificationTriageSystem (smart notification routing)
    ├── FocusTracker (productivity insights)
    ├── UAEBridge (context fusion)
    └── AGIBridge (AGI OS integration)

Data Flow:
    Screen Events → ScreenContextAnalyzer → Context
                                          ↓
    Notifications → NotificationTriage → Filtered Notifs
                                          ↓
    All Context → ProactiveSuggestionEngine → Suggestions
            ↓
    Activity Data → FocusTracker → Productivity Insights
            ↓
    All Data → UAE Bridge → Unified Context → AGI OS
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class IntelligenceConfig:
    """Configuration for the intelligence coordinator."""
    # Component toggles
    enable_screen_context: bool = os.getenv("INTEL_ENABLE_SCREEN", "true").lower() == "true"
    enable_suggestions: bool = os.getenv("INTEL_ENABLE_SUGGESTIONS", "true").lower() == "true"
    enable_notification_triage: bool = os.getenv("INTEL_ENABLE_TRIAGE", "true").lower() == "true"
    enable_focus_tracking: bool = os.getenv("INTEL_ENABLE_FOCUS", "true").lower() == "true"

    # Integration toggles
    enable_uae_integration: bool = os.getenv("INTEL_ENABLE_UAE", "true").lower() == "true"
    enable_agi_integration: bool = os.getenv("INTEL_ENABLE_AGI", "true").lower() == "true"

    # Health monitoring
    health_check_interval_seconds: float = float(os.getenv("INTEL_HEALTH_INTERVAL", "60.0"))
    max_component_restart_attempts: int = int(os.getenv("INTEL_MAX_RESTARTS", "3"))

    # Data flow
    context_update_interval_seconds: float = float(os.getenv("INTEL_CONTEXT_INTERVAL", "5.0"))
    event_buffer_size: int = int(os.getenv("INTEL_EVENT_BUFFER", "100"))

    # Logging
    verbose_logging: bool = os.getenv("INTEL_VERBOSE", "false").lower() == "true"


# =============================================================================
# Component Status
# =============================================================================

@dataclass
class ComponentStatus:
    """Status of an intelligence component."""
    name: str
    is_running: bool = False
    is_healthy: bool = True
    started_at: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    error_count: int = 0
    restart_count: int = 0
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "is_running": self.is_running,
            "is_healthy": self.is_healthy,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "error_count": self.error_count,
            "restart_count": self.restart_count,
            "last_error": self.last_error,
        }


# =============================================================================
# Intelligence Coordinator
# =============================================================================

class IntelligenceCoordinator:
    """
    Central coordinator for all intelligence components.

    Manages lifecycle, data flow, and integration between:
    - Screen Context Analyzer
    - Proactive Suggestion Engine
    - Notification Triage System
    - Focus Tracker
    - UAE Integration
    - AGI OS Bridge
    """

    def __init__(self, config: Optional[IntelligenceConfig] = None):
        """
        Initialize the intelligence coordinator.

        Args:
            config: Coordinator configuration
        """
        self.config = config or IntelligenceConfig()

        # State
        self._running = False
        self._started_at: Optional[datetime] = None

        # Components (lazy loaded)
        self._screen_analyzer = None
        self._suggestion_engine = None
        self._notification_triage = None
        self._focus_tracker = None

        # Component status tracking
        self._component_status: Dict[str, ComponentStatus] = {
            "screen_context": ComponentStatus(name="screen_context"),
            "suggestions": ComponentStatus(name="suggestions"),
            "notification_triage": ComponentStatus(name="notification_triage"),
            "focus_tracker": ComponentStatus(name="focus_tracker"),
        }

        # Cross-component data
        self._current_context: Dict[str, Any] = {}
        self._recent_events: List[Dict[str, Any]] = []

        # External callbacks
        self._on_suggestion: List[Callable[[Any], Coroutine]] = []
        self._on_notification: List[Callable[[Any], Coroutine]] = []
        self._on_insight: List[Callable[[Any], Coroutine]] = []
        self._on_context_changed: List[Callable[[Dict[str, Any]], Coroutine]] = []

        # Background tasks
        self._health_task: Optional[asyncio.Task] = None
        self._context_task: Optional[asyncio.Task] = None

        # Stats
        self._stats = {
            "events_processed": 0,
            "suggestions_generated": 0,
            "notifications_triaged": 0,
            "context_updates": 0,
        }

        logger.debug("IntelligenceCoordinator initialized")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> bool:
        """
        Start the intelligence layer.

        Initializes and starts all enabled components.

        Returns:
            True if started successfully
        """
        if self._running:
            return True

        try:
            logger.info("Starting Intelligence Layer...")
            self._running = True
            self._started_at = datetime.now()

            # Start components in order
            await self._start_components()

            # Setup cross-component wiring
            await self._setup_component_wiring()

            # Start background tasks
            self._health_task = asyncio.create_task(
                self._health_check_loop(),
                name="intelligence_health"
            )

            self._context_task = asyncio.create_task(
                self._context_fusion_loop(),
                name="intelligence_context"
            )

            logger.info("Intelligence Layer started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start Intelligence Layer: {e}")
            self._running = False
            return False

    async def stop(self) -> None:
        """Stop the intelligence layer and all components."""
        if not self._running:
            return

        logger.info("Stopping Intelligence Layer...")
        self._running = False

        # Cancel background tasks
        for task in [self._health_task, self._context_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Stop components
        await self._stop_components()

        logger.info("Intelligence Layer stopped")

    # =========================================================================
    # Component Management
    # =========================================================================

    async def _start_components(self) -> None:
        """Start all enabled components."""
        # Start Screen Context Analyzer
        if self.config.enable_screen_context:
            await self._start_screen_analyzer()

        # Start Proactive Suggestion Engine
        if self.config.enable_suggestions:
            await self._start_suggestion_engine()

        # Start Notification Triage
        if self.config.enable_notification_triage:
            await self._start_notification_triage()

        # Start Focus Tracker
        if self.config.enable_focus_tracking:
            await self._start_focus_tracker()

    async def _start_screen_analyzer(self) -> None:
        """Start the screen context analyzer."""
        try:
            from .screen_context_analyzer import (
                ScreenContextAnalyzer,
                ScreenContextConfig,
            )

            self._screen_analyzer = ScreenContextAnalyzer(ScreenContextConfig())
            if await self._screen_analyzer.start():
                self._component_status["screen_context"].is_running = True
                self._component_status["screen_context"].started_at = datetime.now()
                logger.info("Screen Context Analyzer started")
            else:
                raise Exception("Failed to start Screen Context Analyzer")

        except ImportError as e:
            logger.warning(f"Screen Context Analyzer not available: {e}")
            self._component_status["screen_context"].last_error = str(e)
        except Exception as e:
            logger.error(f"Error starting Screen Context Analyzer: {e}")
            self._component_status["screen_context"].last_error = str(e)
            self._component_status["screen_context"].error_count += 1

    async def _start_suggestion_engine(self) -> None:
        """Start the proactive suggestion engine."""
        try:
            from .proactive_suggestion_engine import (
                ProactiveSuggestionEngine,
                SuggestionEngineConfig,
            )

            self._suggestion_engine = ProactiveSuggestionEngine(SuggestionEngineConfig())
            if await self._suggestion_engine.start():
                self._component_status["suggestions"].is_running = True
                self._component_status["suggestions"].started_at = datetime.now()
                logger.info("Proactive Suggestion Engine started")
            else:
                raise Exception("Failed to start Suggestion Engine")

        except ImportError as e:
            logger.warning(f"Suggestion Engine not available: {e}")
            self._component_status["suggestions"].last_error = str(e)
        except Exception as e:
            logger.error(f"Error starting Suggestion Engine: {e}")
            self._component_status["suggestions"].last_error = str(e)
            self._component_status["suggestions"].error_count += 1

    async def _start_notification_triage(self) -> None:
        """Start the notification triage system."""
        try:
            from .notification_triage import (
                NotificationTriageSystem,
                NotificationTriageConfig,
            )

            self._notification_triage = NotificationTriageSystem(NotificationTriageConfig())
            if await self._notification_triage.start():
                self._component_status["notification_triage"].is_running = True
                self._component_status["notification_triage"].started_at = datetime.now()
                logger.info("Notification Triage System started")
            else:
                raise Exception("Failed to start Notification Triage")

        except ImportError as e:
            logger.warning(f"Notification Triage not available: {e}")
            self._component_status["notification_triage"].last_error = str(e)
        except Exception as e:
            logger.error(f"Error starting Notification Triage: {e}")
            self._component_status["notification_triage"].last_error = str(e)
            self._component_status["notification_triage"].error_count += 1

    async def _start_focus_tracker(self) -> None:
        """Start the focus tracker."""
        try:
            from .focus_tracker import (
                FocusTracker,
                FocusTrackerConfig,
            )

            self._focus_tracker = FocusTracker(FocusTrackerConfig())
            if await self._focus_tracker.start():
                self._component_status["focus_tracker"].is_running = True
                self._component_status["focus_tracker"].started_at = datetime.now()
                logger.info("Focus Tracker started")
            else:
                raise Exception("Failed to start Focus Tracker")

        except ImportError as e:
            logger.warning(f"Focus Tracker not available: {e}")
            self._component_status["focus_tracker"].last_error = str(e)
        except Exception as e:
            logger.error(f"Error starting Focus Tracker: {e}")
            self._component_status["focus_tracker"].last_error = str(e)
            self._component_status["focus_tracker"].error_count += 1

    async def _stop_components(self) -> None:
        """Stop all components."""
        if self._screen_analyzer:
            try:
                await self._screen_analyzer.stop()
            except Exception as e:
                logger.error(f"Error stopping Screen Analyzer: {e}")

        if self._suggestion_engine:
            try:
                await self._suggestion_engine.stop()
            except Exception as e:
                logger.error(f"Error stopping Suggestion Engine: {e}")

        if self._notification_triage:
            try:
                await self._notification_triage.stop()
            except Exception as e:
                logger.error(f"Error stopping Notification Triage: {e}")

        if self._focus_tracker:
            try:
                await self._focus_tracker.stop()
            except Exception as e:
                logger.error(f"Error stopping Focus Tracker: {e}")

        # Update status
        for status in self._component_status.values():
            status.is_running = False

    # =========================================================================
    # Component Wiring
    # =========================================================================

    async def _setup_component_wiring(self) -> None:
        """Setup cross-component event routing."""
        # Wire Screen Context → Suggestion Engine & Focus Tracker
        if self._screen_analyzer:
            self._screen_analyzer.on_context_changed(self._on_screen_context_changed)
            self._screen_analyzer.on_activity_changed(self._on_activity_changed)

        # Wire Suggestion Engine → External callbacks
        if self._suggestion_engine:
            self._suggestion_engine.on_suggestion(self._on_suggestion_generated)

        # Wire Notification Triage → External callbacks
        if self._notification_triage:
            self._notification_triage.on_notification_ready(self._on_notification_ready)
            self._notification_triage.on_batch_ready(self._on_batch_ready)

        # Wire Focus Tracker → Suggestion Engine & External callbacks
        if self._focus_tracker:
            self._focus_tracker.on_focus_changed(self._on_focus_state_changed)
            self._focus_tracker.on_insight_generated(self._on_focus_insight)
            self._focus_tracker.on_break_recommended(self._on_break_recommended)

    async def _on_screen_context_changed(self, change) -> None:
        """Handle screen context change."""
        self._stats["events_processed"] += 1

        # Update current context
        if change.current_context:
            self._current_context["screen"] = change.current_context.to_dict()
            self._current_context["activity_type"] = change.current_context.activity_type.value
            self._current_context["active_app"] = change.current_context.active_app

        # Forward to suggestion engine
        if self._suggestion_engine and change.current_context:
            self._suggestion_engine.update_context(
                activity_type=change.current_context.activity_type.value,
                app_name=change.current_context.active_app,
                context=self._current_context,
            )

        # Forward to focus tracker
        if self._focus_tracker and change.current_context:
            self._focus_tracker.update_activity(
                app_name=change.current_context.active_app,
                window_title=change.current_context.window_title,
                is_idle=change.current_context.is_idle,
            )

        # Notify external callbacks
        for callback in self._on_context_changed:
            try:
                await callback(self._current_context)
            except Exception as e:
                logger.error(f"Context changed callback error: {e}")

    async def _on_activity_changed(self, old_activity, new_activity) -> None:
        """Handle activity type change."""
        if self.config.verbose_logging:
            logger.debug(f"Activity changed: {old_activity} → {new_activity}")

        # Update context
        self._current_context["previous_activity"] = old_activity.value
        self._current_context["current_activity"] = new_activity.value

    async def _on_suggestion_generated(self, suggestion) -> None:
        """Handle new suggestion from engine."""
        self._stats["suggestions_generated"] += 1

        # Forward to external callbacks
        for callback in self._on_suggestion:
            try:
                await callback(suggestion)
            except Exception as e:
                logger.error(f"Suggestion callback error: {e}")

    async def _on_notification_ready(self, notification) -> None:
        """Handle notification ready for delivery."""
        self._stats["notifications_triaged"] += 1

        # Forward to external callbacks
        for callback in self._on_notification:
            try:
                await callback(notification)
            except Exception as e:
                logger.error(f"Notification callback error: {e}")

    async def _on_batch_ready(self, batch) -> None:
        """Handle notification batch ready for delivery."""
        # Forward each notification in batch
        for notification in batch.notifications:
            for callback in self._on_notification:
                try:
                    await callback(notification)
                except Exception as e:
                    logger.error(f"Batch notification callback error: {e}")

    async def _on_focus_state_changed(self, old_state, new_state) -> None:
        """Handle focus state change."""
        if self.config.verbose_logging:
            logger.debug(f"Focus changed: {old_state} → {new_state}")

        # Update context
        self._current_context["focus_state"] = new_state.value

        # Update suggestion engine focus awareness
        if self._suggestion_engine:
            self._suggestion_engine.update_context(
                context={"focus_state": new_state.value}
            )

        # Update notification triage focus mode
        if self._notification_triage:
            from .notification_triage import FocusMode
            from .focus_tracker import FocusState

            # Map focus state to notification focus mode
            mode_map = {
                FocusState.DEEP_FOCUS: FocusMode.WORK,
                FocusState.MEETING: FocusMode.DND,
                FocusState.BREAK: FocusMode.PERSONAL,
            }
            focus_mode = mode_map.get(new_state, FocusMode.NORMAL)
            self._notification_triage.set_focus_mode(focus_mode)

    async def _on_focus_insight(self, insight) -> None:
        """Handle productivity insight from focus tracker."""
        # Forward to external callbacks
        for callback in self._on_insight:
            try:
                await callback(insight)
            except Exception as e:
                logger.error(f"Insight callback error: {e}")

    async def _on_break_recommended(self, recommendation, message) -> None:
        """Handle break recommendation."""
        # Could trigger a suggestion
        if self._suggestion_engine:
            from .proactive_suggestion_engine import SuggestionType, SuggestionPriority

            self._suggestion_engine.manually_trigger_suggestion(
                suggestion_type=SuggestionType.HEALTH,
                title="Break Recommended",
                message=message,
                priority=SuggestionPriority.MEDIUM,
            )

    # =========================================================================
    # Background Tasks
    # =========================================================================

    async def _health_check_loop(self) -> None:
        """Periodic health check of all components."""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval_seconds)
                await self._check_component_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _check_component_health(self) -> None:
        """Check health of all components and restart if needed."""
        now = datetime.now()

        # Check Screen Analyzer
        if self.config.enable_screen_context:
            status = self._component_status["screen_context"]
            if self._screen_analyzer:
                is_healthy = self._screen_analyzer._running
                status.is_healthy = is_healthy
                status.last_health_check = now

                if not is_healthy and status.restart_count < self.config.max_component_restart_attempts:
                    logger.warning("Screen Analyzer unhealthy, attempting restart...")
                    await self._restart_screen_analyzer()

        # Check Suggestion Engine
        if self.config.enable_suggestions:
            status = self._component_status["suggestions"]
            if self._suggestion_engine:
                is_healthy = self._suggestion_engine._running
                status.is_healthy = is_healthy
                status.last_health_check = now

                if not is_healthy and status.restart_count < self.config.max_component_restart_attempts:
                    logger.warning("Suggestion Engine unhealthy, attempting restart...")
                    await self._restart_suggestion_engine()

        # Check Notification Triage
        if self.config.enable_notification_triage:
            status = self._component_status["notification_triage"]
            if self._notification_triage:
                is_healthy = self._notification_triage._running
                status.is_healthy = is_healthy
                status.last_health_check = now

                if not is_healthy and status.restart_count < self.config.max_component_restart_attempts:
                    logger.warning("Notification Triage unhealthy, attempting restart...")
                    await self._restart_notification_triage()

        # Check Focus Tracker
        if self.config.enable_focus_tracking:
            status = self._component_status["focus_tracker"]
            if self._focus_tracker:
                is_healthy = self._focus_tracker._running
                status.is_healthy = is_healthy
                status.last_health_check = now

                if not is_healthy and status.restart_count < self.config.max_component_restart_attempts:
                    logger.warning("Focus Tracker unhealthy, attempting restart...")
                    await self._restart_focus_tracker()

    async def _restart_screen_analyzer(self) -> None:
        """Restart the screen analyzer."""
        if self._screen_analyzer:
            await self._screen_analyzer.stop()
        await self._start_screen_analyzer()
        self._component_status["screen_context"].restart_count += 1

    async def _restart_suggestion_engine(self) -> None:
        """Restart the suggestion engine."""
        if self._suggestion_engine:
            await self._suggestion_engine.stop()
        await self._start_suggestion_engine()
        self._component_status["suggestions"].restart_count += 1

    async def _restart_notification_triage(self) -> None:
        """Restart the notification triage."""
        if self._notification_triage:
            await self._notification_triage.stop()
        await self._start_notification_triage()
        self._component_status["notification_triage"].restart_count += 1

    async def _restart_focus_tracker(self) -> None:
        """Restart the focus tracker."""
        if self._focus_tracker:
            await self._focus_tracker.stop()
        await self._start_focus_tracker()
        self._component_status["focus_tracker"].restart_count += 1

    async def _context_fusion_loop(self) -> None:
        """Periodic context fusion and propagation."""
        while self._running:
            try:
                await asyncio.sleep(self.config.context_update_interval_seconds)
                await self._fuse_context()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Context fusion error: {e}")

    async def _fuse_context(self) -> None:
        """Fuse context from all sources."""
        self._stats["context_updates"] += 1

        # Gather context from all sources
        fused = {"timestamp": datetime.now().isoformat()}

        # Screen context
        if self._screen_analyzer:
            ctx = self._screen_analyzer.get_current_context()
            if ctx:
                fused["screen"] = {
                    "app": ctx.active_app,
                    "activity": ctx.activity_type.value,
                    "is_idle": ctx.is_idle,
                }

        # Focus state
        if self._focus_tracker:
            fused["focus"] = {
                "state": self._focus_tracker.get_current_focus_state().value,
                "score": self._focus_tracker.get_focus_score(),
                "productivity": self._focus_tracker.get_productivity_score(),
            }

        # Notification state
        if self._notification_triage:
            fused["notifications"] = {
                "pending_count": self._notification_triage.get_pending_count(),
                "focus_mode": self._notification_triage.get_focus_mode().value,
            }

        # Suggestion state
        if self._suggestion_engine:
            fused["suggestions"] = {
                "pending": len(self._suggestion_engine.get_pending_suggestions()),
                "shown": len(self._suggestion_engine.get_shown_suggestions()),
            }

        self._current_context = fused

    # =========================================================================
    # Public API
    # =========================================================================

    def get_current_context(self) -> Dict[str, Any]:
        """Get the current fused context."""
        return self._current_context.copy()

    def get_component_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all components."""
        return {
            name: status.to_dict()
            for name, status in self._component_status.items()
        }

    def is_healthy(self) -> bool:
        """Check if intelligence layer is healthy."""
        if not self._running:
            return False

        # Check that at least one component is running and healthy
        return any(
            status.is_running and status.is_healthy
            for status in self._component_status.values()
        )

    def on_suggestion(
        self,
        callback: Callable[[Any], Coroutine]
    ) -> None:
        """Register callback for suggestions."""
        self._on_suggestion.append(callback)

    def on_notification(
        self,
        callback: Callable[[Any], Coroutine]
    ) -> None:
        """Register callback for triaged notifications."""
        self._on_notification.append(callback)

    def on_insight(
        self,
        callback: Callable[[Any], Coroutine]
    ) -> None:
        """Register callback for productivity insights."""
        self._on_insight.append(callback)

    def on_context_changed(
        self,
        callback: Callable[[Dict[str, Any]], Coroutine]
    ) -> None:
        """Register callback for context changes."""
        self._on_context_changed.append(callback)

    # Component accessors
    def get_screen_analyzer(self):
        """Get screen context analyzer instance."""
        return self._screen_analyzer

    def get_suggestion_engine(self):
        """Get suggestion engine instance."""
        return self._suggestion_engine

    def get_notification_triage(self):
        """Get notification triage instance."""
        return self._notification_triage

    def get_focus_tracker(self):
        """Get focus tracker instance."""
        return self._focus_tracker

    async def triage_notification(
        self,
        app_name: str,
        title: str,
        body: str,
        **kwargs
    ):
        """
        Triage an incoming notification.

        Convenience method to access notification triage.
        """
        if self._notification_triage:
            return await self._notification_triage.triage_notification(
                app_name=app_name,
                title=title,
                body=body,
                context=self._current_context,
                **kwargs
            )
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        component_stats = {}

        if self._screen_analyzer:
            component_stats["screen_analyzer"] = self._screen_analyzer.get_stats()

        if self._suggestion_engine:
            component_stats["suggestion_engine"] = self._suggestion_engine.get_stats()

        if self._notification_triage:
            component_stats["notification_triage"] = self._notification_triage.get_stats()

        if self._focus_tracker:
            component_stats["focus_tracker"] = self._focus_tracker.get_stats()

        return {
            "running": self._running,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "is_healthy": self.is_healthy(),
            "components": {name: s.to_dict() for name, s in self._component_status.items()},
            "component_stats": component_stats,
            **self._stats,
        }


# =============================================================================
# Singleton Management
# =============================================================================

_intelligence_coordinator: Optional[IntelligenceCoordinator] = None


async def get_intelligence_coordinator(
    config: Optional[IntelligenceConfig] = None
) -> IntelligenceCoordinator:
    """Get the global intelligence coordinator instance."""
    global _intelligence_coordinator

    if _intelligence_coordinator is None:
        _intelligence_coordinator = IntelligenceCoordinator(config)

    return _intelligence_coordinator


async def start_intelligence_layer(
    config: Optional[IntelligenceConfig] = None
) -> IntelligenceCoordinator:
    """
    Start the intelligence layer.

    This is the main entry point for Phase 2 intelligence.

    Args:
        config: Optional configuration

    Returns:
        Running IntelligenceCoordinator instance
    """
    coordinator = await get_intelligence_coordinator(config)
    if not coordinator._running:
        await coordinator.start()
    return coordinator


async def stop_intelligence_layer() -> None:
    """Stop the intelligence layer."""
    global _intelligence_coordinator

    if _intelligence_coordinator is not None:
        await _intelligence_coordinator.stop()
        _intelligence_coordinator = None
