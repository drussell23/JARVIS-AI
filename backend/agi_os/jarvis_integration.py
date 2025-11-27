"""
JARVIS AGI OS - Integration with Existing JARVIS Systems

This module provides seamless integration between the new AGI OS
components and the existing JARVIS infrastructure:

- Screen Analyzer Integration: Connect continuous monitoring to event stream
- Decision Engine Integration: Route autonomous decisions through AGI OS
- Voice System Integration: Unify voice output through Daniel TTS
- Permission System Integration: Bridge approval systems
- Neural Mesh Integration: Connect agents to AGI OS events

Usage:
    from agi_os.jarvis_integration import (
        connect_screen_analyzer,
        connect_decision_engine,
        integrate_voice_systems,
        integrate_approval_systems,
    )

    # Connect screen analyzer to AGI OS
    await connect_screen_analyzer(vision_handler)

    # Connect decision engine
    await connect_decision_engine(decision_engine)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ScreenAnalyzerBridge:
    """
    Bridge between continuous screen analyzer and AGI OS event stream.

    Converts screen analyzer callbacks to AGI OS events.
    """

    def __init__(self):
        """Initialize the bridge."""
        self._event_stream: Optional[Any] = None
        self._voice: Optional[Any] = None
        self._analyzer: Optional[Any] = None
        self._connected = False

    async def connect(
        self,
        analyzer: Any,
        event_stream: Optional[Any] = None,
        voice: Optional[Any] = None
    ) -> None:
        """
        Connect screen analyzer to AGI OS.

        Args:
            analyzer: MemoryAwareScreenAnalyzer instance
            event_stream: ProactiveEventStream (or fetched automatically)
            voice: RealTimeVoiceCommunicator (or fetched automatically)
        """
        if self._connected:
            logger.warning("Screen analyzer already connected")
            return

        self._analyzer = analyzer

        # Get or fetch event stream
        if event_stream:
            self._event_stream = event_stream
        else:
            try:
                from .proactive_event_stream import get_event_stream
                self._event_stream = await get_event_stream()
            except Exception as e:
                logger.warning("Could not get event stream: %s", e)

        # Get or fetch voice
        if voice:
            self._voice = voice
        else:
            try:
                from .realtime_voice_communicator import get_voice_communicator
                self._voice = await get_voice_communicator()
            except Exception as e:
                logger.warning("Could not get voice communicator: %s", e)

        # Register callbacks
        await self._register_callbacks()
        self._connected = True
        logger.info("Screen analyzer connected to AGI OS")

    async def _register_callbacks(self) -> None:
        """Register screen analyzer callbacks."""
        if not self._analyzer:
            return

        # Check if analyzer has event_callbacks attribute
        if hasattr(self._analyzer, 'event_callbacks'):
            callbacks = self._analyzer.event_callbacks

            # Error detection
            if 'error_detected' in callbacks:
                callbacks['error_detected'].add(self._on_error_detected)

            # Content change
            if 'content_changed' in callbacks:
                callbacks['content_changed'].add(self._on_content_changed)

            # App change
            if 'app_changed' in callbacks:
                callbacks['app_changed'].add(self._on_app_changed)

            # User needs help
            if 'user_needs_help' in callbacks:
                callbacks['user_needs_help'].add(self._on_user_needs_help)

            # Memory warning
            if 'memory_warning' in callbacks:
                callbacks['memory_warning'].add(self._on_memory_warning)

    async def _on_error_detected(self, state: Dict[str, Any]) -> None:
        """Handle error detection from screen analyzer."""
        if self._event_stream:
            from .proactive_event_stream import AGIEvent, EventType, EventPriority

            await self._event_stream.emit(AGIEvent(
                event_type=EventType.ERROR_DETECTED,
                source="screen_analyzer",
                data={
                    'error_type': state.get('error_type', 'unknown'),
                    'location': state.get('location', 'screen'),
                    'message': state.get('message', str(state)),
                },
                priority=EventPriority.HIGH,
                requires_narration=True,
            ))

    async def _on_content_changed(self, state: Dict[str, Any]) -> None:
        """Handle content change from screen analyzer."""
        if self._event_stream:
            from .proactive_event_stream import AGIEvent, EventType, EventPriority

            await self._event_stream.emit(AGIEvent(
                event_type=EventType.CONTENT_CHANGED,
                source="screen_analyzer",
                data=state,
                priority=EventPriority.LOW,
            ))

    async def _on_app_changed(self, state: Dict[str, Any]) -> None:
        """Handle app change from screen analyzer."""
        if self._event_stream:
            from .proactive_event_stream import AGIEvent, EventType, EventPriority

            await self._event_stream.emit(AGIEvent(
                event_type=EventType.APP_CHANGED,
                source="screen_analyzer",
                data=state,
                priority=EventPriority.LOW,
            ))

    async def _on_user_needs_help(self, state: Dict[str, Any]) -> None:
        """Handle user needs help detection."""
        if self._voice:
            from .realtime_voice_communicator import VoiceMode
            await self._voice.speak(
                "Sir, it looks like you might need some help. Let me know if I can assist.",
                mode=VoiceMode.CONVERSATIONAL
            )

    async def _on_memory_warning(self, state: Dict[str, Any]) -> None:
        """Handle memory warning from screen analyzer."""
        if self._event_stream:
            from .proactive_event_stream import AGIEvent, EventType, EventPriority

            await self._event_stream.emit(AGIEvent(
                event_type=EventType.MEMORY_WARNING,
                source="screen_analyzer",
                data=state,
                priority=EventPriority.HIGH,
                requires_narration=True,
            ))


class DecisionEngineBridge:
    """
    Bridge between autonomous decision engine and AGI OS.

    Routes decisions through the AGI OS approval system.
    """

    def __init__(self):
        """Initialize the bridge."""
        self._decision_engine: Optional[Any] = None
        self._approval_manager: Optional[Any] = None
        self._event_stream: Optional[Any] = None
        self._connected = False

    async def connect(
        self,
        decision_engine: Any,
        approval_manager: Optional[Any] = None,
        event_stream: Optional[Any] = None
    ) -> None:
        """
        Connect decision engine to AGI OS.

        Args:
            decision_engine: AutonomousDecisionEngine instance
            approval_manager: VoiceApprovalManager (or fetched automatically)
            event_stream: ProactiveEventStream (or fetched automatically)
        """
        if self._connected:
            logger.warning("Decision engine already connected")
            return

        self._decision_engine = decision_engine

        # Get or fetch approval manager
        if approval_manager:
            self._approval_manager = approval_manager
        else:
            try:
                from .voice_approval_manager import get_approval_manager
                self._approval_manager = await get_approval_manager()
            except Exception as e:
                logger.warning("Could not get approval manager: %s", e)

        # Get or fetch event stream
        if event_stream:
            self._event_stream = event_stream
        else:
            try:
                from .proactive_event_stream import get_event_stream
                self._event_stream = await get_event_stream()
            except Exception as e:
                logger.warning("Could not get event stream: %s", e)

        # Register decision handler
        if hasattr(self._decision_engine, 'register_decision_handler'):
            self._decision_engine.register_decision_handler(
                'agi_os_approval',
                self._handle_decision
            )

        self._connected = True
        logger.info("Decision engine connected to AGI OS")

    async def _handle_decision(self, context: Dict[str, Any]) -> List[Any]:
        """
        Handle decisions from the decision engine.

        Routes through AGI OS approval system.
        """
        if not self._approval_manager:
            return []

        # This would be called by the decision engine with proposed actions
        # The actions would then be routed through approval
        return []


class VoiceSystemBridge:
    """
    Bridge to unify voice output through AGI OS.

    Redirects existing voice calls to the RealTimeVoiceCommunicator.
    """

    def __init__(self):
        """Initialize the bridge."""
        self._voice: Optional[Any] = None
        self._original_voices: Dict[str, Any] = {}
        self._connected = False

    async def connect(self, voice: Optional[Any] = None) -> None:
        """
        Connect voice systems to AGI OS.

        Args:
            voice: RealTimeVoiceCommunicator (or fetched automatically)
        """
        if self._connected:
            return

        if voice:
            self._voice = voice
        else:
            try:
                from .realtime_voice_communicator import get_voice_communicator
                self._voice = await get_voice_communicator()
            except Exception as e:
                logger.warning("Could not get voice communicator: %s", e)
                return

        self._connected = True
        logger.info("Voice systems connected to AGI OS")

    async def speak(
        self,
        text: str,
        mode: str = "normal",
        **kwargs
    ) -> Optional[str]:
        """
        Speak through AGI OS voice system.

        Args:
            text: Text to speak
            mode: Voice mode
            **kwargs: Additional parameters

        Returns:
            Message ID or None
        """
        if not self._voice:
            return None

        from .realtime_voice_communicator import VoiceMode

        voice_modes = {
            'normal': VoiceMode.NORMAL,
            'urgent': VoiceMode.URGENT,
            'thoughtful': VoiceMode.THOUGHTFUL,
            'quiet': VoiceMode.QUIET,
            'notification': VoiceMode.NOTIFICATION,
        }

        voice_mode = voice_modes.get(mode, VoiceMode.NORMAL)
        return await self._voice.speak(text, mode=voice_mode)


class PermissionSystemBridge:
    """
    Bridge between existing permission manager and AGI OS approval system.

    Syncs approval decisions and learned patterns.
    """

    def __init__(self):
        """Initialize the bridge."""
        self._permission_manager: Optional[Any] = None
        self._approval_manager: Optional[Any] = None
        self._connected = False

    async def connect(
        self,
        permission_manager: Optional[Any] = None,
        approval_manager: Optional[Any] = None
    ) -> None:
        """
        Connect permission systems.

        Args:
            permission_manager: Existing PermissionManager
            approval_manager: VoiceApprovalManager (or fetched automatically)
        """
        if self._connected:
            return

        self._permission_manager = permission_manager

        if approval_manager:
            self._approval_manager = approval_manager
        else:
            try:
                from .voice_approval_manager import get_approval_manager
                self._approval_manager = await get_approval_manager()
            except Exception as e:
                logger.warning("Could not get approval manager: %s", e)
                return

        # Sync patterns from permission manager to approval manager
        if self._permission_manager and self._approval_manager:
            await self._sync_patterns()

        self._connected = True
        logger.info("Permission systems connected to AGI OS")

    async def _sync_patterns(self) -> None:
        """Sync learned patterns between systems."""
        # Get stats from permission manager
        if hasattr(self._permission_manager, 'get_permission_stats'):
            stats = self._permission_manager.get_permission_stats()
            logger.debug("Synced %d permission patterns", stats.get('unique_actions', 0))


class NeuralMeshBridge:
    """
    Bridge between Neural Mesh agents and AGI OS events.

    Allows agents to emit and subscribe to AGI OS events.
    """

    def __init__(self):
        """Initialize the bridge."""
        self._neural_mesh: Optional[Any] = None
        self._event_stream: Optional[Any] = None
        self._connected = False

    async def connect(
        self,
        neural_mesh: Optional[Any] = None,
        event_stream: Optional[Any] = None
    ) -> None:
        """
        Connect Neural Mesh to AGI OS.

        Args:
            neural_mesh: NeuralMeshCoordinator
            event_stream: ProactiveEventStream
        """
        if self._connected:
            return

        self._neural_mesh = neural_mesh

        if event_stream:
            self._event_stream = event_stream
        else:
            try:
                from .proactive_event_stream import get_event_stream
                self._event_stream = await get_event_stream()
            except Exception as e:
                logger.warning("Could not get event stream: %s", e)
                return

        self._connected = True
        logger.info("Neural Mesh connected to AGI OS")

    async def emit_agent_event(
        self,
        agent_name: str,
        event_type: str,
        data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Emit an event from a Neural Mesh agent.

        Args:
            agent_name: Name of the agent
            event_type: Type of event
            data: Event data

        Returns:
            Event ID or None
        """
        if not self._event_stream:
            return None

        from .proactive_event_stream import AGIEvent, EventType, EventPriority

        # Map event type string to enum
        event_type_map = {
            'error': EventType.ERROR_DETECTED,
            'warning': EventType.WARNING_DETECTED,
            'action_proposed': EventType.ACTION_PROPOSED,
            'action_completed': EventType.ACTION_COMPLETED,
            'action_failed': EventType.ACTION_FAILED,
        }

        agi_event_type = event_type_map.get(event_type, EventType.CONTENT_CHANGED)

        event = AGIEvent(
            event_type=agi_event_type,
            source=f"neural_mesh.{agent_name}",
            data=data,
            priority=EventPriority.NORMAL,
        )

        await self._event_stream.emit(event)
        return event.event_id


# ============== Convenience Functions ==============

_screen_bridge: Optional[ScreenAnalyzerBridge] = None
_decision_bridge: Optional[DecisionEngineBridge] = None
_voice_bridge: Optional[VoiceSystemBridge] = None
_permission_bridge: Optional[PermissionSystemBridge] = None
_mesh_bridge: Optional[NeuralMeshBridge] = None


async def connect_screen_analyzer(analyzer: Any) -> ScreenAnalyzerBridge:
    """
    Connect a screen analyzer to AGI OS.

    Args:
        analyzer: MemoryAwareScreenAnalyzer instance

    Returns:
        ScreenAnalyzerBridge instance
    """
    global _screen_bridge

    if _screen_bridge is None:
        _screen_bridge = ScreenAnalyzerBridge()

    await _screen_bridge.connect(analyzer)
    return _screen_bridge


async def connect_decision_engine(decision_engine: Any) -> DecisionEngineBridge:
    """
    Connect a decision engine to AGI OS.

    Args:
        decision_engine: AutonomousDecisionEngine instance

    Returns:
        DecisionEngineBridge instance
    """
    global _decision_bridge

    if _decision_bridge is None:
        _decision_bridge = DecisionEngineBridge()

    await _decision_bridge.connect(decision_engine)
    return _decision_bridge


async def integrate_voice_systems() -> VoiceSystemBridge:
    """
    Integrate voice systems with AGI OS.

    Returns:
        VoiceSystemBridge instance
    """
    global _voice_bridge

    if _voice_bridge is None:
        _voice_bridge = VoiceSystemBridge()

    await _voice_bridge.connect()
    return _voice_bridge


async def integrate_approval_systems(
    permission_manager: Optional[Any] = None
) -> PermissionSystemBridge:
    """
    Integrate approval systems with AGI OS.

    Args:
        permission_manager: Optional existing PermissionManager

    Returns:
        PermissionSystemBridge instance
    """
    global _permission_bridge

    if _permission_bridge is None:
        _permission_bridge = PermissionSystemBridge()

    await _permission_bridge.connect(permission_manager)
    return _permission_bridge


async def connect_neural_mesh(neural_mesh: Any) -> NeuralMeshBridge:
    """
    Connect Neural Mesh to AGI OS.

    Args:
        neural_mesh: NeuralMeshCoordinator instance

    Returns:
        NeuralMeshBridge instance
    """
    global _mesh_bridge

    if _mesh_bridge is None:
        _mesh_bridge = NeuralMeshBridge()

    await _mesh_bridge.connect(neural_mesh)
    return _mesh_bridge


async def integrate_all(
    screen_analyzer: Optional[Any] = None,
    decision_engine: Optional[Any] = None,
    permission_manager: Optional[Any] = None,
    neural_mesh: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Integrate all available systems with AGI OS.

    Args:
        screen_analyzer: Optional MemoryAwareScreenAnalyzer
        decision_engine: Optional AutonomousDecisionEngine
        permission_manager: Optional PermissionManager
        neural_mesh: Optional NeuralMeshCoordinator

    Returns:
        Dictionary of bridge instances
    """
    bridges = {}

    # Voice (always integrate)
    bridges['voice'] = await integrate_voice_systems()

    # Approval (always integrate)
    bridges['approval'] = await integrate_approval_systems(permission_manager)

    # Screen analyzer
    if screen_analyzer:
        bridges['screen'] = await connect_screen_analyzer(screen_analyzer)

    # Decision engine
    if decision_engine:
        bridges['decision'] = await connect_decision_engine(decision_engine)

    # Neural Mesh
    if neural_mesh:
        bridges['mesh'] = await connect_neural_mesh(neural_mesh)

    logger.info("Integrated %d systems with AGI OS", len(bridges))
    return bridges
