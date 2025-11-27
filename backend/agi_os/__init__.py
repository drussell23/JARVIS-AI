"""
JARVIS AGI OS - Autonomous General Intelligence Operating System

This module provides the core AGI OS functionality:
- Proactive, intelligent autonomous operation
- Real-time voice communication with Daniel TTS
- Voice-based approval workflows
- Integration with MAS + SAI + CAI + UAE
- Event-driven architecture for autonomous decisions

Components:
- AGIOSCoordinator: Central coordinator for all AGI OS components
- RealTimeVoiceCommunicator: Voice output with Daniel TTS
- VoiceApprovalManager: Voice-based user approval workflows
- ProactiveEventStream: Event-driven autonomous notifications
- IntelligentActionOrchestrator: Connects detection → decision → approval → execution

Usage:
    from agi_os import get_agi_os, start_agi_os

    # Get the AGI OS instance
    agi = await get_agi_os()

    # Start autonomous operation
    await agi.start()

    # AGI OS will now:
    # - Monitor your screen and detect issues
    # - Make intelligent decisions about what to do
    # - Ask for your approval via voice when needed
    # - Execute approved actions automatically
    # - Learn from your approvals to improve over time
"""

from .agi_os_coordinator import (
    AGIOSCoordinator,
    get_agi_os,
    start_agi_os,
    stop_agi_os,
)
from .realtime_voice_communicator import (
    RealTimeVoiceCommunicator,
    VoiceMode,
    VoicePriority,
    get_voice_communicator,
)
from .voice_approval_manager import (
    VoiceApprovalManager,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
    get_approval_manager,
)
from .proactive_event_stream import (
    ProactiveEventStream,
    AGIEvent,
    EventType,
    EventPriority,
    get_event_stream,
)
from .intelligent_action_orchestrator import (
    IntelligentActionOrchestrator,
    get_action_orchestrator,
)
from .jarvis_integration import (
    # Bridges
    ScreenAnalyzerBridge,
    DecisionEngineBridge,
    VoiceSystemBridge,
    PermissionSystemBridge,
    NeuralMeshBridge,
    # Vision types
    VisionEventType,
    ScreenAnalysisResult,
    ProactiveDetectionPattern,
    UnifiedVisionInterface,
    # Connection functions
    connect_screen_analyzer,
    connect_decision_engine,
    integrate_voice_systems,
    integrate_approval_systems,
    connect_neural_mesh,
    integrate_all,
    # New functions
    get_screen_bridge,
    get_unified_vision,
)
from .voice_authentication_narrator import (
    VoiceAuthNarrator,
    AuthenticationContext,
    AuthenticationResult,
    ConfidenceLevel,
    get_auth_narrator,
    create_auth_context,
    create_auth_result,
)
from .owner_identity_service import (
    OwnerIdentityService,
    OwnerProfile,
    IdentityContext,
    IdentitySource,
    IdentityConfidence,
    get_owner_identity,
    get_owner_name,
    verify_is_owner,
    create_identity_context,
)

__all__ = [
    # Main coordinator
    "AGIOSCoordinator",
    "get_agi_os",
    "start_agi_os",
    "stop_agi_os",
    # Voice communication
    "RealTimeVoiceCommunicator",
    "VoiceMode",
    "VoicePriority",
    "get_voice_communicator",
    # Approval system
    "VoiceApprovalManager",
    "ApprovalRequest",
    "ApprovalResponse",
    "ApprovalStatus",
    "get_approval_manager",
    # Event streaming
    "ProactiveEventStream",
    "AGIEvent",
    "EventType",
    "EventPriority",
    "get_event_stream",
    # Action orchestration
    "IntelligentActionOrchestrator",
    "get_action_orchestrator",
    # Integration bridges
    "ScreenAnalyzerBridge",
    "DecisionEngineBridge",
    "VoiceSystemBridge",
    "PermissionSystemBridge",
    "NeuralMeshBridge",
    "connect_screen_analyzer",
    "connect_decision_engine",
    "integrate_voice_systems",
    "integrate_approval_systems",
    "connect_neural_mesh",
    "integrate_all",
    "get_screen_bridge",
    "get_unified_vision",
    # Vision types and interface
    "VisionEventType",
    "ScreenAnalysisResult",
    "ProactiveDetectionPattern",
    "UnifiedVisionInterface",
    # Voice authentication narrator
    "VoiceAuthNarrator",
    "AuthenticationContext",
    "AuthenticationResult",
    "ConfidenceLevel",
    "get_auth_narrator",
    "create_auth_context",
    "create_auth_result",
    # Owner identity service
    "OwnerIdentityService",
    "OwnerProfile",
    "IdentityContext",
    "IdentitySource",
    "IdentityConfidence",
    "get_owner_identity",
    "get_owner_name",
    "verify_is_owner",
    "create_identity_context",
]
