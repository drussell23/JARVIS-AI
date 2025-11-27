"""
JARVIS Neural Mesh - Adapters Module

This module provides adapters to connect existing JARVIS agents
to the Neural Mesh system with minimal code changes.

Adapters allow legacy agents to participate in:
- Message passing via the Communication Bus
- Knowledge sharing via the Knowledge Graph
- Task execution via the Orchestrator
- Service discovery via the Registry

Available Adapters:
- LegacyAgentAdapter: Generic adapter for any existing agent
- IntelligenceEngineAdapter: For UAE, SAI, CAI, CoT, RGE engines
- AutonomyEngineAdapter: For AutonomousAgent, LangGraph, tools
- VoiceSystemAdapter: For voice agents and services
"""

from .legacy_agent_adapter import LegacyAgentAdapter, adapt_agent

from .intelligence_adapter import (
    IntelligenceEngineAdapter,
    IntelligenceEngineType,
    IntelligenceCapabilities,
    create_uae_adapter,
    create_sai_adapter,
    create_cot_adapter,
    create_rge_adapter,
    create_pie_adapter,
)

from .autonomy_adapter import (
    AutonomyEngineAdapter,
    AutonomyComponentType,
    AutonomyCapabilities,
    create_autonomous_agent_adapter,
    create_reasoning_adapter,
    create_tool_orchestrator_adapter,
    create_memory_adapter,
)

from .voice_adapter import (
    VoiceSystemAdapter,
    VoiceComponentType,
    VoiceCapabilities,
    create_voice_memory_adapter,
    create_speaker_verification_adapter,
    create_voice_unlock_adapter,
)

# Convenience aliases for simpler imports
IntelligenceAdapter = IntelligenceEngineAdapter
AutonomyAdapter = AutonomyEngineAdapter
VoiceAdapter = VoiceSystemAdapter

__all__ = [
    # Legacy Adapter
    "LegacyAgentAdapter",
    "adapt_agent",
    # Intelligence Adapters
    "IntelligenceEngineAdapter",
    "IntelligenceAdapter",  # Alias
    "IntelligenceEngineType",
    "IntelligenceCapabilities",
    "create_uae_adapter",
    "create_sai_adapter",
    "create_cot_adapter",
    "create_rge_adapter",
    "create_pie_adapter",
    # Autonomy Adapters
    "AutonomyEngineAdapter",
    "AutonomyAdapter",  # Alias
    "AutonomyComponentType",
    "AutonomyCapabilities",
    "create_autonomous_agent_adapter",
    "create_reasoning_adapter",
    "create_tool_orchestrator_adapter",
    "create_memory_adapter",
    # Voice Adapters
    "VoiceSystemAdapter",
    "VoiceAdapter",  # Alias
    "VoiceComponentType",
    "VoiceCapabilities",
    "create_voice_memory_adapter",
    "create_speaker_verification_adapter",
    "create_voice_unlock_adapter",
]
