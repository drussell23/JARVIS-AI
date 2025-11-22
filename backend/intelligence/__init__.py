"""
JARVIS Intelligence Layer
=========================

Advanced intelligence systems for JARVIS AI Agent with LangGraph
Chain-of-Thought reasoning capabilities.

Modules:
- unified_awareness_engine: Fusion of Context and Situational Awareness
- chain_of_thought: LangGraph-based multi-step reasoning
- uae_langgraph: Enhanced UAE with chain-of-thought
- intelligence_langgraph: Enhanced SAI, CAI, and Unified Orchestrator

Features:
- Multi-step autonomous reasoning with explicit thought chains
- Self-reflection and confidence calibration
- Cross-system intelligence fusion
- Continuous learning from outcomes
- Transparent decision audit trails
"""

# Original UAE components
from .unified_awareness_engine import (
    UnifiedAwarenessEngine,
    ContextIntelligenceLayer,
    SituationalAwarenessLayer,
    AwarenessIntegrationLayer,
    get_uae_engine,
    # Data models
    UnifiedDecision,
    ExecutionResult,
    ContextualData,
    SituationalData,
    ElementPriority,
    # Enums
    DecisionSource,
    ConfidenceSource
)

# Chain-of-Thought Reasoning
from .chain_of_thought import (
    ChainOfThoughtEngine,
    ChainOfThoughtMixin,
    ChainOfThoughtState,
    ReasoningStrategy,
    ThoughtType,
    ConfidenceLevel as CoTConfidenceLevel,
    ReasoningPhase as CoTReasoningPhase,
    Thought,
    ReasoningChain,
    Hypothesis,
    create_cot_engine,
    get_cot_engine,
    reason as cot_reason
)

# Enhanced UAE with LangGraph
from .uae_langgraph import (
    EnhancedUAE,
    UAEGraphState,
    UAEReasoningPhase,
    ReasonedDecision,
    create_enhanced_uae,
    get_enhanced_uae
)

# Enhanced SAI, CAI, and Unified Orchestrator
from .intelligence_langgraph import (
    # Enhanced SAI
    EnhancedSAI,
    SAIGraphState,
    SAIReasoningPhase,
    create_enhanced_sai,
    # Enhanced CAI
    EnhancedCAI,
    CAIGraphState,
    CAIReasoningPhase,
    create_enhanced_cai,
    # Unified Orchestrator
    UnifiedIntelligenceOrchestrator,
    create_unified_orchestrator,
    get_unified_orchestrator
)

__all__ = [
    # Original UAE
    'UnifiedAwarenessEngine',
    'ContextIntelligenceLayer',
    'SituationalAwarenessLayer',
    'AwarenessIntegrationLayer',
    'get_uae_engine',
    'UnifiedDecision',
    'ExecutionResult',
    'ContextualData',
    'SituationalData',
    'ElementPriority',
    'DecisionSource',
    'ConfidenceSource',

    # Chain-of-Thought
    'ChainOfThoughtEngine',
    'ChainOfThoughtMixin',
    'ChainOfThoughtState',
    'ReasoningStrategy',
    'ThoughtType',
    'CoTConfidenceLevel',
    'CoTReasoningPhase',
    'Thought',
    'ReasoningChain',
    'Hypothesis',
    'create_cot_engine',
    'get_cot_engine',
    'cot_reason',

    # Enhanced UAE
    'EnhancedUAE',
    'UAEGraphState',
    'UAEReasoningPhase',
    'ReasonedDecision',
    'create_enhanced_uae',
    'get_enhanced_uae',

    # Enhanced SAI
    'EnhancedSAI',
    'SAIGraphState',
    'SAIReasoningPhase',
    'create_enhanced_sai',

    # Enhanced CAI
    'EnhancedCAI',
    'CAIGraphState',
    'CAIReasoningPhase',
    'create_enhanced_cai',

    # Unified Orchestrator
    'UnifiedIntelligenceOrchestrator',
    'create_unified_orchestrator',
    'get_unified_orchestrator'
]

__version__ = '2.0.0'
