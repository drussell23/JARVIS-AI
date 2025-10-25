"""
JARVIS Intelligence Layer
=========================

Advanced intelligence systems for JARVIS AI Agent.

Modules:
- unified_awareness_engine: Fusion of Context and Situational Awareness
"""

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

__all__ = [
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
    'ConfidenceSource'
]

__version__ = '1.0.0'
