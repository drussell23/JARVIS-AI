"""
JARVIS Neural Mesh - Adapters Module

This module provides adapters to connect existing JARVIS agents
to the Neural Mesh system with minimal code changes.

Adapters allow legacy agents to participate in:
- Message passing via the Communication Bus
- Knowledge sharing via the Knowledge Graph
- Task execution via the Orchestrator
- Service discovery via the Registry
"""

from .legacy_agent_adapter import LegacyAgentAdapter

__all__ = ["LegacyAgentAdapter"]
