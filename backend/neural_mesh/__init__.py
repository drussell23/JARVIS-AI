"""
JARVIS Neural Mesh - Unified Multi-Agent Intelligence Framework

This module provides the core infrastructure for transforming 60+ isolated agents
into a cohesive, collaborative AI ecosystem with:

- Agent Communication Bus: Ultra-fast async message passing between agents
- Shared Knowledge Graph: Persistent, searchable collective memory
- Agent Registry: Service discovery and health monitoring
- Multi-Agent Orchestrator: Workflow coordination and task decomposition
- Base Agent: Unified interface for all agents

Architecture:
    TIER 0: Neural Mesh Intelligence Layer (this module)
    TIER 1: Master Intelligence (UAE, SAI, CAI)
    TIER 2: Core Domain Agents (28 agents)
    TIER 3: Specialized Sub-Agents (30+ agents)
"""

from .data_models import (
    MessageType,
    MessagePriority,
    AgentMessage,
    KnowledgeEntry,
    KnowledgeRelationship,
    KnowledgeType,
    AgentInfo,
    AgentStatus,
    WorkflowTask,
    WorkflowResult,
    ExecutionStrategy,
    HealthStatus,
)

from .communication.agent_communication_bus import AgentCommunicationBus
from .knowledge.shared_knowledge_graph import SharedKnowledgeGraph
from .registry.agent_registry import AgentRegistry
from .orchestration.multi_agent_orchestrator import MultiAgentOrchestrator
from .base.base_neural_mesh_agent import BaseNeuralMeshAgent
from .neural_mesh_coordinator import NeuralMeshCoordinator
from .config import NeuralMeshConfig

__version__ = "2.0.0"
__author__ = "JARVIS AI System"

__all__ = [
    # Data Models
    "MessageType",
    "MessagePriority",
    "AgentMessage",
    "KnowledgeEntry",
    "KnowledgeRelationship",
    "KnowledgeType",
    "AgentInfo",
    "AgentStatus",
    "WorkflowTask",
    "WorkflowResult",
    "ExecutionStrategy",
    "HealthStatus",
    # Core Components
    "AgentCommunicationBus",
    "SharedKnowledgeGraph",
    "AgentRegistry",
    "MultiAgentOrchestrator",
    "BaseNeuralMeshAgent",
    "NeuralMeshCoordinator",
    "NeuralMeshConfig",
]
