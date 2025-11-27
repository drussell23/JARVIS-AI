"""
JARVIS Neural Mesh - Crew Multi-Agent Collaboration System

A CrewAI-inspired framework for orchestrating multi-agent collaboration.

This module provides:
- Crew: Main orchestrator for agent collaboration
- CrewAgent: Wrapper for agents with collaboration capabilities
- CrewTask: Task definitions with dependencies and routing
- Processes: Sequential, Hierarchical, Dynamic, Parallel, Consensus, Pipeline
- Memory: Short-term, Long-term, Entity, Episodic, Procedural
- Delegation: Capability-based, Load-balanced, Priority-based task assignment

Example:
    from neural_mesh.crew import Crew, CrewBuilder, CrewTask, AgentRole, ProcessType

    # Using the builder pattern
    crew = (CrewBuilder()
        .name("Analysis Team")
        .process(ProcessType.DYNAMIC)
        .agent(name="Researcher", role=AgentRole.SPECIALIST, capabilities=["research"])
        .agent(name="Analyst", role=AgentRole.WORKER, capabilities=["analysis"])
        .build())

    # Create and execute tasks
    tasks = [
        crew.create_task("Research topic X", "Summary report"),
        crew.create_task("Analyze findings", "Analysis document"),
    ]
    outputs = await crew.kickoff(tasks)

    # Check results
    for output in outputs:
        print(f"Task {output.task_id}: {'Success' if output.success else 'Failed'}")
"""

# Core crew functionality
from .crew import (
    Crew,
    CrewBuilder,
    create_crew,
    create_research_crew,
    create_development_crew,
)

# Models and types
from .models import (
    # Enums
    ProcessType,
    TaskStatus,
    AgentRole,
    DelegationStrategy,
    CollaborationType,
    MemoryType,
    # Agent models
    Capability,
    CapabilitySet,
    CrewAgentConfig,
    CrewAgent,
    # Task models
    TaskOutput,
    TaskDependency,
    CrewTask,
    # Configuration
    CrewConfig,
    # Collaboration
    CollaborationRequest,
    CollaborationResponse,
    # Memory models
    MemoryEntry,
    EntityMemory,
    # Events
    CrewEvent,
)

# Process types
from .processes import (
    BaseProcess,
    SequentialProcess,
    HierarchicalProcess,
    DynamicProcess,
    ParallelProcess,
    ConsensusProcess,
    PipelineProcess,
    create_process,
)

# Memory system
from .memory import (
    CrewMemory,
    ShortTermMemory,
    LongTermMemory,
    EntityMemoryStore,
    EpisodicMemory,
    ProceduralMemory,
    MemoryBackend,
    InMemoryBackend,
    ChromaDBBackend,
)

# Delegation system
from .delegation import (
    TaskDelegationManager,
    DelegationScore,
    DelegationDecision,
    BaseDelegationStrategy,
    CapabilityMatchStrategy,
    LoadBalanceStrategy,
    PriorityBasedStrategy,
    ExpertiseScoreStrategy,
    AvailabilityStrategy,
    HybridStrategy,
    RoundRobinStrategy,
    TaskRouter,
)

__all__ = [
    # Core
    "Crew",
    "CrewBuilder",
    "create_crew",
    "create_research_crew",
    "create_development_crew",
    # Enums
    "ProcessType",
    "TaskStatus",
    "AgentRole",
    "DelegationStrategy",
    "CollaborationType",
    "MemoryType",
    # Agent
    "Capability",
    "CapabilitySet",
    "CrewAgentConfig",
    "CrewAgent",
    # Task
    "TaskOutput",
    "TaskDependency",
    "CrewTask",
    # Config
    "CrewConfig",
    # Collaboration
    "CollaborationRequest",
    "CollaborationResponse",
    # Memory models
    "MemoryEntry",
    "EntityMemory",
    # Events
    "CrewEvent",
    # Processes
    "BaseProcess",
    "SequentialProcess",
    "HierarchicalProcess",
    "DynamicProcess",
    "ParallelProcess",
    "ConsensusProcess",
    "PipelineProcess",
    "create_process",
    # Memory system
    "CrewMemory",
    "ShortTermMemory",
    "LongTermMemory",
    "EntityMemoryStore",
    "EpisodicMemory",
    "ProceduralMemory",
    "MemoryBackend",
    "InMemoryBackend",
    "ChromaDBBackend",
    # Delegation
    "TaskDelegationManager",
    "DelegationScore",
    "DelegationDecision",
    "BaseDelegationStrategy",
    "CapabilityMatchStrategy",
    "LoadBalanceStrategy",
    "PriorityBasedStrategy",
    "ExpertiseScoreStrategy",
    "AvailabilityStrategy",
    "HybridStrategy",
    "RoundRobinStrategy",
    "TaskRouter",
]
