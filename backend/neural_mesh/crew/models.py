"""
JARVIS Neural Mesh - Crew Models

Core data models for the multi-agent collaboration system inspired by CrewAI.

Features:
- Dynamic agent definitions with capabilities
- Task models with dependencies and delegation
- Crew configuration with process types
- Memory schemas for collaboration context
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Type,
    Union,
)


# =============================================================================
# Enums
# =============================================================================

class ProcessType(str, Enum):
    """Types of crew processes for task orchestration."""
    SEQUENTIAL = "sequential"  # Tasks run in order
    HIERARCHICAL = "hierarchical"  # Manager delegates to workers
    CONSENSUS = "consensus"  # Agents vote on decisions
    DYNAMIC = "dynamic"  # Self-organizing based on capabilities
    PARALLEL = "parallel"  # Tasks run concurrently where possible
    PIPELINE = "pipeline"  # Data flows through agent chain


class TaskStatus(str, Enum):
    """Status of a task in the crew."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    DELEGATED = "delegated"
    WAITING_INPUT = "waiting_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentRole(str, Enum):
    """Standard roles agents can assume in a crew."""
    LEADER = "leader"  # Coordinates and makes decisions
    WORKER = "worker"  # Executes tasks
    SPECIALIST = "specialist"  # Domain expert
    REVIEWER = "reviewer"  # Quality control
    RESEARCHER = "researcher"  # Gathers information
    PLANNER = "planner"  # Designs solutions
    EXECUTOR = "executor"  # Implements solutions
    MONITOR = "monitor"  # Observes and reports
    COORDINATOR = "coordinator"  # Manages workflow


class DelegationStrategy(str, Enum):
    """Strategies for delegating tasks to agents."""
    CAPABILITY_MATCH = "capability_match"  # Match by capabilities
    LOAD_BALANCE = "load_balance"  # Distribute evenly
    PRIORITY_BASED = "priority_based"  # By task priority
    ROUND_ROBIN = "round_robin"  # Rotate through agents
    EXPERTISE_SCORE = "expertise_score"  # By expertise rating
    AVAILABILITY = "availability"  # First available
    HYBRID = "hybrid"  # Combination of strategies


class CollaborationType(str, Enum):
    """Types of collaboration between agents."""
    DELEGATION = "delegation"  # One agent assigns to another
    CONSULTATION = "consultation"  # Agent asks another for advice
    HANDOFF = "handoff"  # Transfer task completely
    PARALLEL_WORK = "parallel_work"  # Work on same task together
    REVIEW = "review"  # One reviews another's work
    ESCALATION = "escalation"  # Escalate to higher authority


class MemoryType(str, Enum):
    """Types of crew memory."""
    SHORT_TERM = "short_term"  # Current session context
    LONG_TERM = "long_term"  # Persistent knowledge
    ENTITY = "entity"  # Knowledge about entities (people, projects)
    PROCEDURAL = "procedural"  # How to do things
    EPISODIC = "episodic"  # Past experiences/events


# =============================================================================
# Capability Models
# =============================================================================

@dataclass
class Capability:
    """A specific capability an agent possesses."""
    name: str
    description: str = ""
    proficiency: float = 1.0  # 0.0 to 1.0
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(self, requirement: str, min_proficiency: float = 0.5) -> bool:
        """Check if this capability matches a requirement."""
        if self.proficiency < min_proficiency:
            return False
        requirement_lower = requirement.lower()
        return (
            requirement_lower in self.name.lower() or
            requirement_lower in self.description.lower() or
            any(requirement_lower in tag.lower() for tag in self.tags)
        )


@dataclass
class CapabilitySet:
    """A collection of capabilities with matching logic."""
    capabilities: Dict[str, Capability] = field(default_factory=dict)

    def add(self, capability: Capability) -> None:
        """Add a capability to the set."""
        self.capabilities[capability.name] = capability

    def remove(self, name: str) -> None:
        """Remove a capability by name."""
        self.capabilities.pop(name, None)

    def has(self, name: str, min_proficiency: float = 0.0) -> bool:
        """Check if set has a capability with minimum proficiency."""
        cap = self.capabilities.get(name)
        return cap is not None and cap.proficiency >= min_proficiency

    def match_requirements(
        self,
        requirements: List[str],
        min_proficiency: float = 0.5,
    ) -> float:
        """Calculate how well capabilities match requirements (0.0 to 1.0)."""
        if not requirements:
            return 1.0

        matched = 0
        for req in requirements:
            for cap in self.capabilities.values():
                if cap.matches(req, min_proficiency):
                    matched += cap.proficiency
                    break

        return matched / len(requirements)

    def to_list(self) -> List[str]:
        """Get list of capability names."""
        return list(self.capabilities.keys())


# =============================================================================
# Agent Models
# =============================================================================

@dataclass
class CrewAgentConfig:
    """Configuration for creating a crew agent."""
    name: str
    role: AgentRole = AgentRole.WORKER
    goal: str = ""
    backstory: str = ""
    capabilities: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 3
    can_delegate: bool = True
    can_be_delegated_to: bool = True
    allow_collaboration: bool = True
    memory_enabled: bool = True
    verbose: bool = False
    tools: List[str] = field(default_factory=list)
    llm_model: Optional[str] = None
    temperature: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrewAgent:
    """
    An agent in the crew system.

    Wraps existing JARVIS agents with crew collaboration capabilities.
    """
    id: str
    name: str
    role: AgentRole
    goal: str
    backstory: str
    capabilities: CapabilitySet
    config: CrewAgentConfig

    # Runtime state
    current_tasks: List[str] = field(default_factory=list)
    completed_tasks: int = 0
    failed_tasks: int = 0
    delegations_made: int = 0
    delegations_received: int = 0
    collaboration_count: int = 0
    last_active: Optional[datetime] = None
    status: str = "idle"

    # References
    underlying_agent: Optional[Any] = None  # Reference to actual JARVIS agent
    crew_id: Optional[str] = None

    @classmethod
    def from_config(cls, config: CrewAgentConfig) -> "CrewAgent":
        """Create agent from configuration."""
        capabilities = CapabilitySet()
        for cap_name in config.capabilities:
            capabilities.add(Capability(name=cap_name))

        return cls(
            id=str(uuid.uuid4()),
            name=config.name,
            role=config.role,
            goal=config.goal,
            backstory=config.backstory,
            capabilities=capabilities,
            config=config,
        )

    @property
    def is_available(self) -> bool:
        """Check if agent can take more tasks."""
        return len(self.current_tasks) < self.config.max_concurrent_tasks

    @property
    def workload(self) -> float:
        """Current workload as percentage (0.0 to 1.0)."""
        if self.config.max_concurrent_tasks == 0:
            return 1.0
        return len(self.current_tasks) / self.config.max_concurrent_tasks

    @property
    def success_rate(self) -> float:
        """Task success rate (0.0 to 1.0)."""
        total = self.completed_tasks + self.failed_tasks
        if total == 0:
            return 1.0
        return self.completed_tasks / total

    def can_handle(self, requirements: List[str], min_match: float = 0.5) -> bool:
        """Check if agent can handle task with given requirements."""
        if not self.is_available:
            return False
        return self.capabilities.match_requirements(requirements) >= min_match

    def expertise_score(self, requirements: List[str]) -> float:
        """Calculate expertise score for given requirements."""
        base_match = self.capabilities.match_requirements(requirements)
        experience_bonus = min(0.2, self.completed_tasks * 0.01)
        success_bonus = self.success_rate * 0.1
        return min(1.0, base_match + experience_bonus + success_bonus)


# =============================================================================
# Task Models
# =============================================================================

@dataclass
class TaskOutput:
    """Output from a completed task."""
    task_id: str
    agent_id: str
    result: Any
    success: bool
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    tokens_used: int = 0
    delegated: bool = False
    delegated_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TaskDependency:
    """A dependency between tasks."""
    task_id: str
    dependency_type: str = "blocking"  # blocking, optional, data
    required_output: Optional[str] = None  # Specific output field required


@dataclass
class CrewTask:
    """
    A task to be executed by the crew.

    Tasks can have dependencies, be delegated, and produce outputs
    that flow to dependent tasks.
    """
    id: str
    description: str
    expected_output: str

    # Assignment
    assigned_agent_id: Optional[str] = None
    assigned_at: Optional[datetime] = None

    # Requirements
    required_capabilities: List[str] = field(default_factory=list)
    priority: int = 5  # 1 (highest) to 10 (lowest)

    # Dependencies
    dependencies: List[TaskDependency] = field(default_factory=list)

    # Execution
    status: TaskStatus = TaskStatus.PENDING
    output: Optional[TaskOutput] = None
    attempts: int = 0
    max_attempts: int = 3

    # Delegation
    allow_delegation: bool = True
    delegation_chain: List[str] = field(default_factory=list)
    original_agent_id: Optional[str] = None

    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    input_data: Any = None
    tools_required: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout_seconds: float = 300.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        description: str,
        expected_output: str,
        required_capabilities: Optional[List[str]] = None,
        priority: int = 5,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "CrewTask":
        """Factory method to create a task."""
        return cls(
            id=str(uuid.uuid4()),
            description=description,
            expected_output=expected_output,
            required_capabilities=required_capabilities or [],
            priority=priority,
            context=context or {},
            **kwargs,
        )

    @property
    def is_ready(self) -> bool:
        """Check if task is ready to execute (all dependencies met)."""
        return self.status == TaskStatus.PENDING and not self.dependencies

    @property
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.attempts < self.max_attempts and self.status == TaskStatus.FAILED

    @property
    def execution_time_ms(self) -> Optional[float]:
        """Get execution time in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None


# =============================================================================
# Crew Configuration
# =============================================================================

@dataclass
class CrewConfig:
    """Configuration for a crew of agents."""
    name: str
    description: str = ""
    process_type: ProcessType = ProcessType.DYNAMIC
    delegation_strategy: DelegationStrategy = DelegationStrategy.HYBRID

    # Behavior
    verbose: bool = False
    max_parallel_tasks: int = 10
    task_timeout_seconds: float = 300.0
    max_delegation_depth: int = 3
    allow_self_delegation: bool = False

    # Memory
    memory_enabled: bool = True
    shared_memory: bool = True

    # Callbacks
    on_task_start: Optional[Callable] = None
    on_task_complete: Optional[Callable] = None
    on_delegation: Optional[Callable] = None
    on_error: Optional[Callable] = None

    # Integration
    neural_mesh_integration: bool = True
    langfuse_tracing: bool = True

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Collaboration Models
# =============================================================================

@dataclass
class CollaborationRequest:
    """A request for collaboration between agents."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    collaboration_type: CollaborationType = CollaborationType.CONSULTATION
    from_agent_id: str = ""
    to_agent_id: str = ""
    task_id: Optional[str] = None
    message: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5
    timeout_seconds: float = 60.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"


@dataclass
class CollaborationResponse:
    """Response to a collaboration request."""
    request_id: str
    from_agent_id: str
    accepted: bool
    response: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Memory Models
# =============================================================================

@dataclass
class MemoryEntry:
    """An entry in crew memory."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: MemoryType = MemoryType.SHORT_TERM
    content: Any = None
    summary: str = ""
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    relevance_score: float = 1.0
    access_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if memory entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at


@dataclass
class EntityMemory:
    """Memory about a specific entity (person, project, etc.)."""
    entity_id: str
    entity_type: str  # person, project, tool, etc.
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, str]] = field(default_factory=list)
    interactions: List[MemoryEntry] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Event Models
# =============================================================================

@dataclass
class CrewEvent:
    """An event in the crew system."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    crew_id: Optional[str] = None
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def task_started(cls, crew_id: str, agent_id: str, task_id: str) -> "CrewEvent":
        return cls(
            event_type="task_started",
            crew_id=crew_id,
            agent_id=agent_id,
            task_id=task_id,
        )

    @classmethod
    def task_completed(
        cls,
        crew_id: str,
        agent_id: str,
        task_id: str,
        success: bool,
    ) -> "CrewEvent":
        return cls(
            event_type="task_completed",
            crew_id=crew_id,
            agent_id=agent_id,
            task_id=task_id,
            data={"success": success},
        )

    @classmethod
    def delegation(
        cls,
        crew_id: str,
        from_agent: str,
        to_agent: str,
        task_id: str,
    ) -> "CrewEvent":
        return cls(
            event_type="delegation",
            crew_id=crew_id,
            task_id=task_id,
            data={"from_agent": from_agent, "to_agent": to_agent},
        )
