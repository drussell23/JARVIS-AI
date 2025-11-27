"""
JARVIS Neural Mesh - Crew Multi-Agent Collaboration

The main Crew class that orchestrates multi-agent collaboration.

Features:
- Dynamic agent management
- Multiple process types (sequential, hierarchical, dynamic, parallel)
- Task delegation and routing
- Shared memory and context
- Neural Mesh integration
- Event-driven architecture
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from .models import (
    CrewAgent,
    CrewAgentConfig,
    CrewTask,
    TaskOutput,
    TaskStatus,
    CrewConfig,
    ProcessType,
    DelegationStrategy,
    AgentRole,
    CrewEvent,
    CollaborationRequest,
    CollaborationResponse,
    CollaborationType,
)
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
from .memory import CrewMemory
from .delegation import TaskDelegationManager, DelegationDecision

logger = logging.getLogger(__name__)


# =============================================================================
# Crew Class
# =============================================================================

class Crew:
    """
    A crew of AI agents that collaborate to accomplish tasks.

    The Crew is the central orchestrator that manages agents, tasks,
    processes, memory, and communication.

    Example:
        crew = Crew(
            config=CrewConfig(
                name="Research Team",
                process_type=ProcessType.DYNAMIC,
            )
        )

        # Add agents
        crew.add_agent(CrewAgent.from_config(CrewAgentConfig(
            name="Researcher",
            role=AgentRole.SPECIALIST,
            capabilities=["research", "analysis"],
        )))

        # Execute tasks
        outputs = await crew.kickoff([
            CrewTask.create("Research topic X", "Summary report"),
            CrewTask.create("Analyze findings", "Analysis document"),
        ])
    """

    def __init__(
        self,
        config: Optional[CrewConfig] = None,
        agents: Optional[List[CrewAgent]] = None,
    ) -> None:
        """
        Initialize a crew.

        Args:
            config: Crew configuration
            agents: Initial list of agents
        """
        self.id = str(uuid.uuid4())
        self.config = config or CrewConfig(name="Default Crew")

        # Event system (initialize first since other components may emit events)
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._event_history: List[CrewEvent] = []

        # Task tracking
        self._tasks: Dict[str, CrewTask] = {}
        self._task_outputs: Dict[str, TaskOutput] = {}

        # State
        self._running = False
        self._started_at: Optional[datetime] = None

        # Neural Mesh integration
        self._mesh_coordinator = None
        self._mesh_adapters: Dict[str, Any] = {}

        # Agent management
        self.agents: Dict[str, CrewAgent] = {}

        # Process orchestration
        self._process: Optional[BaseProcess] = None

        # Memory system
        self.memory = CrewMemory(
            use_chromadb=True,
            short_term_ttl=3600,
            short_term_max=100,
        )

        # Delegation
        self.delegation = TaskDelegationManager(self)

        # Add agents after everything else is initialized
        if agents:
            for agent in agents:
                self.add_agent(agent)

        logger.info(f"Crew '{self.config.name}' created with ID {self.id}")

    # =========================================================================
    # Agent Management
    # =========================================================================

    def add_agent(
        self,
        agent: Union[CrewAgent, CrewAgentConfig],
    ) -> CrewAgent:
        """
        Add an agent to the crew.

        Args:
            agent: Agent instance or configuration

        Returns:
            The added agent
        """
        if isinstance(agent, CrewAgentConfig):
            agent = CrewAgent.from_config(agent)

        agent.crew_id = self.id
        self.agents[agent.id] = agent

        self.emit_event(CrewEvent(
            event_type="agent_added",
            crew_id=self.id,
            agent_id=agent.id,
            data={"agent_name": agent.name, "role": agent.role.value},
        ))

        logger.info(f"Agent '{agent.name}' added to crew '{self.config.name}'")
        return agent

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the crew."""
        if agent_id not in self.agents:
            return False

        agent = self.agents.pop(agent_id)

        self.emit_event(CrewEvent(
            event_type="agent_removed",
            crew_id=self.id,
            agent_id=agent_id,
            data={"agent_name": agent.name},
        ))

        logger.info(f"Agent '{agent.name}' removed from crew")
        return True

    def get_agent(self, agent_id: str) -> Optional[CrewAgent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)

    def get_agent_by_name(self, name: str) -> Optional[CrewAgent]:
        """Get an agent by name."""
        for agent in self.agents.values():
            if agent.name.lower() == name.lower():
                return agent
        return None

    def get_agents_by_role(self, role: AgentRole) -> List[CrewAgent]:
        """Get all agents with a specific role."""
        return [a for a in self.agents.values() if a.role == role]

    def get_available_agents(self) -> List[CrewAgent]:
        """Get all available agents."""
        return [a for a in self.agents.values() if a.is_available]

    # =========================================================================
    # Task Management
    # =========================================================================

    def create_task(
        self,
        description: str,
        expected_output: str,
        required_capabilities: Optional[List[str]] = None,
        priority: int = 5,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> CrewTask:
        """
        Create a new task.

        Args:
            description: What the task should accomplish
            expected_output: Description of expected output
            required_capabilities: Capabilities needed for this task
            priority: Task priority (1-10, lower is higher priority)
            context: Additional context for the task
            **kwargs: Additional task parameters

        Returns:
            Created task
        """
        task = CrewTask.create(
            description=description,
            expected_output=expected_output,
            required_capabilities=required_capabilities,
            priority=priority,
            context=context,
            **kwargs,
        )

        self._tasks[task.id] = task
        return task

    def get_task(self, task_id: str) -> Optional[CrewTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_pending_tasks(self) -> List[CrewTask]:
        """Get all pending tasks."""
        return [
            t for t in self._tasks.values()
            if t.status == TaskStatus.PENDING
        ]

    def get_task_output(self, task_id: str) -> Optional[TaskOutput]:
        """Get output for a completed task."""
        return self._task_outputs.get(task_id)

    # =========================================================================
    # Execution
    # =========================================================================

    async def kickoff(
        self,
        tasks: Optional[List[CrewTask]] = None,
        context: Optional[Dict[str, Any]] = None,
        process_type: Optional[ProcessType] = None,
    ) -> List[TaskOutput]:
        """
        Start the crew working on tasks.

        This is the main entry point for crew execution.

        Args:
            tasks: List of tasks to execute (uses pending tasks if not provided)
            context: Shared context for all tasks
            process_type: Override the default process type

        Returns:
            List of task outputs
        """
        if not self.agents:
            raise ValueError("Cannot kickoff crew without agents")

        # Use provided tasks or get pending ones
        task_list = tasks or self.get_pending_tasks()
        if not task_list:
            logger.warning("No tasks to execute")
            return []

        # Store tasks
        for task in task_list:
            if task.id not in self._tasks:
                self._tasks[task.id] = task

        self._running = True
        self._started_at = datetime.utcnow()

        # Create process
        pt = process_type or self.config.process_type
        self._process = create_process(pt, self)

        # Start delegation manager
        await self.delegation.start()

        self.emit_event(CrewEvent(
            event_type="crew_started",
            crew_id=self.id,
            data={
                "task_count": len(task_list),
                "agent_count": len(self.agents),
                "process_type": pt.value,
            },
        ))

        try:
            # Execute tasks
            outputs = await self._process.execute(task_list, context)

            # Store outputs
            for output in outputs:
                self._task_outputs[output.task_id] = output

                # Update task status
                task = self._tasks.get(output.task_id)
                if task:
                    task.output = output

                # Record in memory
                agent = self.agents.get(output.agent_id)
                if agent and task:
                    await self.memory.record_task_memory(task, output, agent)

            self.emit_event(CrewEvent(
                event_type="crew_completed",
                crew_id=self.id,
                data={
                    "total_tasks": len(task_list),
                    "successful": sum(1 for o in outputs if o.success),
                    "failed": sum(1 for o in outputs if not o.success),
                },
            ))

            return outputs

        except Exception as e:
            logger.error(f"Crew execution failed: {e}")
            self.emit_event(CrewEvent(
                event_type="crew_failed",
                crew_id=self.id,
                data={"error": str(e)},
            ))
            raise

        finally:
            self._running = False
            await self.delegation.stop()

    async def execute_agent_task(
        self,
        agent: CrewAgent,
        task: CrewTask,
    ) -> Any:
        """
        Execute a task with a specific agent.

        This is called by processes to execute individual tasks.
        Override this method to customize task execution.

        Args:
            agent: Agent to execute the task
            task: Task to execute

        Returns:
            Task result
        """
        self.emit_event(CrewEvent.task_started(self.id, agent.id, task.id))

        # Get relevant context from memory
        relevant_context = await self.memory.get_relevant_context(task, agent)
        task.context.update(relevant_context)

        try:
            # Check for Neural Mesh integration
            if self._mesh_coordinator and agent.underlying_agent:
                # Execute through Neural Mesh
                result = await self._execute_via_mesh(agent, task)
            else:
                # Execute locally
                result = await self._execute_local(agent, task)

            self.emit_event(CrewEvent.task_completed(
                self.id, agent.id, task.id, success=True
            ))

            return result

        except Exception as e:
            self.emit_event(CrewEvent.task_completed(
                self.id, agent.id, task.id, success=False
            ))
            raise

    async def _execute_local(
        self,
        agent: CrewAgent,
        task: CrewTask,
    ) -> Any:
        """Execute task locally without Neural Mesh."""
        # This is a basic implementation - override for custom behavior
        # In a full implementation, this would invoke the agent's LLM

        if agent.underlying_agent:
            # If agent wraps a real JARVIS agent, use it
            underlying = agent.underlying_agent
            if hasattr(underlying, 'execute_task'):
                return await underlying.execute_task(task.context)
            elif hasattr(underlying, 'process'):
                return await underlying.process({
                    "task": task.description,
                    "context": task.context,
                })

        # Placeholder for demonstration
        return {
            "status": "completed",
            "agent": agent.name,
            "task": task.description,
            "result": f"Task '{task.description}' completed by {agent.name}",
        }

    async def _execute_via_mesh(
        self,
        agent: CrewAgent,
        task: CrewTask,
    ) -> Any:
        """Execute task through Neural Mesh."""
        if not self._mesh_coordinator:
            return await self._execute_local(agent, task)

        # Get the adapter for this agent
        adapter = self._mesh_adapters.get(agent.id)
        if not adapter:
            return await self._execute_local(agent, task)

        # Execute through the adapter
        result = await adapter.execute_task({
            "action": task.description,
            "context": task.context,
            "expected_output": task.expected_output,
        })

        return result

    # =========================================================================
    # Neural Mesh Integration
    # =========================================================================

    async def connect_to_mesh(
        self,
        coordinator: Any,
    ) -> None:
        """
        Connect crew to Neural Mesh.

        Args:
            coordinator: NeuralMeshCoordinator instance
        """
        self._mesh_coordinator = coordinator

        # Create adapters for each agent's underlying JARVIS agent
        for agent in self.agents.values():
            if agent.underlying_agent:
                await self._create_mesh_adapter(agent)

        logger.info(f"Crew '{self.config.name}' connected to Neural Mesh")

    async def _create_mesh_adapter(self, agent: CrewAgent) -> None:
        """Create a Neural Mesh adapter for an agent."""
        if not self._mesh_coordinator:
            return

        # Determine adapter type based on agent role/capabilities
        try:
            from ..adapters import (
                IntelligenceAdapter,
                AutonomyAdapter,
                VoiceAdapter,
            )

            # Simple heuristic for adapter type
            caps = agent.capabilities.to_list()

            if any("voice" in c.lower() for c in caps):
                # Voice-related adapter
                pass  # Would create voice adapter
            elif any("reason" in c.lower() or "think" in c.lower() for c in caps):
                # Intelligence adapter
                pass  # Would create intelligence adapter
            else:
                # Default to autonomy adapter
                pass  # Would create autonomy adapter

            # Store adapter reference
            # self._mesh_adapters[agent.id] = adapter

        except ImportError:
            logger.warning("Neural Mesh adapters not available")

    def wrap_jarvis_agent(
        self,
        jarvis_agent: Any,
        config: CrewAgentConfig,
    ) -> CrewAgent:
        """
        Wrap an existing JARVIS agent for crew use.

        Args:
            jarvis_agent: Existing JARVIS agent instance
            config: Configuration for the crew agent wrapper

        Returns:
            CrewAgent that wraps the JARVIS agent
        """
        crew_agent = CrewAgent.from_config(config)
        crew_agent.underlying_agent = jarvis_agent
        self.add_agent(crew_agent)

        return crew_agent

    # =========================================================================
    # Events
    # =========================================================================

    def emit_event(self, event: CrewEvent) -> None:
        """Emit an event."""
        self._event_history.append(event)

        # Call handlers
        handlers = self._event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

        # Call config callbacks
        if event.event_type == "task_started" and self.config.on_task_start:
            self.config.on_task_start(event)
        elif event.event_type == "task_completed" and self.config.on_task_complete:
            self.config.on_task_complete(event)
        elif event.event_type == "delegation" and self.config.on_delegation:
            self.config.on_delegation(event)
        elif "error" in event.data and self.config.on_error:
            self.config.on_error(event)

        if self.config.verbose:
            logger.info(f"Event: {event.event_type} - {event.data}")

    def on(self, event_type: str, handler: Callable) -> None:
        """Register an event handler."""
        self._event_handlers[event_type].append(handler)

    def off(self, event_type: str, handler: Callable) -> None:
        """Remove an event handler."""
        handlers = self._event_handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    def get_event_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[CrewEvent]:
        """Get event history."""
        events = self._event_history[-limit:]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events

    # =========================================================================
    # Collaboration
    # =========================================================================

    async def request_collaboration(
        self,
        from_agent: CrewAgent,
        to_agent: CrewAgent,
        collaboration_type: CollaborationType,
        message: str,
        task_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> CollaborationResponse:
        """
        Request collaboration between agents.

        Args:
            from_agent: Agent requesting collaboration
            to_agent: Agent being requested
            collaboration_type: Type of collaboration
            message: Description of collaboration needed
            task_id: Related task ID
            context: Additional context

        Returns:
            Collaboration response
        """
        request = CollaborationRequest(
            collaboration_type=collaboration_type,
            from_agent_id=from_agent.id,
            to_agent_id=to_agent.id,
            task_id=task_id,
            message=message,
            context=context or {},
        )

        return await self.delegation.request_collaboration(request)

    # =========================================================================
    # State & Status
    # =========================================================================

    @property
    def is_running(self) -> bool:
        """Check if crew is currently running."""
        return self._running

    @property
    def agent_count(self) -> int:
        """Get total agent count."""
        return len(self.agents)

    @property
    def available_agent_count(self) -> int:
        """Get available agent count."""
        return len(self.get_available_agents())

    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive crew status."""
        memory_summary = await self.memory.summarize()
        delegation_stats = self.delegation.get_delegation_stats()

        return {
            "crew_id": self.id,
            "name": self.config.name,
            "running": self._running,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "process_type": self.config.process_type.value,
            "agents": {
                "total": len(self.agents),
                "available": self.available_agent_count,
                "by_role": {
                    role.value: len(self.get_agents_by_role(role))
                    for role in AgentRole
                },
            },
            "tasks": {
                "total": len(self._tasks),
                "pending": len(self.get_pending_tasks()),
                "completed": len(self._task_outputs),
            },
            "memory": memory_summary,
            "delegation": delegation_stats,
            "events": len(self._event_history),
        }

    def summary(self) -> str:
        """Get human-readable status summary."""
        lines = [
            f"=== Crew: {self.config.name} ===",
            f"ID: {self.id}",
            f"Status: {'Running' if self._running else 'Idle'}",
            f"Process: {self.config.process_type.value}",
            "",
            f"Agents: {len(self.agents)} ({self.available_agent_count} available)",
        ]

        for agent in self.agents.values():
            status_icon = "✓" if agent.is_available else "⏳"
            lines.append(
                f"  {status_icon} {agent.name} ({agent.role.value}) - "
                f"{agent.completed_tasks} completed"
            )

        lines.append("")
        lines.append(f"Tasks: {len(self._tasks)} total, {len(self._task_outputs)} completed")
        lines.append(f"Events: {len(self._event_history)} recorded")

        return "\n".join(lines)


# =============================================================================
# Crew Builder
# =============================================================================

class CrewBuilder:
    """
    Builder pattern for creating crews.

    Example:
        crew = (CrewBuilder()
            .name("Research Team")
            .process(ProcessType.DYNAMIC)
            .agent(name="Researcher", role=AgentRole.SPECIALIST, capabilities=["research"])
            .agent(name="Analyst", role=AgentRole.WORKER, capabilities=["analysis"])
            .build())
    """

    def __init__(self) -> None:
        self._config = CrewConfig(name="Crew")
        self._agents: List[CrewAgentConfig] = []

    def name(self, name: str) -> "CrewBuilder":
        """Set crew name."""
        self._config.name = name
        return self

    def description(self, description: str) -> "CrewBuilder":
        """Set crew description."""
        self._config.description = description
        return self

    def process(self, process_type: ProcessType) -> "CrewBuilder":
        """Set process type."""
        self._config.process_type = process_type
        return self

    def delegation_strategy(
        self,
        strategy: DelegationStrategy,
    ) -> "CrewBuilder":
        """Set delegation strategy."""
        self._config.delegation_strategy = strategy
        return self

    def verbose(self, verbose: bool = True) -> "CrewBuilder":
        """Enable verbose logging."""
        self._config.verbose = verbose
        return self

    def max_parallel_tasks(self, max_tasks: int) -> "CrewBuilder":
        """Set max parallel tasks."""
        self._config.max_parallel_tasks = max_tasks
        return self

    def memory_enabled(self, enabled: bool = True) -> "CrewBuilder":
        """Enable/disable memory."""
        self._config.memory_enabled = enabled
        return self

    def agent(
        self,
        name: str,
        role: AgentRole = AgentRole.WORKER,
        goal: str = "",
        backstory: str = "",
        capabilities: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "CrewBuilder":
        """Add an agent."""
        self._agents.append(CrewAgentConfig(
            name=name,
            role=role,
            goal=goal,
            backstory=backstory,
            capabilities=capabilities or [],
            **kwargs,
        ))
        return self

    def on_task_start(self, callback: Callable) -> "CrewBuilder":
        """Set task start callback."""
        self._config.on_task_start = callback
        return self

    def on_task_complete(self, callback: Callable) -> "CrewBuilder":
        """Set task complete callback."""
        self._config.on_task_complete = callback
        return self

    def on_delegation(self, callback: Callable) -> "CrewBuilder":
        """Set delegation callback."""
        self._config.on_delegation = callback
        return self

    def on_error(self, callback: Callable) -> "CrewBuilder":
        """Set error callback."""
        self._config.on_error = callback
        return self

    def build(self) -> Crew:
        """Build the crew."""
        agents = [CrewAgent.from_config(config) for config in self._agents]
        return Crew(config=self._config, agents=agents)


# =============================================================================
# Factory Functions
# =============================================================================

def create_crew(
    name: str,
    agents: List[CrewAgentConfig],
    process_type: ProcessType = ProcessType.DYNAMIC,
    **kwargs: Any,
) -> Crew:
    """
    Factory function to create a crew.

    Args:
        name: Crew name
        agents: List of agent configurations
        process_type: Process type for task orchestration
        **kwargs: Additional config options

    Returns:
        Configured Crew instance
    """
    config = CrewConfig(
        name=name,
        process_type=process_type,
        **kwargs,
    )

    agent_instances = [CrewAgent.from_config(cfg) for cfg in agents]

    return Crew(config=config, agents=agent_instances)


def create_research_crew(verbose: bool = False) -> Crew:
    """Create a pre-configured research crew."""
    return (CrewBuilder()
        .name("Research Crew")
        .description("A crew specialized in research and analysis")
        .process(ProcessType.SEQUENTIAL)
        .verbose(verbose)
        .agent(
            name="Lead Researcher",
            role=AgentRole.LEADER,
            goal="Coordinate research efforts",
            capabilities=["research", "coordination", "analysis"],
        )
        .agent(
            name="Data Analyst",
            role=AgentRole.SPECIALIST,
            goal="Analyze and interpret data",
            capabilities=["analysis", "statistics", "visualization"],
        )
        .agent(
            name="Content Writer",
            role=AgentRole.WORKER,
            goal="Write clear documentation",
            capabilities=["writing", "documentation", "summarization"],
        )
        .build())


def create_development_crew(verbose: bool = False) -> Crew:
    """Create a pre-configured development crew."""
    return (CrewBuilder()
        .name("Development Crew")
        .description("A crew specialized in software development")
        .process(ProcessType.DYNAMIC)
        .verbose(verbose)
        .agent(
            name="Tech Lead",
            role=AgentRole.LEADER,
            goal="Architect solutions and guide development",
            capabilities=["architecture", "code_review", "planning"],
        )
        .agent(
            name="Senior Developer",
            role=AgentRole.SPECIALIST,
            goal="Implement complex features",
            capabilities=["coding", "debugging", "optimization"],
        )
        .agent(
            name="QA Engineer",
            role=AgentRole.REVIEWER,
            goal="Ensure code quality",
            capabilities=["testing", "code_review", "documentation"],
        )
        .agent(
            name="DevOps Engineer",
            role=AgentRole.SPECIALIST,
            goal="Manage deployment and infrastructure",
            capabilities=["deployment", "monitoring", "automation"],
        )
        .build())
