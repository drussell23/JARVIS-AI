"""
JARVIS Neural Mesh - Crew Processes

Process orchestrators that control how tasks flow through the crew.

Supports:
- Sequential: Tasks run in defined order
- Hierarchical: Manager delegates to workers
- Consensus: Agents vote on decisions
- Dynamic: Self-organizing based on capabilities
- Parallel: Concurrent execution where possible
- Pipeline: Data flows through agent chain
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
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
    TYPE_CHECKING,
)

from .models import (
    CrewAgent,
    CrewTask,
    TaskOutput,
    TaskStatus,
    ProcessType,
    DelegationStrategy,
    CrewEvent,
)

if TYPE_CHECKING:
    from .crew import Crew

logger = logging.getLogger(__name__)


# =============================================================================
# Base Process
# =============================================================================

class BaseProcess(ABC):
    """
    Abstract base class for crew processes.

    A process defines how tasks are orchestrated and executed
    across the crew's agents.
    """

    process_type: ProcessType

    def __init__(self, crew: "Crew") -> None:
        self.crew = crew
        self._running = False
        self._execution_history: List[Dict[str, Any]] = []

    @abstractmethod
    async def execute(
        self,
        tasks: List[CrewTask],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TaskOutput]:
        """Execute tasks according to the process type."""
        pass

    @abstractmethod
    async def assign_task(
        self,
        task: CrewTask,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[CrewAgent]:
        """Assign a task to the best available agent."""
        pass

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the execution history."""
        return self._execution_history.copy()


# =============================================================================
# Sequential Process
# =============================================================================

class SequentialProcess(BaseProcess):
    """
    Execute tasks in sequence, one after another.

    Each task must complete before the next one starts.
    Output from one task can be input to the next.
    """

    process_type = ProcessType.SEQUENTIAL

    async def execute(
        self,
        tasks: List[CrewTask],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TaskOutput]:
        """Execute tasks sequentially."""
        outputs: List[TaskOutput] = []
        current_context = context or {}

        for task in tasks:
            # Update task context with previous outputs
            task.context.update(current_context)
            if outputs:
                task.context["previous_output"] = outputs[-1]

            # Assign and execute
            agent = await self.assign_task(task)
            if not agent:
                logger.warning(f"No agent available for task {task.id}")
                outputs.append(TaskOutput(
                    task_id=task.id,
                    agent_id="",
                    result=None,
                    success=False,
                    error="No agent available",
                ))
                continue

            output = await self._execute_single_task(agent, task)
            outputs.append(output)

            # Update context with this output
            if output.success and output.result:
                current_context[f"task_{task.id}_output"] = output.result

            # Record history
            self._execution_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "task_id": task.id,
                "agent_id": agent.id,
                "success": output.success,
            })

        return outputs

    async def assign_task(
        self,
        task: CrewTask,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[CrewAgent]:
        """Assign task to best matching available agent."""
        best_agent = None
        best_score = 0.0

        for agent in self.crew.agents.values():
            if not agent.is_available:
                continue

            score = agent.expertise_score(task.required_capabilities)
            if score > best_score:
                best_score = score
                best_agent = agent

        return best_agent

    async def _execute_single_task(
        self,
        agent: CrewAgent,
        task: CrewTask,
    ) -> TaskOutput:
        """Execute a single task with an agent."""
        task.assigned_agent_id = agent.id
        task.assigned_at = datetime.utcnow()
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.utcnow()
        agent.current_tasks.append(task.id)
        agent.status = "working"

        try:
            # Execute via crew's task executor
            result = await self.crew.execute_agent_task(agent, task)

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            agent.completed_tasks += 1

            return TaskOutput(
                task_id=task.id,
                agent_id=agent.id,
                result=result,
                success=True,
                execution_time_ms=task.execution_time_ms or 0,
            )

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            agent.failed_tasks += 1

            return TaskOutput(
                task_id=task.id,
                agent_id=agent.id,
                result=None,
                success=False,
                error=str(e),
            )

        finally:
            agent.current_tasks.remove(task.id)
            agent.last_active = datetime.utcnow()
            agent.status = "idle"


# =============================================================================
# Hierarchical Process
# =============================================================================

class HierarchicalProcess(BaseProcess):
    """
    Manager agent delegates tasks to worker agents.

    A designated manager receives all tasks and decides
    which workers should handle them.
    """

    process_type = ProcessType.HIERARCHICAL

    def __init__(self, crew: "Crew", manager_id: Optional[str] = None) -> None:
        super().__init__(crew)
        self._manager_id = manager_id
        self._delegation_depth: Dict[str, int] = {}

    @property
    def manager(self) -> Optional[CrewAgent]:
        """Get the manager agent."""
        if self._manager_id:
            return self.crew.agents.get(self._manager_id)

        # Find agent with LEADER role
        for agent in self.crew.agents.values():
            from .models import AgentRole
            if agent.role == AgentRole.LEADER:
                return agent

        # Fallback to first agent
        if self.crew.agents:
            return next(iter(self.crew.agents.values()))
        return None

    async def execute(
        self,
        tasks: List[CrewTask],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TaskOutput]:
        """Execute tasks through hierarchical delegation."""
        outputs: List[TaskOutput] = []
        manager = self.manager

        if not manager:
            logger.error("No manager agent available for hierarchical process")
            return outputs

        for task in tasks:
            task.context.update(context or {})

            # Manager decides delegation
            delegate = await self._manager_decide_delegation(manager, task)

            if delegate and delegate.id != manager.id:
                # Delegate to worker
                self._delegation_depth[task.id] = 1
                output = await self._execute_delegated_task(delegate, task, manager.id)
            else:
                # Manager handles it
                output = await self._execute_single_task(manager, task)

            outputs.append(output)

            self._execution_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "task_id": task.id,
                "manager_id": manager.id,
                "delegated_to": delegate.id if delegate else None,
                "success": output.success,
            })

        return outputs

    async def assign_task(
        self,
        task: CrewTask,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[CrewAgent]:
        """Manager assigns task to best worker."""
        manager = self.manager
        if not manager:
            return None

        return await self._manager_decide_delegation(manager, task)

    async def _manager_decide_delegation(
        self,
        manager: CrewAgent,
        task: CrewTask,
    ) -> Optional[CrewAgent]:
        """Manager decides who should handle the task."""
        workers = [
            a for a in self.crew.agents.values()
            if a.id != manager.id and a.is_available and a.config.can_be_delegated_to
        ]

        if not workers:
            return manager  # Manager handles it

        # Find best worker based on expertise
        best_worker = None
        best_score = 0.0

        for worker in workers:
            score = worker.expertise_score(task.required_capabilities)
            if score > best_score:
                best_score = score
                best_worker = worker

        # Delegate if worker is significantly better
        manager_score = manager.expertise_score(task.required_capabilities)
        if best_worker and best_score > manager_score * 1.2:
            return best_worker

        return manager

    async def _execute_delegated_task(
        self,
        agent: CrewAgent,
        task: CrewTask,
        delegator_id: str,
    ) -> TaskOutput:
        """Execute a delegated task."""
        task.delegation_chain.append(delegator_id)
        agent.delegations_received += 1

        output = await self._execute_single_task(agent, task)
        output.delegated = True
        output.delegated_to = agent.id

        # Emit delegation event
        self.crew.emit_event(CrewEvent.delegation(
            self.crew.id,
            delegator_id,
            agent.id,
            task.id,
        ))

        return output

    async def _execute_single_task(
        self,
        agent: CrewAgent,
        task: CrewTask,
    ) -> TaskOutput:
        """Execute a single task."""
        task.assigned_agent_id = agent.id
        task.assigned_at = datetime.utcnow()
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.utcnow()
        agent.current_tasks.append(task.id)

        try:
            result = await self.crew.execute_agent_task(agent, task)
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            agent.completed_tasks += 1

            return TaskOutput(
                task_id=task.id,
                agent_id=agent.id,
                result=result,
                success=True,
                execution_time_ms=task.execution_time_ms or 0,
            )

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            agent.failed_tasks += 1

            return TaskOutput(
                task_id=task.id,
                agent_id=agent.id,
                result=None,
                success=False,
                error=str(e),
            )

        finally:
            agent.current_tasks.remove(task.id)
            agent.last_active = datetime.utcnow()


# =============================================================================
# Dynamic Process
# =============================================================================

class DynamicProcess(BaseProcess):
    """
    Self-organizing process that adapts based on capabilities.

    Agents dynamically form working groups, negotiate task
    assignments, and collaborate based on real-time conditions.
    """

    process_type = ProcessType.DYNAMIC

    def __init__(self, crew: "Crew") -> None:
        super().__init__(crew)
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._agent_assignments: Dict[str, Set[str]] = defaultdict(set)

    async def execute(
        self,
        tasks: List[CrewTask],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TaskOutput]:
        """Execute tasks dynamically with self-organization."""
        outputs: List[TaskOutput] = []
        pending_outputs: Dict[str, asyncio.Future] = {}

        # Sort tasks by priority and dependencies
        sorted_tasks = self._topological_sort(tasks)

        for task in sorted_tasks:
            task.context.update(context or {})
            future: asyncio.Future = asyncio.Future()
            pending_outputs[task.id] = future

            # Check dependencies
            deps_met = await self._wait_for_dependencies(task, pending_outputs)
            if not deps_met:
                future.set_result(TaskOutput(
                    task_id=task.id,
                    agent_id="",
                    result=None,
                    success=False,
                    error="Dependencies not met",
                ))
                continue

            # Dynamic assignment
            agent = await self.assign_task(task)
            if not agent:
                future.set_result(TaskOutput(
                    task_id=task.id,
                    agent_id="",
                    result=None,
                    success=False,
                    error="No suitable agent found",
                ))
                continue

            # Execute
            output = await self._execute_with_adaptation(agent, task)
            future.set_result(output)
            outputs.append(output)

            self._execution_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "task_id": task.id,
                "agent_id": agent.id,
                "success": output.success,
                "adaptation_used": output.metadata.get("adaptation_used", False),
            })

        return outputs

    async def assign_task(
        self,
        task: CrewTask,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[CrewAgent]:
        """Dynamically assign task based on multiple factors."""
        candidates = []

        for agent in self.crew.agents.values():
            if not agent.is_available:
                continue

            score = self._calculate_dynamic_score(agent, task)
            candidates.append((agent, score))

        if not candidates:
            return None

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Check if top candidates should collaborate
        if len(candidates) >= 2:
            top_two = candidates[:2]
            if top_two[0][1] - top_two[1][1] < 0.1:
                # Scores are close - consider collaboration
                if await self._should_collaborate(top_two[0][0], top_two[1][0], task):
                    task.metadata["collaboration_requested"] = True
                    task.metadata["collaborators"] = [
                        top_two[0][0].id,
                        top_two[1][0].id,
                    ]

        return candidates[0][0]

    def _calculate_dynamic_score(
        self,
        agent: CrewAgent,
        task: CrewTask,
    ) -> float:
        """Calculate dynamic assignment score."""
        # Base capability match
        capability_score = agent.expertise_score(task.required_capabilities)

        # Workload penalty
        workload_penalty = agent.workload * 0.3

        # Success rate bonus
        success_bonus = agent.success_rate * 0.2

        # Recency bonus (prefer recently active agents)
        recency_bonus = 0.0
        if agent.last_active:
            seconds_idle = (datetime.utcnow() - agent.last_active).total_seconds()
            if seconds_idle < 60:  # Active in last minute
                recency_bonus = 0.1

        # Priority boost for specialists
        from .models import AgentRole
        specialist_bonus = 0.15 if agent.role == AgentRole.SPECIALIST else 0.0

        return (
            capability_score
            - workload_penalty
            + success_bonus
            + recency_bonus
            + specialist_bonus
        )

    async def _should_collaborate(
        self,
        agent1: CrewAgent,
        agent2: CrewAgent,
        task: CrewTask,
    ) -> bool:
        """Determine if two agents should collaborate on a task."""
        # Check if agents have complementary capabilities
        caps1 = set(agent1.capabilities.to_list())
        caps2 = set(agent2.capabilities.to_list())

        overlap = len(caps1 & caps2)
        total = len(caps1 | caps2)

        if total == 0:
            return False

        # Collaborate if some overlap but also unique strengths
        overlap_ratio = overlap / total
        return 0.2 < overlap_ratio < 0.7

    def _topological_sort(self, tasks: List[CrewTask]) -> List[CrewTask]:
        """Sort tasks respecting dependencies."""
        task_map = {t.id: t for t in tasks}
        sorted_tasks: List[CrewTask] = []
        visited: Set[str] = set()
        temp_visited: Set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in temp_visited:
                raise ValueError(f"Circular dependency detected for task {task_id}")
            if task_id in visited:
                return

            temp_visited.add(task_id)
            task = task_map.get(task_id)
            if task:
                for dep in task.dependencies:
                    if dep.task_id in task_map:
                        visit(dep.task_id)

            temp_visited.remove(task_id)
            visited.add(task_id)
            if task:
                sorted_tasks.append(task)

        for task in tasks:
            if task.id not in visited:
                visit(task.id)

        return sorted_tasks

    async def _wait_for_dependencies(
        self,
        task: CrewTask,
        outputs: Dict[str, asyncio.Future],
    ) -> bool:
        """Wait for task dependencies to complete."""
        for dep in task.dependencies:
            if dep.task_id not in outputs:
                continue

            try:
                output = await asyncio.wait_for(
                    outputs[dep.task_id],
                    timeout=task.timeout_seconds,
                )
                if not output.success and dep.dependency_type == "blocking":
                    return False

                # Pass output to task context
                if dep.required_output:
                    task.context[f"dep_{dep.task_id}"] = output.result

            except asyncio.TimeoutError:
                logger.warning(f"Dependency {dep.task_id} timed out for task {task.id}")
                if dep.dependency_type == "blocking":
                    return False

        return True

    async def _execute_with_adaptation(
        self,
        agent: CrewAgent,
        task: CrewTask,
    ) -> TaskOutput:
        """Execute with dynamic adaptation."""
        task.assigned_agent_id = agent.id
        task.assigned_at = datetime.utcnow()
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.utcnow()
        agent.current_tasks.append(task.id)

        adaptation_used = False

        try:
            result = await self.crew.execute_agent_task(agent, task)
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            agent.completed_tasks += 1

            return TaskOutput(
                task_id=task.id,
                agent_id=agent.id,
                result=result,
                success=True,
                execution_time_ms=task.execution_time_ms or 0,
                metadata={"adaptation_used": adaptation_used},
            )

        except Exception as e:
            # Try adaptation - reassign to different agent
            if task.can_retry:
                task.attempts += 1
                alternate = await self._find_alternate_agent(agent, task)

                if alternate:
                    adaptation_used = True
                    agent.current_tasks.remove(task.id)
                    return await self._execute_with_adaptation(alternate, task)

            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            agent.failed_tasks += 1

            return TaskOutput(
                task_id=task.id,
                agent_id=agent.id,
                result=None,
                success=False,
                error=str(e),
                metadata={"adaptation_used": adaptation_used},
            )

        finally:
            if task.id in agent.current_tasks:
                agent.current_tasks.remove(task.id)
            agent.last_active = datetime.utcnow()

    async def _find_alternate_agent(
        self,
        exclude: CrewAgent,
        task: CrewTask,
    ) -> Optional[CrewAgent]:
        """Find an alternate agent for a failed task."""
        for agent in self.crew.agents.values():
            if agent.id == exclude.id:
                continue
            if not agent.is_available:
                continue
            if agent.can_handle(task.required_capabilities, min_match=0.3):
                return agent
        return None


# =============================================================================
# Parallel Process
# =============================================================================

class ParallelProcess(BaseProcess):
    """
    Execute independent tasks concurrently.

    Maximizes throughput by running multiple tasks in parallel
    while respecting dependencies.
    """

    process_type = ProcessType.PARALLEL

    def __init__(
        self,
        crew: "Crew",
        max_concurrent: int = 10,
    ) -> None:
        super().__init__(crew)
        self.max_concurrent = max_concurrent
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def execute(
        self,
        tasks: List[CrewTask],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TaskOutput]:
        """Execute tasks in parallel where possible."""
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        # Group by dependency level
        levels = self._group_by_dependency_level(tasks)
        outputs: List[TaskOutput] = []

        for level_tasks in levels:
            # Execute all tasks in this level concurrently
            level_outputs = await asyncio.gather(*[
                self._execute_with_semaphore(task, context)
                for task in level_tasks
            ])
            outputs.extend(level_outputs)

            # Update context with outputs for next level
            if context is None:
                context = {}
            for output in level_outputs:
                if output.success:
                    context[f"task_{output.task_id}"] = output.result

        return outputs

    async def assign_task(
        self,
        task: CrewTask,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[CrewAgent]:
        """Assign task to first available capable agent."""
        for agent in self.crew.agents.values():
            if agent.is_available and agent.can_handle(task.required_capabilities):
                return agent
        return None

    def _group_by_dependency_level(
        self,
        tasks: List[CrewTask],
    ) -> List[List[CrewTask]]:
        """Group tasks by dependency level for parallel execution."""
        task_map = {t.id: t for t in tasks}
        levels: Dict[str, int] = {}

        def get_level(task_id: str) -> int:
            if task_id in levels:
                return levels[task_id]

            task = task_map.get(task_id)
            if not task or not task.dependencies:
                levels[task_id] = 0
                return 0

            max_dep_level = 0
            for dep in task.dependencies:
                if dep.task_id in task_map:
                    dep_level = get_level(dep.task_id)
                    max_dep_level = max(max_dep_level, dep_level + 1)

            levels[task_id] = max_dep_level
            return max_dep_level

        for task in tasks:
            get_level(task.id)

        # Group by level
        level_groups: Dict[int, List[CrewTask]] = defaultdict(list)
        for task in tasks:
            level_groups[levels.get(task.id, 0)].append(task)

        return [level_groups[i] for i in sorted(level_groups.keys())]

    async def _execute_with_semaphore(
        self,
        task: CrewTask,
        context: Optional[Dict[str, Any]],
    ) -> TaskOutput:
        """Execute task with semaphore control."""
        async with self._semaphore:
            task.context.update(context or {})
            agent = await self.assign_task(task)

            if not agent:
                return TaskOutput(
                    task_id=task.id,
                    agent_id="",
                    result=None,
                    success=False,
                    error="No agent available",
                )

            return await self._execute_single_task(agent, task)

    async def _execute_single_task(
        self,
        agent: CrewAgent,
        task: CrewTask,
    ) -> TaskOutput:
        """Execute a single task."""
        task.assigned_agent_id = agent.id
        task.assigned_at = datetime.utcnow()
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.utcnow()
        agent.current_tasks.append(task.id)

        try:
            result = await self.crew.execute_agent_task(agent, task)
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            agent.completed_tasks += 1

            return TaskOutput(
                task_id=task.id,
                agent_id=agent.id,
                result=result,
                success=True,
                execution_time_ms=task.execution_time_ms or 0,
            )

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            agent.failed_tasks += 1

            return TaskOutput(
                task_id=task.id,
                agent_id=agent.id,
                result=None,
                success=False,
                error=str(e),
            )

        finally:
            agent.current_tasks.remove(task.id)
            agent.last_active = datetime.utcnow()


# =============================================================================
# Consensus Process
# =============================================================================

class ConsensusProcess(BaseProcess):
    """
    Agents vote on decisions and reach consensus.

    Useful for critical decisions where multiple perspectives
    are valuable.
    """

    process_type = ProcessType.CONSENSUS

    def __init__(
        self,
        crew: "Crew",
        min_voters: int = 3,
        consensus_threshold: float = 0.6,
    ) -> None:
        super().__init__(crew)
        self.min_voters = min_voters
        self.consensus_threshold = consensus_threshold

    async def execute(
        self,
        tasks: List[CrewTask],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TaskOutput]:
        """Execute tasks with consensus voting."""
        outputs: List[TaskOutput] = []

        for task in tasks:
            task.context.update(context or {})

            # Get votes from multiple agents
            votes = await self._collect_votes(task)

            if not votes:
                outputs.append(TaskOutput(
                    task_id=task.id,
                    agent_id="",
                    result=None,
                    success=False,
                    error="No votes collected",
                ))
                continue

            # Reach consensus
            consensus = self._reach_consensus(votes)
            outputs.append(consensus)

            self._execution_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "task_id": task.id,
                "vote_count": len(votes),
                "consensus_reached": consensus.success,
            })

        return outputs

    async def assign_task(
        self,
        task: CrewTask,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[CrewAgent]:
        """Select multiple agents for voting."""
        # In consensus, we don't assign to single agent
        return None

    async def _collect_votes(
        self,
        task: CrewTask,
    ) -> List[Tuple[CrewAgent, Any]]:
        """Collect votes from available agents."""
        voters = [
            a for a in self.crew.agents.values()
            if a.is_available and a.can_handle(task.required_capabilities, 0.3)
        ][:self.min_voters + 2]  # Get extra in case some fail

        if len(voters) < self.min_voters:
            return []

        votes: List[Tuple[CrewAgent, Any]] = []

        # Collect votes concurrently
        async def get_vote(agent: CrewAgent) -> Optional[Tuple[CrewAgent, Any]]:
            try:
                result = await self.crew.execute_agent_task(agent, task)
                return (agent, result)
            except Exception:
                return None

        results = await asyncio.gather(*[get_vote(a) for a in voters])

        for result in results:
            if result is not None:
                votes.append(result)

        return votes

    def _reach_consensus(
        self,
        votes: List[Tuple[CrewAgent, Any]],
    ) -> TaskOutput:
        """Determine consensus from votes."""
        if not votes:
            return TaskOutput(
                task_id="",
                agent_id="",
                result=None,
                success=False,
                error="No votes",
            )

        # Simple majority consensus
        # In a real implementation, you'd compare results more intelligently
        voter_ids = [v[0].id for v in votes]

        return TaskOutput(
            task_id="",
            agent_id=",".join(voter_ids),
            result=votes[0][1],  # Use first vote as result
            success=True,
            metadata={
                "vote_count": len(votes),
                "voters": voter_ids,
            },
        )


# =============================================================================
# Pipeline Process
# =============================================================================

class PipelineProcess(BaseProcess):
    """
    Data flows through a chain of agents.

    Each agent transforms the data and passes to the next,
    like a Unix pipeline.
    """

    process_type = ProcessType.PIPELINE

    def __init__(
        self,
        crew: "Crew",
        pipeline_agents: Optional[List[str]] = None,
    ) -> None:
        super().__init__(crew)
        self._pipeline_agents = pipeline_agents or []

    async def execute(
        self,
        tasks: List[CrewTask],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TaskOutput]:
        """Execute tasks through agent pipeline."""
        outputs: List[TaskOutput] = []

        for task in tasks:
            task.context.update(context or {})
            output = await self._run_pipeline(task)
            outputs.append(output)

            self._execution_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "task_id": task.id,
                "pipeline_length": len(self._pipeline_agents),
                "success": output.success,
            })

        return outputs

    async def assign_task(
        self,
        task: CrewTask,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[CrewAgent]:
        """Get first agent in pipeline."""
        if not self._pipeline_agents:
            return None
        return self.crew.agents.get(self._pipeline_agents[0])

    async def _run_pipeline(self, task: CrewTask) -> TaskOutput:
        """Run task through the agent pipeline."""
        pipeline = self._get_pipeline_agents()

        if not pipeline:
            return TaskOutput(
                task_id=task.id,
                agent_id="",
                result=None,
                success=False,
                error="No pipeline agents defined",
            )

        current_data = task.input_data
        final_agent_id = ""

        for agent in pipeline:
            # Create sub-task for this pipeline stage
            stage_task = CrewTask.create(
                description=f"Pipeline stage: {task.description}",
                expected_output=task.expected_output,
                required_capabilities=[],
                context={"pipeline_input": current_data, **task.context},
            )
            stage_task.input_data = current_data

            try:
                result = await self.crew.execute_agent_task(agent, stage_task)
                current_data = result
                final_agent_id = agent.id
                agent.completed_tasks += 1
            except Exception as e:
                return TaskOutput(
                    task_id=task.id,
                    agent_id=agent.id,
                    result=current_data,
                    success=False,
                    error=f"Pipeline failed at agent {agent.name}: {e}",
                )

        return TaskOutput(
            task_id=task.id,
            agent_id=final_agent_id,
            result=current_data,
            success=True,
            metadata={"pipeline_stages": len(pipeline)},
        )

    def _get_pipeline_agents(self) -> List[CrewAgent]:
        """Get ordered list of pipeline agents."""
        agents = []
        for agent_id in self._pipeline_agents:
            agent = self.crew.agents.get(agent_id)
            if agent:
                agents.append(agent)
        return agents


# =============================================================================
# Process Factory
# =============================================================================

def create_process(
    process_type: ProcessType,
    crew: "Crew",
    **kwargs: Any,
) -> BaseProcess:
    """Factory function to create a process by type."""
    process_classes = {
        ProcessType.SEQUENTIAL: SequentialProcess,
        ProcessType.HIERARCHICAL: HierarchicalProcess,
        ProcessType.DYNAMIC: DynamicProcess,
        ProcessType.PARALLEL: ParallelProcess,
        ProcessType.CONSENSUS: ConsensusProcess,
        ProcessType.PIPELINE: PipelineProcess,
    }

    process_class = process_classes.get(process_type)
    if not process_class:
        raise ValueError(f"Unknown process type: {process_type}")

    return process_class(crew, **kwargs)
