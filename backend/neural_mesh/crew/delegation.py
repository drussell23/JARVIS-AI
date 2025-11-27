"""
JARVIS Neural Mesh - Crew Delegation System

Dynamic task delegation and routing between agents.

Features:
- Capability-based assignment
- Load balancing
- Smart routing
- Delegation chains
- Automatic reassignment
- Collaboration negotiation
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from heapq import heappush, heappop
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
    TaskStatus,
    DelegationStrategy,
    CollaborationType,
    CollaborationRequest,
    CollaborationResponse,
    AgentRole,
    CrewEvent,
)

if TYPE_CHECKING:
    from .crew import Crew

logger = logging.getLogger(__name__)


# =============================================================================
# Delegation Scores
# =============================================================================

@dataclass
class DelegationScore:
    """Score for a potential task delegation."""
    agent_id: str
    agent_name: str
    overall_score: float
    capability_score: float
    availability_score: float
    workload_score: float
    performance_score: float
    priority_score: float
    breakdown: Dict[str, float] = field(default_factory=dict)

    def __lt__(self, other: "DelegationScore") -> bool:
        # Higher score = better (negate for min-heap behavior)
        return self.overall_score > other.overall_score


@dataclass
class DelegationDecision:
    """Result of a delegation decision."""
    task_id: str
    selected_agent_id: Optional[str]
    strategy_used: DelegationStrategy
    scores: List[DelegationScore]
    reason: str
    collaboration_suggested: bool = False
    suggested_collaborators: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Delegation Strategies
# =============================================================================

class BaseDelegationStrategy(ABC):
    """Base class for delegation strategies."""

    strategy_type: DelegationStrategy

    def __init__(self, crew: "Crew") -> None:
        self.crew = crew

    @abstractmethod
    def calculate_score(
        self,
        agent: CrewAgent,
        task: CrewTask,
    ) -> DelegationScore:
        """Calculate delegation score for an agent-task pair."""
        pass

    async def select_agent(
        self,
        task: CrewTask,
        exclude_agents: Optional[Set[str]] = None,
    ) -> DelegationDecision:
        """Select the best agent for a task."""
        exclude = exclude_agents or set()
        candidates = []

        for agent in self.crew.agents.values():
            if agent.id in exclude:
                continue
            if not agent.config.can_be_delegated_to:
                continue

            score = self.calculate_score(agent, task)
            candidates.append(score)

        if not candidates:
            return DelegationDecision(
                task_id=task.id,
                selected_agent_id=None,
                strategy_used=self.strategy_type,
                scores=[],
                reason="No eligible agents available",
            )

        # Sort by score (highest first)
        candidates.sort()
        best = candidates[0]

        return DelegationDecision(
            task_id=task.id,
            selected_agent_id=best.agent_id,
            strategy_used=self.strategy_type,
            scores=candidates,
            reason=f"Selected based on {self.strategy_type.value} with score {best.overall_score:.2f}",
        )


class CapabilityMatchStrategy(BaseDelegationStrategy):
    """Select agent based on capability match."""

    strategy_type = DelegationStrategy.CAPABILITY_MATCH

    def calculate_score(
        self,
        agent: CrewAgent,
        task: CrewTask,
    ) -> DelegationScore:
        capability_score = agent.capabilities.match_requirements(
            task.required_capabilities
        )

        return DelegationScore(
            agent_id=agent.id,
            agent_name=agent.name,
            overall_score=capability_score,
            capability_score=capability_score,
            availability_score=1.0 if agent.is_available else 0.0,
            workload_score=1.0 - agent.workload,
            performance_score=agent.success_rate,
            priority_score=1.0,
            breakdown={"capability_match": capability_score},
        )


class LoadBalanceStrategy(BaseDelegationStrategy):
    """Select agent with lowest workload."""

    strategy_type = DelegationStrategy.LOAD_BALANCE

    def calculate_score(
        self,
        agent: CrewAgent,
        task: CrewTask,
    ) -> DelegationScore:
        workload_score = 1.0 - agent.workload
        capability_score = agent.capabilities.match_requirements(
            task.required_capabilities
        )

        # Must have minimum capability
        if capability_score < 0.3:
            workload_score = 0.0

        return DelegationScore(
            agent_id=agent.id,
            agent_name=agent.name,
            overall_score=workload_score,
            capability_score=capability_score,
            availability_score=1.0 if agent.is_available else 0.0,
            workload_score=workload_score,
            performance_score=agent.success_rate,
            priority_score=1.0,
            breakdown={
                "workload": workload_score,
                "current_tasks": len(agent.current_tasks),
            },
        )


class PriorityBasedStrategy(BaseDelegationStrategy):
    """Select agent based on task priority and agent role."""

    strategy_type = DelegationStrategy.PRIORITY_BASED

    def calculate_score(
        self,
        agent: CrewAgent,
        task: CrewTask,
    ) -> DelegationScore:
        # High priority tasks go to specialists
        capability_score = agent.capabilities.match_requirements(
            task.required_capabilities
        )

        # Role bonus
        role_bonus = 0.0
        if task.priority <= 3:  # High priority
            if agent.role in [AgentRole.SPECIALIST, AgentRole.LEADER]:
                role_bonus = 0.3
        elif task.priority >= 8:  # Low priority
            if agent.role == AgentRole.WORKER:
                role_bonus = 0.2

        priority_score = capability_score + role_bonus

        return DelegationScore(
            agent_id=agent.id,
            agent_name=agent.name,
            overall_score=priority_score,
            capability_score=capability_score,
            availability_score=1.0 if agent.is_available else 0.0,
            workload_score=1.0 - agent.workload,
            performance_score=agent.success_rate,
            priority_score=priority_score,
            breakdown={
                "capability": capability_score,
                "role_bonus": role_bonus,
                "task_priority": task.priority,
            },
        )


class ExpertiseScoreStrategy(BaseDelegationStrategy):
    """Select agent with highest expertise for task requirements."""

    strategy_type = DelegationStrategy.EXPERTISE_SCORE

    def calculate_score(
        self,
        agent: CrewAgent,
        task: CrewTask,
    ) -> DelegationScore:
        expertise_score = agent.expertise_score(task.required_capabilities)

        return DelegationScore(
            agent_id=agent.id,
            agent_name=agent.name,
            overall_score=expertise_score,
            capability_score=agent.capabilities.match_requirements(
                task.required_capabilities
            ),
            availability_score=1.0 if agent.is_available else 0.0,
            workload_score=1.0 - agent.workload,
            performance_score=agent.success_rate,
            priority_score=1.0,
            breakdown={
                "expertise": expertise_score,
                "completed_tasks": agent.completed_tasks,
                "success_rate": agent.success_rate,
            },
        )


class AvailabilityStrategy(BaseDelegationStrategy):
    """Select first available agent."""

    strategy_type = DelegationStrategy.AVAILABILITY

    def calculate_score(
        self,
        agent: CrewAgent,
        task: CrewTask,
    ) -> DelegationScore:
        availability = 1.0 if agent.is_available else 0.0
        capability = agent.capabilities.match_requirements(
            task.required_capabilities
        )

        # Must meet minimum capability
        overall = availability if capability >= 0.2 else 0.0

        return DelegationScore(
            agent_id=agent.id,
            agent_name=agent.name,
            overall_score=overall,
            capability_score=capability,
            availability_score=availability,
            workload_score=1.0 - agent.workload,
            performance_score=agent.success_rate,
            priority_score=1.0,
            breakdown={
                "is_available": agent.is_available,
                "meets_capability": capability >= 0.2,
            },
        )


class HybridStrategy(BaseDelegationStrategy):
    """
    Combined strategy using multiple factors.

    This is the most sophisticated strategy that considers:
    - Capability match
    - Current workload
    - Historical performance
    - Agent role suitability
    - Task priority
    """

    strategy_type = DelegationStrategy.HYBRID

    def __init__(
        self,
        crew: "Crew",
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__(crew)
        self.weights = weights or {
            "capability": 0.35,
            "availability": 0.15,
            "workload": 0.15,
            "performance": 0.20,
            "priority": 0.15,
        }

    def calculate_score(
        self,
        agent: CrewAgent,
        task: CrewTask,
    ) -> DelegationScore:
        # Calculate individual scores
        capability_score = agent.capabilities.match_requirements(
            task.required_capabilities
        )
        availability_score = 1.0 if agent.is_available else 0.0
        workload_score = 1.0 - agent.workload
        performance_score = agent.success_rate

        # Priority score based on role alignment
        priority_score = self._calculate_priority_score(agent, task)

        # Weighted combination
        overall = (
            self.weights["capability"] * capability_score +
            self.weights["availability"] * availability_score +
            self.weights["workload"] * workload_score +
            self.weights["performance"] * performance_score +
            self.weights["priority"] * priority_score
        )

        # Penalty if not available
        if not agent.is_available:
            overall *= 0.3

        return DelegationScore(
            agent_id=agent.id,
            agent_name=agent.name,
            overall_score=overall,
            capability_score=capability_score,
            availability_score=availability_score,
            workload_score=workload_score,
            performance_score=performance_score,
            priority_score=priority_score,
            breakdown={
                "weighted_capability": self.weights["capability"] * capability_score,
                "weighted_availability": self.weights["availability"] * availability_score,
                "weighted_workload": self.weights["workload"] * workload_score,
                "weighted_performance": self.weights["performance"] * performance_score,
                "weighted_priority": self.weights["priority"] * priority_score,
            },
        )

    def _calculate_priority_score(
        self,
        agent: CrewAgent,
        task: CrewTask,
    ) -> float:
        """Calculate priority alignment score."""
        score = 0.5  # Base score

        # High priority tasks favor leaders and specialists
        if task.priority <= 3:
            if agent.role in [AgentRole.LEADER, AgentRole.SPECIALIST]:
                score += 0.3
            elif agent.role == AgentRole.EXECUTOR:
                score += 0.2

        # Medium priority - any capable agent
        elif 3 < task.priority <= 7:
            score += 0.2

        # Low priority - favor workers to free up specialists
        else:
            if agent.role == AgentRole.WORKER:
                score += 0.3

        return min(1.0, score)


class RoundRobinStrategy(BaseDelegationStrategy):
    """Rotate through agents in round-robin fashion."""

    strategy_type = DelegationStrategy.ROUND_ROBIN

    def __init__(self, crew: "Crew") -> None:
        super().__init__(crew)
        self._last_assigned: Dict[str, int] = {}
        self._assignment_order: List[str] = []

    def calculate_score(
        self,
        agent: CrewAgent,
        task: CrewTask,
    ) -> DelegationScore:
        # Check capability minimum
        capability = agent.capabilities.match_requirements(
            task.required_capabilities
        )
        if capability < 0.2:
            return DelegationScore(
                agent_id=agent.id,
                agent_name=agent.name,
                overall_score=0.0,
                capability_score=capability,
                availability_score=0.0,
                workload_score=0.0,
                performance_score=0.0,
                priority_score=0.0,
            )

        # Calculate position in rotation
        if not self._assignment_order:
            self._assignment_order = list(self.crew.agents.keys())

        try:
            position = self._assignment_order.index(agent.id)
        except ValueError:
            self._assignment_order.append(agent.id)
            position = len(self._assignment_order) - 1

        last_pos = self._last_assigned.get(task.required_capabilities[0] if task.required_capabilities else "default", -1)

        # Score based on rotation position
        if position == (last_pos + 1) % len(self._assignment_order):
            score = 1.0
        else:
            # Distance from next in rotation
            distance = (position - last_pos - 1) % len(self._assignment_order)
            score = 0.5 / (1 + distance)

        return DelegationScore(
            agent_id=agent.id,
            agent_name=agent.name,
            overall_score=score if agent.is_available else 0.0,
            capability_score=capability,
            availability_score=1.0 if agent.is_available else 0.0,
            workload_score=1.0 - agent.workload,
            performance_score=agent.success_rate,
            priority_score=1.0,
            breakdown={"rotation_position": position},
        )


# =============================================================================
# Task Delegation Manager
# =============================================================================

class TaskDelegationManager:
    """
    Manages task delegation across the crew.

    Coordinates agent selection, collaboration, and task routing.
    """

    def __init__(self, crew: "Crew") -> None:
        self.crew = crew
        self._strategies: Dict[DelegationStrategy, BaseDelegationStrategy] = {}
        self._delegation_history: List[DelegationDecision] = []
        self._pending_collaborations: Dict[str, CollaborationRequest] = {}
        self._delegation_queue: asyncio.Queue = asyncio.Queue()
        self._running = False

        # Initialize strategies
        self._init_strategies()

    def _init_strategies(self) -> None:
        """Initialize all delegation strategies."""
        self._strategies = {
            DelegationStrategy.CAPABILITY_MATCH: CapabilityMatchStrategy(self.crew),
            DelegationStrategy.LOAD_BALANCE: LoadBalanceStrategy(self.crew),
            DelegationStrategy.PRIORITY_BASED: PriorityBasedStrategy(self.crew),
            DelegationStrategy.EXPERTISE_SCORE: ExpertiseScoreStrategy(self.crew),
            DelegationStrategy.AVAILABILITY: AvailabilityStrategy(self.crew),
            DelegationStrategy.HYBRID: HybridStrategy(self.crew),
            DelegationStrategy.ROUND_ROBIN: RoundRobinStrategy(self.crew),
        }

    async def start(self) -> None:
        """Start the delegation manager."""
        self._running = True
        logger.info("TaskDelegationManager started")

    async def stop(self) -> None:
        """Stop the delegation manager."""
        self._running = False
        logger.info("TaskDelegationManager stopped")

    def get_strategy(
        self,
        strategy_type: Optional[DelegationStrategy] = None,
    ) -> BaseDelegationStrategy:
        """Get a delegation strategy."""
        st = strategy_type or self.crew.config.delegation_strategy
        return self._strategies.get(st, self._strategies[DelegationStrategy.HYBRID])

    async def delegate_task(
        self,
        task: CrewTask,
        from_agent: Optional[CrewAgent] = None,
        strategy: Optional[DelegationStrategy] = None,
        exclude_agents: Optional[Set[str]] = None,
    ) -> DelegationDecision:
        """
        Delegate a task to the best available agent.

        Args:
            task: Task to delegate
            from_agent: Agent delegating the task (if any)
            strategy: Delegation strategy to use
            exclude_agents: Agents to exclude from consideration

        Returns:
            Delegation decision with selected agent
        """
        exclude = exclude_agents or set()

        # Exclude the delegating agent if self-delegation not allowed
        if from_agent and not self.crew.config.allow_self_delegation:
            exclude.add(from_agent.id)

        # Check delegation depth
        if len(task.delegation_chain) >= self.crew.config.max_delegation_depth:
            return DelegationDecision(
                task_id=task.id,
                selected_agent_id=None,
                strategy_used=strategy or self.crew.config.delegation_strategy,
                scores=[],
                reason=f"Maximum delegation depth ({self.crew.config.max_delegation_depth}) reached",
            )

        # Get strategy and make decision
        strat = self.get_strategy(strategy)
        decision = await strat.select_agent(task, exclude)

        # Record delegation
        if from_agent and decision.selected_agent_id:
            task.delegation_chain.append(from_agent.id)
            from_agent.delegations_made += 1

            target_agent = self.crew.agents.get(decision.selected_agent_id)
            if target_agent:
                target_agent.delegations_received += 1

            # Emit event
            self.crew.emit_event(CrewEvent.delegation(
                self.crew.id,
                from_agent.id,
                decision.selected_agent_id,
                task.id,
            ))

        self._delegation_history.append(decision)
        return decision

    async def find_collaborators(
        self,
        task: CrewTask,
        primary_agent: CrewAgent,
        max_collaborators: int = 2,
    ) -> List[CrewAgent]:
        """
        Find agents who could collaborate on a task.

        Looks for agents with complementary capabilities.
        """
        collaborators = []
        primary_caps = set(primary_agent.capabilities.to_list())

        for agent in self.crew.agents.values():
            if agent.id == primary_agent.id:
                continue
            if not agent.is_available:
                continue
            if not agent.config.allow_collaboration:
                continue

            # Check for complementary capabilities
            agent_caps = set(agent.capabilities.to_list())
            overlap = primary_caps & agent_caps
            unique = agent_caps - primary_caps

            # Good collaborator has some overlap but also unique capabilities
            if overlap and unique:
                overlap_ratio = len(overlap) / len(primary_caps | agent_caps)
                if 0.1 < overlap_ratio < 0.6:
                    collaborators.append((agent, len(unique)))

        # Sort by unique capability count
        collaborators.sort(key=lambda x: x[1], reverse=True)

        return [c[0] for c in collaborators[:max_collaborators]]

    async def request_collaboration(
        self,
        request: CollaborationRequest,
    ) -> CollaborationResponse:
        """
        Request collaboration from another agent.

        Args:
            request: Collaboration request

        Returns:
            Response from the target agent
        """
        target_agent = self.crew.agents.get(request.to_agent_id)
        if not target_agent:
            return CollaborationResponse(
                request_id=request.id,
                from_agent_id=request.to_agent_id,
                accepted=False,
                error="Agent not found",
            )

        if not target_agent.config.allow_collaboration:
            return CollaborationResponse(
                request_id=request.id,
                from_agent_id=request.to_agent_id,
                accepted=False,
                error="Agent does not accept collaborations",
            )

        if not target_agent.is_available:
            return CollaborationResponse(
                request_id=request.id,
                from_agent_id=request.to_agent_id,
                accepted=False,
                error="Agent is not available",
            )

        # Store pending collaboration
        self._pending_collaborations[request.id] = request
        request.status = "accepted"
        target_agent.collaboration_count += 1

        return CollaborationResponse(
            request_id=request.id,
            from_agent_id=request.to_agent_id,
            accepted=True,
        )

    async def reassign_task(
        self,
        task: CrewTask,
        from_agent: CrewAgent,
        reason: str = "reassignment",
    ) -> DelegationDecision:
        """
        Reassign a task to a different agent.

        Used when an agent fails or becomes unavailable.
        """
        logger.info(f"Reassigning task {task.id} from {from_agent.name}: {reason}")

        # Remove from current agent
        if task.id in from_agent.current_tasks:
            from_agent.current_tasks.remove(task.id)

        # Reset task status
        task.status = TaskStatus.PENDING
        task.attempts += 1

        # Delegate to new agent, excluding the failed one
        return await self.delegate_task(
            task,
            from_agent=None,  # Not a delegation chain continuation
            exclude_agents={from_agent.id},
        )

    async def get_agent_workload_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get workload summary for all agents."""
        summary = {}

        for agent in self.crew.agents.values():
            summary[agent.id] = {
                "name": agent.name,
                "role": agent.role.value,
                "current_tasks": len(agent.current_tasks),
                "max_tasks": agent.config.max_concurrent_tasks,
                "workload_pct": agent.workload * 100,
                "is_available": agent.is_available,
                "completed_tasks": agent.completed_tasks,
                "failed_tasks": agent.failed_tasks,
                "success_rate": agent.success_rate * 100,
                "delegations_made": agent.delegations_made,
                "delegations_received": agent.delegations_received,
            }

        return summary

    def get_delegation_history(
        self,
        limit: int = 100,
        task_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[DelegationDecision]:
        """Get delegation history."""
        history = self._delegation_history[-limit:]

        if task_id:
            history = [d for d in history if d.task_id == task_id]
        if agent_id:
            history = [d for d in history if d.selected_agent_id == agent_id]

        return history

    def get_delegation_stats(self) -> Dict[str, Any]:
        """Get delegation statistics."""
        if not self._delegation_history:
            return {
                "total_delegations": 0,
                "successful_delegations": 0,
                "failed_delegations": 0,
                "by_strategy": {},
            }

        by_strategy: Dict[str, int] = defaultdict(int)
        successful = 0
        failed = 0

        for decision in self._delegation_history:
            by_strategy[decision.strategy_used.value] += 1
            if decision.selected_agent_id:
                successful += 1
            else:
                failed += 1

        return {
            "total_delegations": len(self._delegation_history),
            "successful_delegations": successful,
            "failed_delegations": failed,
            "success_rate": successful / len(self._delegation_history) * 100,
            "by_strategy": dict(by_strategy),
        }


# =============================================================================
# Task Router
# =============================================================================

class TaskRouter:
    """
    Routes tasks to appropriate crews or sub-crews.

    Used for organizing work across multiple crews.
    """

    def __init__(self) -> None:
        self._routes: Dict[str, Callable] = {}
        self._default_route: Optional[Callable] = None

    def register_route(
        self,
        pattern: str,
        handler: Callable,
    ) -> None:
        """Register a route pattern."""
        self._routes[pattern] = handler

    def set_default(self, handler: Callable) -> None:
        """Set default route handler."""
        self._default_route = handler

    async def route(
        self,
        task: CrewTask,
    ) -> Optional[Callable]:
        """Route a task to appropriate handler."""
        # Check specific routes first
        for pattern, handler in self._routes.items():
            if self._matches_pattern(task, pattern):
                return handler

        return self._default_route

    def _matches_pattern(self, task: CrewTask, pattern: str) -> bool:
        """Check if task matches a route pattern."""
        # Simple capability-based matching
        pattern_lower = pattern.lower()

        # Check task description
        if pattern_lower in task.description.lower():
            return True

        # Check required capabilities
        for cap in task.required_capabilities:
            if pattern_lower in cap.lower():
                return True

        # Check task type in metadata
        task_type = task.metadata.get("task_type", "")
        if pattern_lower in task_type.lower():
            return True

        return False
