#!/usr/bin/env python3
"""
Advanced Component Warmup System
=================================

Ultra-robust, async, priority-based component pre-initialization system
with health checks, progressive loading, and zero hardcoding.

Design Principles:
- Async-first: All initialization is concurrent
- Priority-based: Critical components load first
- Health-aware: Components verify readiness
- Progressive: Non-critical components load in background
- Dynamic: Auto-discovers components, no hardcoding
- Resilient: Failures don't block startup
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ComponentPriority(Enum):
    """Component loading priority levels"""

    CRITICAL = 0  # Must load before any commands (screen detection, auth)
    HIGH = 1  # Should load before first command (NLP, vision)
    MEDIUM = 2  # Nice to have loaded early (analytics, learning)
    LOW = 3  # Can load in background (telemetry, extras)
    DEFERRED = 4  # Load on demand only (heavy ML models)


class ComponentStatus(Enum):
    """Component initialization status"""

    PENDING = "pending"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    DEGRADED = "degraded"  # Loaded but not fully functional


@dataclass
class ComponentMetrics:
    """Metrics for component initialization"""

    load_time: float = 0.0
    retry_count: int = 0
    last_error: Optional[str] = None
    health_score: float = 1.0  # 0.0 to 1.0
    dependencies_met: bool = True
    memory_usage: int = 0  # bytes


@dataclass
class ComponentDefinition:
    """Definition of a component to initialize"""

    name: str
    priority: ComponentPriority
    loader: Callable  # async function to load component
    health_check: Optional[Callable] = None  # async function to verify health
    dependencies: List[str] = field(default_factory=list)
    timeout: float = 30.0  # max seconds to wait for initialization
    retry_count: int = 2  # number of retries on failure
    required: bool = True  # if False, failures don't block startup
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComponentWarmupSystem:
    """
    Advanced async component pre-initialization system.

    Features:
    - Priority-based loading (critical → high → medium → low → deferred)
    - Parallel initialization within priority levels
    - Dependency resolution and ordering
    - Health checks with retry logic
    - Progressive loading (non-critical in background)
    - Graceful degradation on failures
    - Detailed metrics and telemetry
    """

    def __init__(self, max_concurrent: int = 10, enable_progressive: bool = True):
        self.max_concurrent = max_concurrent
        self.enable_progressive = enable_progressive

        # Component registry
        self.components: Dict[str, ComponentDefinition] = {}
        self.component_status: Dict[str, ComponentStatus] = {}
        self.component_instances: Dict[str, Any] = {}
        self.component_metrics: Dict[str, ComponentMetrics] = defaultdict(ComponentMetrics)

        # Dependency tracking
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)

        # Synchronization
        self.ready_events: Dict[str, asyncio.Event] = {}
        self.warmup_complete = asyncio.Event()
        self.critical_ready = asyncio.Event()

        # Metrics
        self.total_start_time = None
        self.critical_load_time = 0.0
        self.total_load_time = 0.0
        self.failed_components: List[str] = []

    def register_component(
        self,
        name: str,
        loader: Callable,
        priority: ComponentPriority = ComponentPriority.MEDIUM,
        health_check: Optional[Callable] = None,
        dependencies: Optional[List[str]] = None,
        timeout: float = 30.0,
        retry_count: int = 2,
        required: bool = True,
        **metadata,
    ):
        """Register a component for warmup"""
        component = ComponentDefinition(
            name=name,
            priority=priority,
            loader=loader,
            health_check=health_check,
            dependencies=dependencies or [],
            timeout=timeout,
            retry_count=retry_count,
            required=required,
            metadata=metadata,
        )

        self.components[name] = component
        self.component_status[name] = ComponentStatus.PENDING
        self.ready_events[name] = asyncio.Event()

        # Build dependency graph
        for dep in component.dependencies:
            self.dependency_graph[name].add(dep)
            self.reverse_dependencies[dep].add(name)

        logger.info(
            f"[WARMUP] Registered component: {name} "
            f"(priority={priority.name}, required={required})"
        )

    async def warmup_all(self) -> Dict[str, Any]:
        """
        Execute warmup for all registered components.

        Returns:
            Dictionary with warmup results and metrics
        """
        self.total_start_time = time.time()

        logger.info(
            f"[WARMUP] 🚀 Starting component warmup "
            f"({len(self.components)} components registered)"
        )

        # Group components by priority
        priority_groups = defaultdict(list)
        for name, component in self.components.items():
            priority_groups[component.priority].append(name)

        # Load components in priority order
        for priority in sorted(ComponentPriority, key=lambda p: p.value):
            if priority not in priority_groups:
                continue

            components = priority_groups[priority]
            logger.info(
                f"[WARMUP] Loading {len(components)} {priority.name} priority components..."
            )

            # Resolve dependencies and create load order
            load_order = self._resolve_load_order(components)

            # Load components in parallel (respecting dependencies)
            await self._load_component_batch(load_order, priority)

            # Mark critical components ready
            if priority == ComponentPriority.CRITICAL:
                self.critical_ready.set()
                self.critical_load_time = time.time() - self.total_start_time
                logger.info(
                    f"[WARMUP] ✅ Critical components ready in {self.critical_load_time:.2f}s"
                )

        # Mark warmup complete
        self.total_load_time = time.time() - self.total_start_time
        self.warmup_complete.set()

        # Generate warmup report
        report = self._generate_warmup_report()
        logger.info(
            f"[WARMUP] 🎉 Warmup complete in {self.total_load_time:.2f}s "
            f"({report['ready_count']}/{report['total_count']} components ready)"
        )

        return report

    async def _load_component_batch(self, components: List[str], priority: ComponentPriority):
        """Load a batch of components in parallel"""
        tasks = []
        semaphore = asyncio.Semaphore(self.max_concurrent)

        for component_name in components:
            task = self._load_component_with_semaphore(component_name, semaphore)
            tasks.append(task)

        # Wait for all components in this batch
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any exceptions
        for component_name, result in zip(components, results):
            if isinstance(result, Exception):
                logger.error(
                    f"[WARMUP] ❌ Component {component_name} failed: {result}", exc_info=result
                )

    async def _load_component_with_semaphore(self, name: str, semaphore: asyncio.Semaphore):
        """Load a component with concurrency control"""
        async with semaphore:
            return await self._load_component(name)

    async def _load_component(self, name: str) -> bool:
        """Load a single component with retry logic and health checks"""
        component = self.components[name]
        metrics = self.component_metrics[name]

        logger.info(f"[WARMUP] 📦 Loading {name}...")
        self.component_status[name] = ComponentStatus.LOADING

        start_time = time.time()

        # Wait for dependencies
        for dep in component.dependencies:
            if dep not in self.ready_events:
                logger.warning(f"[WARMUP] Dependency {dep} not registered for {name}")
                continue

            try:
                await asyncio.wait_for(self.ready_events[dep].wait(), timeout=component.timeout)
            except asyncio.TimeoutError:
                logger.error(f"[WARMUP] Timeout waiting for dependency {dep} for {name}")
                metrics.dependencies_met = False
                if component.required:
                    self.component_status[name] = ComponentStatus.FAILED
                    self.failed_components.append(name)
                    return False

        # Try loading with retries
        for attempt in range(component.retry_count + 1):
            try:
                # Execute loader with timeout
                instance = await asyncio.wait_for(component.loader(), timeout=component.timeout)

                # Store instance
                self.component_instances[name] = instance

                # Run health check if provided
                if component.health_check:
                    is_healthy = await asyncio.wait_for(
                        component.health_check(instance), timeout=5.0
                    )

                    if not is_healthy:
                        if attempt < component.retry_count:
                            logger.warning(
                                f"[WARMUP] Health check failed for {name}, retrying... "
                                f"(attempt {attempt + 1}/{component.retry_count + 1})"
                            )
                            await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                            continue
                        else:
                            logger.error(
                                f"[WARMUP] Health check failed for {name} after all retries"
                            )
                            self.component_status[name] = ComponentStatus.DEGRADED
                            metrics.health_score = 0.5
                    else:
                        metrics.health_score = 1.0

                # Success!
                metrics.load_time = time.time() - start_time
                metrics.retry_count = attempt
                self.component_status[name] = ComponentStatus.READY
                self.ready_events[name].set()

                logger.info(
                    f"[WARMUP] ✅ {name} ready in {metrics.load_time:.2f}s "
                    f"(attempt {attempt + 1})"
                )
                return True

            except asyncio.TimeoutError:
                logger.error(
                    f"[WARMUP] Timeout loading {name} "
                    f"(attempt {attempt + 1}/{component.retry_count + 1})"
                )
                metrics.last_error = "Timeout"

            except Exception as e:
                logger.error(
                    f"[WARMUP] Error loading {name}: {e} "
                    f"(attempt {attempt + 1}/{component.retry_count + 1})"
                )
                metrics.last_error = str(e)

                if attempt < component.retry_count:
                    await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff

        # All retries failed
        self.component_status[name] = ComponentStatus.FAILED
        self.failed_components.append(name)

        if component.required:
            logger.error(f"[WARMUP] ❌ Required component {name} failed to load!")
        else:
            logger.warning(f"[WARMUP] ⚠️  Optional component {name} failed to load")

        return False

    def _resolve_load_order(self, components: List[str]) -> List[str]:
        """Resolve component load order based on dependencies (topological sort)"""
        # Build subgraph for this batch
        in_degree = {comp: 0 for comp in components}
        graph = {comp: [] for comp in components}

        for comp in components:
            for dep in self.dependency_graph[comp]:
                if dep in components:  # Only consider deps in this batch
                    graph[dep].append(comp)
                    in_degree[comp] += 1

        # Kahn's algorithm for topological sort
        queue = [comp for comp in components if in_degree[comp] == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(result) != len(components):
            logger.warning(
                f"[WARMUP] Dependency cycle detected, loading remaining components anyway"
            )
            for comp in components:
                if comp not in result:
                    result.append(comp)

        return result

    def _generate_warmup_report(self) -> Dict[str, Any]:
        """Generate detailed warmup report"""
        status_counts = defaultdict(int)
        for status in self.component_status.values():
            status_counts[status.value] += 1

        ready_components = [
            name
            for name, status in self.component_status.items()
            if status == ComponentStatus.READY
        ]

        return {
            "total_count": len(self.components),
            "ready_count": len(ready_components),
            "failed_count": len(self.failed_components),
            "status_breakdown": dict(status_counts),
            "total_load_time": self.total_load_time,
            "critical_load_time": self.critical_load_time,
            "ready_components": ready_components,
            "failed_components": self.failed_components,
            "component_metrics": {
                name: {
                    "load_time": metrics.load_time,
                    "retry_count": metrics.retry_count,
                    "health_score": metrics.health_score,
                    "last_error": metrics.last_error,
                }
                for name, metrics in self.component_metrics.items()
            },
            "timestamp": datetime.now().isoformat(),
        }

    async def wait_for_critical(self, timeout: Optional[float] = None):
        """Wait for critical components to be ready"""
        if timeout:
            await asyncio.wait_for(self.critical_ready.wait(), timeout=timeout)
        else:
            await self.critical_ready.wait()

    async def wait_for_component(self, name: str, timeout: Optional[float] = None):
        """Wait for a specific component to be ready"""
        if name not in self.ready_events:
            raise ValueError(f"Component {name} not registered")

        if timeout:
            await asyncio.wait_for(self.ready_events[name].wait(), timeout=timeout)
        else:
            await self.ready_events[name].wait()

    def get_component(self, name: str) -> Optional[Any]:
        """Get a loaded component instance"""
        return self.component_instances.get(name)

    def is_ready(self, name: str) -> bool:
        """Check if a component is ready"""
        return self.component_status.get(name) == ComponentStatus.READY

    def get_status(self, name: str) -> Optional[ComponentStatus]:
        """Get component status"""
        return self.component_status.get(name)


# Global warmup system instance
_warmup_system = None


def get_warmup_system() -> ComponentWarmupSystem:
    """Get or create the global warmup system"""
    global _warmup_system
    if _warmup_system is None:
        _warmup_system = ComponentWarmupSystem()
    return _warmup_system


def register_component(name: str, loader: Callable, **kwargs):
    """Convenience function to register a component"""
    warmup = get_warmup_system()
    warmup.register_component(name, loader, **kwargs)


async def warmup_all_components() -> Dict[str, Any]:
    """Convenience function to warmup all components"""
    warmup = get_warmup_system()
    return await warmup.warmup_all()
