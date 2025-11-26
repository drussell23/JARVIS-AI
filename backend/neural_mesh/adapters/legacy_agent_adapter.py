"""
JARVIS Neural Mesh - Legacy Agent Adapter

Adapter class that wraps existing JARVIS agents to make them
compatible with the Neural Mesh system.

This allows existing agents to:
- Be registered in the Agent Registry
- Send and receive messages via the Communication Bus
- Access the Shared Knowledge Graph
- Participate in multi-agent workflows

Usage:
    # Wrap an existing agent
    existing_agent = VisionAnalyzer()
    adapted = LegacyAgentAdapter(
        wrapped_agent=existing_agent,
        agent_name="vision_analyzer",
        agent_type="vision",
        capabilities={"screen_capture", "error_detection"},
        task_handler=existing_agent.analyze,  # Method to handle tasks
    )

    # Register with Neural Mesh
    await coordinator.register_agent(adapted)
"""

from __future__ import annotations

import asyncio
import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Union,
)

from ..base.base_neural_mesh_agent import BaseNeuralMeshAgent

logger = logging.getLogger(__name__)


class LegacyAgentAdapter(BaseNeuralMeshAgent):
    """
    Adapter that wraps existing agents for Neural Mesh compatibility.

    This adapter allows you to integrate existing JARVIS agents without
    modifying their original code. Simply wrap them with this adapter
    and they gain full Neural Mesh capabilities.

    Example - Wrapping a simple agent:
        class ExistingVisionAgent:
            def analyze_screen(self, space_id: int) -> dict:
                # Original logic
                return {"errors": [...]}

        # Wrap it
        existing = ExistingVisionAgent()
        adapted = LegacyAgentAdapter(
            wrapped_agent=existing,
            agent_name="vision_agent",
            agent_type="vision",
            capabilities={"screen_capture", "error_detection"},
            task_handler=existing.analyze_screen,
        )

    Example - Wrapping with multiple task handlers:
        adapted = LegacyAgentAdapter(
            wrapped_agent=existing,
            agent_name="vision_agent",
            agent_type="vision",
            capabilities={"screen_capture", "error_detection", "ocr"},
            task_handlers={
                "screen_capture": existing.capture,
                "error_detection": existing.detect_errors,
                "ocr": existing.extract_text,
            },
        )
    """

    def __init__(
        self,
        wrapped_agent: Any,
        agent_name: str,
        agent_type: str,
        capabilities: Set[str],
        task_handler: Optional[Callable[..., Any]] = None,
        task_handlers: Optional[Dict[str, Callable[..., Any]]] = None,
        init_handler: Optional[Callable[[], Any]] = None,
        cleanup_handler: Optional[Callable[[], Any]] = None,
        backend: str = "local",
        version: str = "1.0.0",
    ) -> None:
        """Initialize the adapter.

        Args:
            wrapped_agent: The existing agent instance to wrap
            agent_name: Unique name for this agent
            agent_type: Category of agent
            capabilities: Set of capabilities this agent provides
            task_handler: Default handler for all tasks (if single handler)
            task_handlers: Dict mapping capabilities to handler methods
            init_handler: Optional initialization method to call
            cleanup_handler: Optional cleanup method to call on stop
            backend: Where this agent runs
            version: Agent version
        """
        super().__init__(
            agent_name=agent_name,
            agent_type=agent_type,
            capabilities=capabilities,
            backend=backend,
            version=version,
        )

        self._wrapped_agent = wrapped_agent
        self._task_handler = task_handler
        self._task_handlers = task_handlers or {}
        self._init_handler = init_handler
        self._cleanup_handler = cleanup_handler

        # Validate handlers
        if not task_handler and not task_handlers:
            logger.warning(
                "No task handlers provided for %s - tasks will fail",
                agent_name,
            )

    async def on_initialize(self) -> None:
        """Initialize the wrapped agent if it has an init method."""
        if self._init_handler:
            result = self._init_handler()
            if asyncio.iscoroutine(result):
                await result

        # If wrapped agent has initialize method, call it
        if hasattr(self._wrapped_agent, "initialize"):
            result = self._wrapped_agent.initialize()
            if asyncio.iscoroutine(result):
                await result

        logger.info(
            "Initialized legacy adapter for %s",
            self._wrapped_agent.__class__.__name__,
        )

    async def on_stop(self) -> None:
        """Cleanup the wrapped agent if it has a cleanup method."""
        if self._cleanup_handler:
            result = self._cleanup_handler()
            if asyncio.iscoroutine(result):
                await result

        # If wrapped agent has cleanup/close method, call it
        for method_name in ("cleanup", "close", "shutdown", "stop"):
            if hasattr(self._wrapped_agent, method_name):
                method = getattr(self._wrapped_agent, method_name)
                result = method()
                if asyncio.iscoroutine(result):
                    await result
                break

    async def execute_task(self, payload: Dict[str, Any]) -> Any:
        """Execute a task using the wrapped agent's handlers."""
        action = payload.get("action", "")
        input_data = payload.get("input", {})

        # Find appropriate handler
        handler = None

        # Check capability-specific handlers
        if action in self._task_handlers:
            handler = self._task_handlers[action]
        elif self._task_handler:
            handler = self._task_handler
        else:
            raise ValueError(f"No handler for action: {action}")

        # Execute handler
        try:
            # Handle both sync and async handlers
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**input_data)
            else:
                # Run sync handler in executor to not block
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: handler(**input_data),
                )

            return result

        except TypeError as e:
            # Handler might not accept kwargs, try with just the payload
            logger.debug(
                "Retrying handler call with single payload argument: %s",
                e,
            )

            if asyncio.iscoroutinefunction(handler):
                result = await handler(input_data)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: handler(input_data),
                )

            return result

    @property
    def wrapped(self) -> Any:
        """Access the wrapped agent directly."""
        return self._wrapped_agent

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped agent."""
        # Avoid infinite recursion for internal attributes
        if name.startswith("_"):
            raise AttributeError(name)

        # Delegate to wrapped agent
        if hasattr(self._wrapped_agent, name):
            return getattr(self._wrapped_agent, name)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


def adapt_agent(
    agent: Any,
    name: str,
    agent_type: str,
    capabilities: Set[str],
    **kwargs: Any,
) -> LegacyAgentAdapter:
    """
    Convenience function to create a LegacyAgentAdapter.

    This function tries to automatically detect task handlers
    based on the agent's methods matching capability names.

    Args:
        agent: The agent instance to wrap
        name: Unique name for this agent
        agent_type: Category of agent
        capabilities: Set of capabilities
        **kwargs: Additional arguments passed to LegacyAgentAdapter

    Returns:
        LegacyAgentAdapter wrapping the agent

    Example:
        # Auto-detect handlers based on capabilities
        adapted = adapt_agent(
            existing_agent,
            name="vision_agent",
            agent_type="vision",
            capabilities={"capture", "analyze"},  # Will look for capture() and analyze() methods
        )
    """
    # Try to auto-detect task handlers
    task_handlers: Dict[str, Callable[..., Any]] = {}

    for capability in capabilities:
        # Try exact match
        if hasattr(agent, capability):
            task_handlers[capability] = getattr(agent, capability)
            continue

        # Try with underscores
        method_name = capability.replace("-", "_").replace(" ", "_")
        if hasattr(agent, method_name):
            task_handlers[method_name] = getattr(agent, method_name)
            task_handlers[capability] = getattr(agent, method_name)
            continue

        # Try common variations
        for prefix in ("do_", "handle_", "execute_", "run_", ""):
            if hasattr(agent, f"{prefix}{method_name}"):
                task_handlers[capability] = getattr(agent, f"{prefix}{method_name}")
                break

    return LegacyAgentAdapter(
        wrapped_agent=agent,
        agent_name=name,
        agent_type=agent_type,
        capabilities=capabilities,
        task_handlers=task_handlers if task_handlers else None,
        **kwargs,
    )
