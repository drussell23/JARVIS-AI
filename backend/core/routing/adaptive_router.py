"""
Adaptive Routing Engine with Plugin Architecture
Routes follow-ups and intents to appropriate handlers dynamically.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from backend.core.models.context_envelope import ContextEnvelope, ContextPayload
from backend.core.intent.adaptive_classifier import IntentResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RoutingResult:
    """Result of routing operation."""
    success: bool
    response: str
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class RouteHandler(Protocol):
    """Protocol for route handlers."""

    async def handle(
        self,
        user_input: str,
        intent: IntentResult,
        context: ContextEnvelope | None,
        extras: dict[str, Any],
    ) -> RoutingResult:
        """
        Handle a routed request.

        Args:
            user_input: Raw user text
            intent: Classified intent
            context: Active context envelope (if any)
            extras: Additional routing metadata

        Returns:
            RoutingResult with response and metadata
        """
        ...


class BaseRouteHandler(ABC):
    """Base class for route handlers with common utilities."""

    def __init__(self, name: str, priority: int = 50):
        self._name = name
        self._priority = priority

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    @abstractmethod
    async def handle(
        self,
        user_input: str,
        intent: IntentResult,
        context: ContextEnvelope | None,
        extras: dict[str, Any],
    ) -> RoutingResult:
        """Implement routing logic."""
        ...

    async def handle_safe(
        self,
        user_input: str,
        intent: IntentResult,
        context: ContextEnvelope | None,
        extras: dict[str, Any],
    ) -> RoutingResult:
        """Wrapped handle with error handling."""
        try:
            return await self.handle(user_input, intent, context, extras)
        except Exception as e:
            logger.error(
                f"Handler {self.name} failed for input '{user_input[:50]}...': {e}",
                exc_info=True,
            )
            return RoutingResult(
                success=False,
                response="I encountered an error processing your request.",
                error=str(e),
            )


@dataclass(slots=True)
class RouteConfig:
    """Configuration for a route."""
    intent_label: str
    handler: BaseRouteHandler
    requires_context: bool = False
    context_categories: tuple[str, ...] = field(default_factory=tuple)
    min_confidence: float = 0.5
    fallback_handler: BaseRouteHandler | None = None


class RouteMatcher:
    """Matches intents and contexts to routes."""

    def __init__(self):
        self._routes: list[RouteConfig] = []

    def add_route(self, config: RouteConfig) -> None:
        """Register a route."""
        self._routes.append(config)
        # Sort by handler priority (descending)
        self._routes.sort(key=lambda r: r.handler.priority, reverse=True)
        logger.debug(
            f"Registered route: {config.intent_label} -> {config.handler.name} "
            f"(requires_context={config.requires_context})"
        )

    def find_match(
        self, intent: IntentResult, context: ContextEnvelope | None
    ) -> RouteConfig | None:
        """
        Find best matching route.

        Matching criteria:
        1. Intent label matches
        2. Confidence above threshold
        3. Context requirements satisfied
        4. Category constraints satisfied
        """
        for route in self._routes:
            # Check intent label
            if route.intent_label != intent.primary_intent:
                continue

            # Check confidence
            if intent.confidence < route.min_confidence:
                continue

            # Check context requirement
            if route.requires_context and context is None:
                continue

            # Check category constraints
            if route.context_categories and context:
                if context.metadata.category.name not in route.context_categories:
                    continue

            # Match found
            return route

        return None

    def get_all_routes(self) -> list[RouteConfig]:
        """Get all registered routes."""
        return self._routes.copy()


class AdaptiveRouter:
    """
    Main routing engine that orchestrates intent classification,
    context retrieval, and handler dispatching.
    """

    def __init__(
        self,
        matcher: RouteMatcher,
        default_handler: BaseRouteHandler | None = None,
    ):
        self._matcher = matcher
        self._default_handler = default_handler
        self._middleware: list[Callable] = []

    def add_route(self, config: RouteConfig) -> None:
        """Add route to matcher."""
        self._matcher.add_route(config)

    def use_middleware(self, middleware: Callable) -> None:
        """
        Add middleware function.
        Signature: async def (user_input, intent, context, extras) -> (input, intent, context, extras)
        """
        self._middleware.append(middleware)

    async def route(
        self,
        user_input: str,
        intent: IntentResult,
        context: ContextEnvelope | None = None,
        extras: dict[str, Any] | None = None,
    ) -> RoutingResult:
        """
        Route request to appropriate handler.

        Flow:
        1. Run middleware chain
        2. Find matching route
        3. Execute handler
        4. Return result
        """
        extras = extras or {}

        # Apply middleware
        for mw in self._middleware:
            try:
                user_input, intent, context, extras = await mw(
                    user_input, intent, context, extras
                )
            except Exception as e:
                logger.error(f"Middleware {mw.__name__} failed: {e}", exc_info=True)

        # Find route
        route = self._matcher.find_match(intent, context)

        if route:
            handler = route.handler
            logger.info(
                f"Routing '{user_input[:50]}...' to handler '{handler.name}' "
                f"(intent={intent.primary_intent}, confidence={intent.confidence:.2f})"
            )
        elif self._default_handler:
            handler = self._default_handler
            logger.info(
                f"No route match, using default handler '{handler.name}'"
            )
        else:
            logger.warning(f"No route or default handler for intent '{intent.primary_intent}'")
            return RoutingResult(
                success=False,
                response="I'm not sure how to handle that request.",
                metadata={"intent": intent.primary_intent, "confidence": intent.confidence},
            )

        # Execute handler
        result = await handler.handle_safe(user_input, intent, context, extras)

        # Log result
        if result.success:
            logger.debug(f"Handler '{handler.name}' succeeded")
        else:
            logger.warning(
                f"Handler '{handler.name}' failed: {result.error or 'unknown error'}"
            )

        return result


# Built-in middleware

async def logging_middleware(
    user_input: str,
    intent: IntentResult,
    context: ContextEnvelope | None,
    extras: dict[str, Any],
) -> tuple:
    """Log routing details."""
    logger.debug(
        f"Routing: input='{user_input[:30]}...', "
        f"intent={intent.primary_intent}, "
        f"context={'present' if context else 'none'}"
    )
    return user_input, intent, context, extras


async def context_validation_middleware(
    user_input: str,
    intent: IntentResult,
    context: ContextEnvelope | None,
    extras: dict[str, Any],
) -> tuple:
    """Validate and refresh context if present."""
    if context and not context.is_valid():
        logger.warning(f"Context {context.metadata.id} is invalid, clearing")
        context = None
        extras["context_invalid"] = True
    return user_input, intent, context, extras


async def rate_limiting_middleware(
    user_input: str,
    intent: IntentResult,
    context: ContextEnvelope | None,
    extras: dict[str, Any],
) -> tuple:
    """Simple rate limiting (can be enhanced with Redis)."""
    # Placeholder - implement per-user rate tracking
    return user_input, intent, context, extras


# Plugin registry

class HandlerPlugin(ABC):
    """Base class for handler plugins."""

    @property
    @abstractmethod
    def routes(self) -> list[RouteConfig]:
        """Return route configurations for this plugin."""
        ...

    async def on_load(self) -> None:
        """Called when plugin is loaded."""
        pass

    async def on_unload(self) -> None:
        """Called when plugin is unloaded."""
        pass


class PluginRegistry:
    """Manage handler plugins."""

    def __init__(self, router: AdaptiveRouter):
        self._router = router
        self._plugins: dict[str, HandlerPlugin] = {}

    async def register_plugin(self, name: str, plugin: HandlerPlugin) -> None:
        """Register and load plugin."""
        if name in self._plugins:
            logger.warning(f"Plugin '{name}' already registered, replacing")
            await self.unregister_plugin(name)

        self._plugins[name] = plugin

        # Add routes
        for route in plugin.routes:
            self._router.add_route(route)

        # Initialize
        await plugin.on_load()

        logger.info(
            f"Registered plugin '{name}' with {len(plugin.routes)} route(s)"
        )

    async def unregister_plugin(self, name: str) -> None:
        """Unregister and unload plugin."""
        plugin = self._plugins.pop(name, None)
        if plugin:
            await plugin.on_unload()
            logger.info(f"Unregistered plugin '{name}'")

    def get_plugin(self, name: str) -> HandlerPlugin | None:
        """Retrieve plugin by name."""
        return self._plugins.get(name)

    @property
    def plugin_names(self) -> list[str]:
        """Get all registered plugin names."""
        return list(self._plugins.keys())
