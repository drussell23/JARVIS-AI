"""
Adaptive Routing Engine with Plugin Architecture

This module provides a comprehensive routing system that dynamically routes
follow-ups and intents to appropriate handlers. It features a plugin architecture
for extensibility, middleware support for request processing, and flexible
route matching based on intent classification and context requirements.

The main components include:
- AdaptiveRouter: Core routing engine
- RouteHandler: Protocol for implementing handlers
- RouteMatcher: Intent and context matching logic
- PluginRegistry: Dynamic plugin management
- Built-in middleware for common functionality

Example:
    >>> from backend.core.routing.adaptive_router import AdaptiveRouter, RouteMatcher
    >>> matcher = RouteMatcher()
    >>> router = AdaptiveRouter(matcher)
    >>> # Add routes and handlers...
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


@dataclass(frozen=True)
class RoutingResult:
    """Result of a routing operation.
    
    Contains the outcome of processing a user request through the routing system,
    including success status, response message, metadata, and error information.
    
    Attributes:
        success: Whether the routing and handling was successful
        response: The response message to return to the user
        metadata: Additional data about the routing operation
        error: Error message if the operation failed, None otherwise
    """
    success: bool
    response: str
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class RouteHandler(Protocol):
    """Protocol defining the interface for route handlers.
    
    Route handlers are responsible for processing user requests that have been
    classified with specific intents and optionally have associated context.
    """

    async def handle(
        self,
        user_input: str,
        intent: IntentResult,
        context: ContextEnvelope | None,
        extras: dict[str, Any],
    ) -> RoutingResult:
        """Handle a routed request.

        Args:
            user_input: Raw user text input
            intent: Classified intent with confidence score
            context: Active context envelope if available
            extras: Additional routing metadata and parameters

        Returns:
            RoutingResult containing the response and operation metadata
        """
        ...


class BaseRouteHandler(ABC):
    """Base class for route handlers with common utilities.
    
    Provides a foundation for implementing route handlers with built-in
    error handling, priority management, and safe execution wrappers.
    
    Attributes:
        _name: Handler identifier name
        _priority: Handler priority for route ordering (higher = higher priority)
    """

    def __init__(self, name: str, priority: int = 50):
        """Initialize the base route handler.
        
        Args:
            name: Unique identifier for this handler
            priority: Priority level for route matching (default: 50)
        """
        self._name = name
        self._priority = priority

    @property
    def name(self) -> str:
        """Get the handler name.
        
        Returns:
            The handler's unique identifier
        """
        return self._name

    @property
    def priority(self) -> int:
        """Get the handler priority.
        
        Returns:
            The handler's priority level for route ordering
        """
        return self._priority

    @abstractmethod
    async def handle(
        self,
        user_input: str,
        intent: IntentResult,
        context: ContextEnvelope | None,
        extras: dict[str, Any],
    ) -> RoutingResult:
        """Implement routing logic.
        
        Subclasses must implement this method to define their specific
        request handling behavior.
        
        Args:
            user_input: Raw user text input
            intent: Classified intent with confidence score
            context: Active context envelope if available
            extras: Additional routing metadata and parameters
            
        Returns:
            RoutingResult containing the response and operation metadata
        """
        ...

    async def handle_safe(
        self,
        user_input: str,
        intent: IntentResult,
        context: ContextEnvelope | None,
        extras: dict[str, Any],
    ) -> RoutingResult:
        """Wrapped handle with error handling.
        
        Provides a safe wrapper around the handle method that catches
        exceptions and returns appropriate error responses.
        
        Args:
            user_input: Raw user text input
            intent: Classified intent with confidence score
            context: Active context envelope if available
            extras: Additional routing metadata and parameters
            
        Returns:
            RoutingResult with either successful response or error information
        """
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


@dataclass
class RouteConfig:
    """Configuration for a route mapping.
    
    Defines how intents should be matched to handlers, including requirements
    for context presence, category constraints, and confidence thresholds.
    
    Attributes:
        intent_label: The intent label this route handles
        handler: The handler instance to process matching requests
        requires_context: Whether this route requires active context
        context_categories: Tuple of allowed context categories (empty = any)
        min_confidence: Minimum confidence threshold for intent matching
        fallback_handler: Optional fallback handler if primary fails
    """
    intent_label: str
    handler: BaseRouteHandler
    requires_context: bool = False
    context_categories: tuple[str, ...] = field(default_factory=tuple)
    min_confidence: float = 0.5
    fallback_handler: BaseRouteHandler | None = None


class RouteMatcher:
    """Matches intents and contexts to appropriate routes.
    
    Responsible for finding the best matching route configuration based on
    intent classification results and available context information.
    
    Attributes:
        _routes: List of registered route configurations, sorted by priority
    """

    def __init__(self):
        """Initialize the route matcher with empty route list."""
        self._routes: list[RouteConfig] = []

    def add_route(self, config: RouteConfig) -> None:
        """Register a route configuration.
        
        Adds a new route and maintains priority-based sorting of all routes.
        
        Args:
            config: Route configuration to register
        """
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
        """Find the best matching route for given intent and context.

        Applies matching criteria in order:
        1. Intent label matches exactly
        2. Confidence meets minimum threshold
        3. Context requirements are satisfied
        4. Category constraints are met (if specified)

        Args:
            intent: Classified intent result with confidence
            context: Available context envelope or None

        Returns:
            First matching RouteConfig or None if no match found
            
        Example:
            >>> matcher = RouteMatcher()
            >>> # Add routes...
            >>> match = matcher.find_match(intent_result, context_envelope)
            >>> if match:
            ...     handler = match.handler
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
        """Get all registered routes.
        
        Returns:
            Copy of the routes list to prevent external modification
        """
        return self._routes.copy()


class AdaptiveRouter:
    """Main routing engine that orchestrates request processing.
    
    Coordinates intent classification, context retrieval, middleware execution,
    and handler dispatching to provide a complete routing solution.
    
    Attributes:
        _matcher: Route matcher for finding appropriate handlers
        _default_handler: Fallback handler when no routes match
        _middleware: List of middleware functions to apply
    """

    def __init__(
        self,
        matcher: RouteMatcher,
        default_handler: BaseRouteHandler | None = None,
    ):
        """Initialize the adaptive router.
        
        Args:
            matcher: Route matcher instance for finding handlers
            default_handler: Optional fallback handler for unmatched requests
        """
        self._matcher = matcher
        self._default_handler = default_handler
        self._middleware: list[Callable] = []

    def add_route(self, config: RouteConfig) -> None:
        """Add a route configuration to the matcher.
        
        Args:
            config: Route configuration to register
        """
        self._matcher.add_route(config)

    def use_middleware(self, middleware: Callable) -> None:
        """Add middleware function to the processing chain.
        
        Middleware functions are called in registration order and can modify
        the request parameters before routing occurs.
        
        Args:
            middleware: Async function with signature:
                async def (user_input, intent, context, extras) -> 
                    (user_input, intent, context, extras)
        """
        self._middleware.append(middleware)

    async def route(
        self,
        user_input: str,
        intent: IntentResult,
        context: ContextEnvelope | None = None,
        extras: dict[str, Any] | None = None,
    ) -> RoutingResult:
        """Route request to appropriate handler.

        Executes the complete routing flow:
        1. Apply middleware chain to modify request parameters
        2. Find matching route using the matcher
        3. Execute the selected handler
        4. Return the result with logging

        Args:
            user_input: Raw user text input
            intent: Classified intent result
            context: Optional context envelope
            extras: Additional routing parameters

        Returns:
            RoutingResult containing response and metadata

        Raises:
            No exceptions are raised - all errors are captured in RoutingResult
            
        Example:
            >>> router = AdaptiveRouter(matcher)
            >>> result = await router.route("Hello", intent_result)
            >>> if result.success:
            ...     print(result.response)
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
    """Log routing details for debugging and monitoring.
    
    Middleware function that logs key routing information without modifying
    the request parameters.
    
    Args:
        user_input: Raw user text input
        intent: Classified intent result
        context: Optional context envelope
        extras: Additional routing parameters
        
    Returns:
        Tuple of unmodified parameters (user_input, intent, context, extras)
    """
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
    """Validate and refresh context if present.
    
    Middleware function that checks context validity and clears invalid
    contexts to prevent routing errors.
    
    Args:
        user_input: Raw user text input
        intent: Classified intent result
        context: Optional context envelope to validate
        extras: Additional routing parameters (modified with validation info)
        
    Returns:
        Tuple with potentially modified context and extras
    """
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
    """Simple rate limiting middleware.
    
    Placeholder implementation for rate limiting functionality.
    Can be enhanced with Redis or other storage backends for
    per-user rate tracking.
    
    Args:
        user_input: Raw user text input
        intent: Classified intent result
        context: Optional context envelope
        extras: Additional routing parameters
        
    Returns:
        Tuple of unmodified parameters (user_input, intent, context, extras)
        
    Note:
        Current implementation is a placeholder - implement per-user
        rate tracking as needed.
    """
    # Placeholder - implement per-user rate tracking
    return user_input, intent, context, extras


# Plugin registry

class HandlerPlugin(ABC):
    """Base class for handler plugins.
    
    Plugins provide a way to dynamically add routing capabilities to the
    system. Each plugin defines routes and can perform initialization
    and cleanup operations.
    """

    @property
    @abstractmethod
    def routes(self) -> list[RouteConfig]:
        """Return route configurations for this plugin.
        
        Returns:
            List of RouteConfig objects defining the plugin's routes
        """
        ...

    async def on_load(self) -> None:
        """Called when plugin is loaded.
        
        Override to perform plugin initialization, such as setting up
        resources, connections, or configuration.
        """
        pass

    async def on_unload(self) -> None:
        """Called when plugin is unloaded.
        
        Override to perform cleanup operations, such as closing connections
        or releasing resources.
        """
        pass


class PluginRegistry:
    """Manage handler plugins dynamically.
    
    Provides functionality to register, unregister, and manage plugins
    that extend the routing system's capabilities.
    
    Attributes:
        _router: The adaptive router to register plugin routes with
        _plugins: Dictionary mapping plugin names to plugin instances
    """

    def __init__(self, router: AdaptiveRouter):
        """Initialize the plugin registry.
        
        Args:
            router: AdaptiveRouter instance to register plugin routes with
        """
        self._router = router
        self._plugins: dict[str, HandlerPlugin] = {}

    async def register_plugin(self, name: str, plugin: HandlerPlugin) -> None:
        """Register and load a plugin.
        
        Adds the plugin's routes to the router and calls the plugin's
        initialization method.
        
        Args:
            name: Unique identifier for the plugin
            plugin: Plugin instance to register
            
        Note:
            If a plugin with the same name exists, it will be replaced
            after proper unloading.
        """
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
        """Unregister and unload a plugin.
        
        Removes the plugin from the registry and calls its cleanup method.
        Note that routes are not automatically removed from the router.
        
        Args:
            name: Name of the plugin to unregister
        """
        plugin = self._plugins.pop(name, None)
        if plugin:
            await plugin.on_unload()
            logger.info(f"Unregistered plugin '{name}'")

    def get_plugin(self, name: str) -> HandlerPlugin | None:
        """Retrieve a plugin by name.
        
        Args:
            name: Name of the plugin to retrieve
            
        Returns:
            Plugin instance if found, None otherwise
        """
        return self._plugins.get(name)

    @property
    def plugin_names(self) -> list[str]:
        """Get all registered plugin names.
        
        Returns:
            List of currently registered plugin names
        """
        return list(self._plugins.keys())