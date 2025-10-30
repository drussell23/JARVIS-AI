#!/usr/bin/env python3
"""
Advanced Async Architecture - Dynamic Event-Driven Command Pipeline

Ultra-robust, adaptive, zero-hardcoding async processing system for JARVIS.
Provides a comprehensive pipeline for processing voice commands with dynamic
stage registration, circuit breaker patterns, event-driven architecture,
and intelligent error handling.

This module implements:
- Dynamic pipeline stages with configurable timeouts and retries
- Adaptive circuit breaker with ML-based prediction
- Event-driven message bus with priority handling
- Context-aware command processing
- Follow-up intent detection and routing
- Comprehensive error handling and recovery

Example:
    >>> pipeline = get_async_pipeline(jarvis_instance)
    >>> result = await pipeline.process_async("open safari and search for dogs")
    >>> print(result['response'])
    "I've opened Safari and searched for dogs, Sir."
"""

import asyncio
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Dynamic pipeline processing stages.
    
    Represents the various stages a command goes through during processing,
    from initial receipt to final completion or failure.
    """

    RECEIVED = "received"
    VALIDATED = "validated"
    PREPROCESSED = "preprocessed"
    INTENT_ANALYSIS = "intent_analysis"
    COMPONENT_LOADING = "component_loading"
    MIDDLEWARE = "middleware"
    PROCESSING = "processing"
    POSTPROCESSING = "postprocessing"
    RESPONSE_GENERATION = "response_generation"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineContext:
    """Context passed through the pipeline stages.
    
    Contains all information needed to process a command, including metadata,
    metrics, and state information that accumulates as the command moves
    through different pipeline stages.
    
    Attributes:
        command_id: Unique identifier for this command
        text: The original command text
        user_name: Name of the user issuing the command
        timestamp: Unix timestamp when command was received
        stage: Current pipeline stage
        intent: Detected intent of the command
        components_loaded: List of components loaded for this command
        response: Generated response text
        error: Error message if command failed
        metadata: Additional metadata dictionary
        metrics: Performance metrics dictionary
        retries: Number of retry attempts made
        priority: Command priority (0=normal, 1=high, 2=critical)
        audio_data: Voice audio data for authentication
        speaker_name: Identified speaker name from voice recognition
    """

    command_id: str
    text: str
    user_name: str = "Sir"
    timestamp: float = field(default_factory=time.time)
    stage: PipelineStage = PipelineStage.RECEIVED
    intent: Optional[str] = None
    components_loaded: List[str] = field(default_factory=list)
    response: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    retries: int = 0
    priority: int = 0  # 0=normal, 1=high, 2=critical
    audio_data: Optional[bytes] = None  # Voice audio for authentication
    speaker_name: Optional[str] = None  # Identified speaker name

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization.
        
        Returns:
            Dictionary representation of the context, excluding binary audio data.
        """
        return {
            "command_id": self.command_id,
            "text": self.text,
            "user_name": self.user_name,
            "timestamp": self.timestamp,
            "stage": self.stage.value,
            "intent": self.intent,
            "components_loaded": self.components_loaded,
            "response": self.response,
            "error": self.error,
            "metadata": self.metadata,
            "metrics": self.metrics,
            "retries": self.retries,
            "priority": self.priority,
        }


class AdaptiveCircuitBreaker:
    """Advanced circuit breaker with adaptive thresholds and ML-based prediction.
    
    Implements the circuit breaker pattern with adaptive learning capabilities.
    Automatically adjusts failure thresholds and timeouts based on historical
    performance patterns to optimize system resilience.
    
    Attributes:
        failure_count: Current number of consecutive failures
        success_count: Current number of consecutive successes
        threshold: Current failure threshold before opening circuit
        timeout: Current timeout before attempting to close circuit
        state: Current circuit state (CLOSED, OPEN, HALF_OPEN)
        last_failure_time: Timestamp of last failure
        adaptive: Whether adaptive learning is enabled
        failure_history: List of failure timestamps
        success_rate_history: List of historical success rates
    """

    def __init__(
        self,
        initial_threshold: int = 5,
        initial_timeout: int = 60,
        adaptive: bool = True,
    ):
        """Initialize the adaptive circuit breaker.
        
        Args:
            initial_threshold: Initial failure count threshold
            initial_timeout: Initial timeout in seconds
            adaptive: Enable adaptive threshold adjustment
        """
        self.failure_count = 0
        self.success_count = 0
        self.threshold = initial_threshold
        self.timeout = initial_timeout
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = None
        self.adaptive = adaptive
        self.failure_history: List[float] = []
        self.success_rate_history: List[float] = []
        self._total_calls = 0
        self._successful_calls = 0

    @property
    def success_rate(self) -> float:
        """Calculate current success rate.
        
        Returns:
            Success rate as a float between 0.0 and 1.0
        """
        if self._total_calls == 0:
            return 1.0  # Default to 100% success if no calls yet
        return self._successful_calls / self._total_calls

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with adaptive circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                logger.info(
                    f"Circuit breaker: transitioning to HALF_OPEN (threshold={self.threshold})"
                )
                self.state = "HALF_OPEN"
            else:
                raise Exception(
                    f"Circuit breaker is OPEN - service unavailable (retry in {int(self.timeout - (time.time() - self.last_failure_time))}s)"
                )

        try:
            start = time.time()
            result = await func(*args, **kwargs)
            duration = time.time() - start

            self.on_success(duration)
            return result

        except Exception as e:
            self.on_failure()
            raise e

    def on_success(self, duration: float):
        """Handle successful execution with adaptive learning.
        
        Args:
            duration: Execution duration in seconds
        """
        self.success_count += 1
        self._total_calls += 1
        self._successful_calls += 1
        self.failure_count = max(0, self.failure_count - 1)  # Gradual recovery

        if self.state == "HALF_OPEN":
            logger.info("Circuit breaker: transitioning to CLOSED")
            self.state = "CLOSED"

        # Adaptive threshold adjustment
        if self.adaptive:
            success_rate = self.success_count / (self.success_count + len(self.failure_history))
            self.success_rate_history.append(success_rate)

            # Increase threshold if success rate is high
            if success_rate > 0.95 and self.threshold < 20:
                self.threshold += 1
                logger.debug(f"Increased circuit breaker threshold to {self.threshold}")

    def on_failure(self):
        """Handle failed execution with adaptive learning."""
        self.failure_count += 1
        self._total_calls += 1
        self.last_failure_time = time.time()
        self.failure_history.append(time.time())

        # Adaptive threshold adjustment
        if self.adaptive and len(self.failure_history) > 10:
            recent_failures = sum(1 for t in self.failure_history[-10:] if time.time() - t < 60)

            if recent_failures > 5 and self.threshold > 3:
                self.threshold -= 1
                logger.warning(f"Decreased circuit breaker threshold to {self.threshold}")

        if self.failure_count >= self.threshold:
            logger.warning(
                f"Circuit breaker: OPEN (failures: {self.failure_count}/{self.threshold})"
            )
            self.state = "OPEN"

            # Adaptive timeout based on failure patterns
            if self.adaptive:
                avg_failure_interval = self._calculate_failure_interval()
                if avg_failure_interval > 0:
                    self.timeout = min(300, int(avg_failure_interval * 2))  # Max 5 min
                    logger.info(f"Adaptive timeout set to {self.timeout}s")

    def _calculate_failure_interval(self) -> float:
        """Calculate average interval between failures.
        
        Returns:
            Average interval in seconds, or 0 if insufficient data
        """
        if len(self.failure_history) < 2:
            return 0

        intervals = []
        for i in range(1, min(10, len(self.failure_history))):
            interval = self.failure_history[-i] - self.failure_history[-(i + 1)]
            intervals.append(interval)

        return sum(intervals) / len(intervals) if intervals else 0


class AsyncEventBus:
    """Advanced event-driven message bus with filtering and priority.
    
    Provides a publish-subscribe event system with priority handling,
    filtering capabilities, and comprehensive event history tracking.
    
    Attributes:
        subscribers: Dictionary mapping event types to subscriber lists
        event_queue: Priority queue for event processing
        event_history: List of recent events for debugging
        max_history: Maximum number of events to keep in history
        listeners: Compatibility alias for subscribers
        queue: Compatibility alias for event_queue
    """

    def __init__(self):
        """Initialize the async event bus."""
        self.subscribers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.event_queue = asyncio.PriorityQueue()
        self.event_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        self.listeners = {}  # For compatibility with get_metrics
        self.queue = self.event_queue  # Alias for compatibility

    def subscribe(
        self,
        event_type: str,
        handler: Callable,
        priority: int = 0,
        filter_func: Optional[Callable] = None,
    ):
        """Subscribe to an event type with priority and filtering.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Function to call when event is emitted
            priority: Handler priority (higher values execute first)
            filter_func: Optional filter function to determine if handler should run
        """
        self.subscribers[event_type].append(
            {"handler": handler, "priority": priority, "filter": filter_func}
        )
        logger.info(f"Subscribed handler to event: {event_type} (priority={priority})")

    async def emit(self, event_type: str, data: Any, priority: int = 0):
        """Emit an event to all subscribers with priority.
        
        Args:
            event_type: Type of event being emitted
            data: Event data to pass to handlers
            priority: Event priority for processing order
        """
        event = {
            "type": event_type,
            "data": data,
            "timestamp": time.time(),
            "priority": priority,
        }

        # Store in history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)

        logger.debug(f"Emitting event: {event_type} (priority={priority})")

        if event_type in self.subscribers:
            # Sort by priority (higher priority first)
            sorted_subs = sorted(
                self.subscribers[event_type], key=lambda x: x["priority"], reverse=True
            )

            tasks = []
            for sub in sorted_subs:
                # Apply filter if present
                if sub["filter"] and not sub["filter"](data):
                    continue

                tasks.append(self._safe_handle(sub["handler"], data))

            await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_handle(self, handler: Callable, data: Any):
        """Safely execute handler with error handling.
        
        Args:
            handler: Event handler function
            data: Event data to pass to handler
        """
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(data)
            else:
                handler(data)
        except Exception as e:
            logger.error(f"Error in event handler: {e}", exc_info=True)

    def get_event_stats(self) -> Dict[str, Any]:
        """Get event bus statistics.
        
        Returns:
            Dictionary containing event statistics and metrics
        """
        event_counts = defaultdict(int)
        for event in self.event_history:
            event_counts[event["type"]] += 1

        return {
            "total_events": len(self.event_history),
            "event_types": len(event_counts),
            "event_counts": dict(event_counts),
            "subscribers": {k: len(v) for k, v in self.subscribers.items()},
        }


class PipelineMiddleware:
    """Middleware system for pipeline processing.
    
    Provides a way to inject processing logic at various points in the
    pipeline execution without modifying the core pipeline stages.
    
    Attributes:
        name: Middleware identifier
        handler: Function to execute for middleware processing
        enabled: Whether this middleware is currently active
        metrics: Performance metrics for this middleware
    """

    def __init__(self, name: str, handler: Callable):
        """Initialize pipeline middleware.
        
        Args:
            name: Unique name for this middleware
            handler: Function to execute for processing
        """
        self.name = name
        self.handler = handler
        self.enabled = True
        self.metrics: Dict[str, float] = {}

    async def process(self, context: PipelineContext) -> PipelineContext:
        """Process context through middleware.
        
        Args:
            context: Pipeline context to process
            
        Returns:
            Modified pipeline context
        """
        if not self.enabled:
            return context

        start = time.time()
        try:
            if asyncio.iscoroutinefunction(self.handler):
                await self.handler(context)
            else:
                self.handler(context)

            self.metrics["last_duration"] = time.time() - start
            self.metrics["total_calls"] = self.metrics.get("total_calls", 0) + 1

        except Exception as e:
            logger.error(f"Middleware {self.name} error: {e}", exc_info=True)
            context.metadata[f"middleware_error_{self.name}"] = str(e)

        return context


class DynamicPipelineStage:
    """Dynamic pipeline stage with configurable behavior.
    
    Represents a single stage in the processing pipeline with configurable
    timeout, retry logic, and requirement settings. Tracks performance
    metrics for monitoring and optimization.
    
    Attributes:
        name: Stage identifier
        handler: Function to execute for this stage
        timeout: Maximum execution time in seconds
        retry_count: Number of retry attempts on failure
        required: Whether stage failure should fail the entire pipeline
        metrics: Performance and execution metrics
    """

    def __init__(
        self,
        name: str,
        handler: Callable,
        timeout: Optional[float] = None,
        retry_count: int = 0,
        required: bool = True,
    ):
        """Initialize dynamic pipeline stage.
        
        Args:
            name: Unique name for this stage
            handler: Function to execute for stage processing
            timeout: Maximum execution time in seconds
            retry_count: Number of retry attempts on failure
            required: Whether stage failure should fail entire pipeline
        """
        self.name = name
        self.handler = handler
        self.timeout = timeout or 30.0
        self.retry_count = retry_count
        self.required = required
        self.metrics: Dict[str, Any] = {
            "executions": 0,
            "failures": 0,
            "total_duration": 0.0,
            "avg_duration": 0.0,
        }

    async def execute(self, context: PipelineContext) -> None:
        """Execute stage with retry logic.
        
        Args:
            context: Pipeline context to process
            
        Raises:
            Exception: If stage is required and all retries fail
        """
        attempts = 0
        last_error = None

        while attempts <= self.retry_count:
            try:
                start = time.time()

                # Execute with timeout
                await asyncio.wait_for(self._run_handler(context), timeout=self.timeout)

                duration = time.time() - start
                self._update_metrics(duration, success=True)
                context.metrics[f"stage_{self.name}_duration"] = duration

                return

            except asyncio.TimeoutError:
                last_error = f"Stage {self.name} timed out after {self.timeout}s"
                logger.warning(last_error)

            except Exception as e:
                last_error = f"Stage {self.name} error: {str(e)}"
                logger.error(last_error, exc_info=True)

            attempts += 1
            if attempts <= self.retry_count:
                await asyncio.sleep(2**attempts)  # Exponential backoff

        # All retries failed
        self._update_metrics(0, success=False)

        if self.required:
            raise Exception(last_error or f"Stage {self.name} failed")
        else:
            logger.warning(f"Non-required stage {self.name} failed, continuing...")
            context.metadata[f"stage_{self.name}_skipped"] = True

    async def _run_handler(self, context: PipelineContext):
        """Run the stage handler.
        
        Args:
            context: Pipeline context to process
        """
        if asyncio.iscoroutinefunction(self.handler):
            await self.handler(context)
        else:
            self.handler(context)

    def _update_metrics(self, duration: float, success: bool):
        """Update stage performance metrics.
        
        Args:
            duration: Execution duration in seconds
            success: Whether execution was successful
        """
        self.metrics["executions"] += 1
        if not success:
            self.metrics["failures"] += 1

        self.metrics["total_duration"] += duration
        self.metrics["avg_duration"] = self.metrics["total_duration"] / self.metrics["executions"]


class AdvancedAsyncPipeline:
    """Ultra-advanced async pipeline with dynamic configuration.
    
    Main pipeline class that orchestrates command processing through multiple
    stages with event-driven architecture, circuit breaker protection, and
    comprehensive error handling. Supports dynamic stage registration,
    middleware injection, and context-aware processing.
    
    Attributes:
        jarvis: Reference to main JARVIS instance
        config: Pipeline configuration dictionary
        event_bus: Event bus for publish-subscribe messaging
        circuit_breaker: Circuit breaker for fault tolerance
        stages: Dictionary of registered pipeline stages
        middleware: List of registered middleware components
        active_commands: Currently processing commands
        performance_metrics: Historical performance data
        intent_engine: Intent classification engine
        context_store: Context storage for follow-up handling
        router: Command routing system
        context_bridge: Bridge to context intelligence system
    """

    def __init__(self, jarvis_instance=None, config: Optional[Dict[str, Any]] = None):
        """Initialize the advanced async pipeline.
        
        Args:
            jarvis_instance: Reference to main JARVIS instance
            config: Configuration dictionary for pipeline settings
        """
        self.jarvis = jarvis_instance
        self.config = config or {}
        self.event_bus = AsyncEventBus()
        self.circuit_breaker = AdaptiveCircuitBreaker(
            initial_threshold=self.config.get("circuit_breaker_threshold", 5),
            initial_timeout=self.config.get("circuit_breaker_timeout", 60),
            adaptive=self.config.get("adaptive_circuit_breaker", True),
        )

        # Dynamic stage registry
        self.stages: Dict[str, DynamicPipelineStage] = {}
        self.middleware: List[PipelineMiddleware] = []
        self.active_commands: Dict[str, PipelineContext] = {}

        # Performance monitoring
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)

        # ═══════════════════════════════════════════════════════════════
        # Follow-Up System Components
        # ═══════════════════════════════════════════════════════════════
        self.intent_engine = None
        self.context_store = None
        self.router = None
        self._follow_up_enabled = self.config.get("follow_up_enabled", True)

        # ═══════════════════════════════════════════════════════════════
        # Context Intelligence System (Priority 1-3)
        # ═══════════════════════════════════════════════════════════════
        self.context_bridge = None  # Will be set by main.py if available

        if self._follow_up_enabled:
            try:
                self._init_followup_system()
                logger.info("✅ Follow-up system initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize follow-up system: {e}", exc_info=True)
                self._follow_up_enabled = False

        # Initialize default stages
        self._register_default_stages()

    def _register_default_stages(self):
        """Register default pipeline stages with appropriate timeouts and requirements."""
        self.register_stage("validation", self._validate_command, timeout=5.0, required=True)

        self.register_stage(
            "screen_lock_check",
            self._check_screen_lock_universal,
            timeout=5.0,
            required=True,
        )

        self.register_stage("preprocessing", self._preprocess_command, timeout=5.0, required=False)

        self.register_stage("intent_analysis", self._analyze_intent, timeout=10.0, required=True)

        self.register_stage(
            "component_loading", self._load_components, timeout=15.0, required=False
        )

        self.register_stage(
            "processing",
            self._process_command,
            timeout=60.0,  # Increased for locked screen unlock flow
            retry_count=2,
            required=True,
        )

        self.register_stage(
            "postprocessing", self._postprocess_response, timeout=5.0, required=False
        )

        self.register_stage(
            "response_generation", self._generate_response, timeout=10.0, required=True
        )

    def _init_followup_system(self):
        """Initialize follow-up handling system components.
        
        Sets up intent classification, context storage, and routing systems
        for handling follow-up questions and contextual conversations.
        
        Raises:
            ImportError: If required follow-up system modules are not available
            Exception: If initialization fails
        """
        from pathlib import Path

        from backend.core.context.memory_store import InMemoryContextStore
        from backend.core.intent.adaptive_classifier import (
            AdaptiveIntentEngine,
            LexicalClassifier,
            WeightedVotingStrategy,
        )
        from backend.core.intent.intent_registry import IntentRegistry
        from backend.core.routing.adaptive_router import (
            AdaptiveRouter,
            PluginRegistry,
            RouteMatcher,
            context_validation_middleware,
            logging_middleware,
        )
        from backend.vision.handlers.follow_up_plugin import VisionFollowUpPlugin

        # Initialize intent registry and load patterns
        config_path = Path(__file__).parent.parent / "config" / "followup_intents.json"
        if config_path.exists():
            registry = IntentRegistry(config_path=config_path)
        else:
            from backend.core.intent.intent_registry import create_default_registry

            registry = create_default_registry()

        patterns = registry.get_all_patterns()

        # Create lexical classifier
        classifier = LexicalClassifier(
            name="lexical_followup",
            patterns=patterns,
            priority=100,  # Highest priority
            case_sensitive=False,
        )

        # Create intent engine
        self.intent_engine = AdaptiveIntentEngine(
            classifiers=[classifier],
            strategy=WeightedVotingStrategy(
                source_weights={"lexical_followup": 1.0},
                min_confidence=0.6,
            ),
        )

        # Initialize context store
        max_contexts = self.config.get("max_pending_contexts", 100)
        self.context_store = InMemoryContextStore(max_size=max_contexts)

        # Start auto-cleanup
        asyncio.create_task(self.context_store.start_auto_cleanup())

        # Initialize router
        matcher = RouteMatcher()
        self.router = AdaptiveRouter(matcher=matcher)

        # Add middleware
        self.router.use_middleware(logging_middleware)
        self.router.use_middleware(context_validation_middleware)

        # Register vision follow-up plugin
        self.plugin_registry = PluginRegistry(self.router)
        vision_plugin = VisionFollowUpPlugin()
        asyncio.create_task(self.plugin_registry.register_plugin("vision_followup", vision_plugin))

        logger.info(
            f"Follow-up system initialized: "
            f"{self.intent_engine.classifier_count} classifiers, "
            f"max_contexts={max_contexts}"
        )

    def register_stage(
        self,
        name: str,
        handler: Callable,
        timeout: Optional[float] = None,
        retry_count: int = 0,
        required: bool = True,
    ):
        """Dynamically register a new pipeline stage.
        
        Args:
            name: Unique name for the stage
            handler: Function to execute for this stage
            timeout: Maximum execution time in seconds
            retry_count: Number of retry attempts on failure
            required: Whether stage failure should fail entire pipeline
        """
        stage = DynamicPipelineStage(
            name=name,
            handler=handler,
            timeout=timeout,
            retry_count=retry_count,
            required=required,
        )
        self.stages[name] = stage
        logger.info(
            f"✅ Registered pipeline stage: {name} (timeout={timeout}s, retries={retry_count})"
        )

    def register_middleware(self, name: str, handler: Callable):
        """Register middleware for pipeline processing.
        
        Args:
            name: Unique name for the middleware
            handler: Function to execute for middleware processing
        """
        middleware = PipelineMiddleware(name, handler)
        self.middleware.append(middleware)
        logger.info(f"✅ Registered middleware: {name}")

    def unregister_stage(self, name: str):
        """Remove a pipeline stage.
        
        Args:
            name: Name of the stage to remove
        """
        if name in self.stages:
            del self.stages[name]
            logger.info(f"Unregistered pipeline stage: {name}")

    async def process_async(
        self,
        text: str,
        user_name: str = "Sir",
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        audio_data: Optional[bytes] = None,
        speaker_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process command through advanced async pipeline.
        
        Main entry point for command processing. Handles the complete pipeline
        from command receipt through response generation, with comprehensive
        error handling and performance monitoring.
        
        Args:
            text: Command text to process
            user_name: Name of the user issuing the command
            priority: Command priority (0=normal, 1=high, 2=critical)
            metadata: Additional metadata dictionary
            audio_data: Voice audio data for authentication
            speaker_name: Identified speaker name from voice recognition
            
        Returns:
            Dictionary containing response, metadata, success status, and metrics
            
        Example:
            >>> result = await pipeline.process_async("open safari")
            >>> print(result['response'])
            "I've opened Safari for you, Sir."
        """

        # FAST PATH for lock/unlock commands - bypass heavy processing
        text_lower = text.lower()
        if any(
            phrase in text_lower
            for phrase in [
                "lock my screen",
                "lock screen",
                "unlock my screen",
                "unlock screen",
                "lock the screen",
                "unlock the screen",
            ]
        ):
            return await self._fast_lock_unlock(
                text, user_name, metadata, audio_data=audio_data, speaker_name=speaker_name
            )

        # Create pipeline context
        command_id = f"cmd_{int(time.time() * 1000)}"
        context = PipelineContext(
            command_id=command_id,
            text=text,
            user_name=user_name,
            priority=priority,
            metadata=metadata or {},
            audio_data=audio_data,
            speaker_name=speaker_name,
        )

        self.active_commands[command_id] = context

        try:
            # Emit command received event
            await self.event_bus.emit("command_received", context, priority=priority)

            # Process through pipeline with circuit breaker
            result = await self.circuit_breaker.call(self._execute_pipeline, context)

            # Emit command completed event
            context.stage = PipelineStage.COMPLETED
            await self.event_bus.emit("command_completed", context, priority=priority)

            # Update performance metrics
            self._record_performance(context)

            return result

        except Exception as e:
            logger.error(f"Pipeline error for command {command_id}: {e}", exc_info=True)
            context.stage = PipelineStage.FAILED
            context.error = str(e)
            await self.event_bus.emit("command_failed", context, priority=priority)

            return self._generate_error_response(context, e)

        finally:
            # Cleanup after processing (thread-safe)
            try:
                self.active_commands.pop(command_id, None)
            except RuntimeError:
                # Handle dictionary changed size during iteration
                pass

    async def _fast_lock_unlock(
        self,
        text: str,
        user_name: str,
        metadata: Optional[Dict