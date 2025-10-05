#!/usr/bin/env python3
"""
Advanced Async Architecture - Dynamic Event-Driven Command Pipeline
Ultra-robust, adaptive, zero-hardcoding async processing system
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time
import inspect
from functools import wraps
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Dynamic pipeline processing stages"""
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
    """Context passed through the pipeline"""
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
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
            "priority": self.priority
        }


class AdaptiveCircuitBreaker:
    """Advanced circuit breaker with adaptive thresholds and ML-based prediction"""

    def __init__(self,
                 initial_threshold: int = 5,
                 initial_timeout: int = 60,
                 adaptive: bool = True):
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
        """Calculate current success rate"""
        if self._total_calls == 0:
            return 1.0  # Default to 100% success if no calls yet
        return self._successful_calls / self._total_calls

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with adaptive circuit breaker protection"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                logger.info(f"Circuit breaker: transitioning to HALF_OPEN (threshold={self.threshold})")
                self.state = "HALF_OPEN"
            else:
                raise Exception(f"Circuit breaker is OPEN - service unavailable (retry in {int(self.timeout - (time.time() - self.last_failure_time))}s)")

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
        """Handle successful execution with adaptive learning"""
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
        """Handle failed execution with adaptive learning"""
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
            logger.warning(f"Circuit breaker: OPEN (failures: {self.failure_count}/{self.threshold})")
            self.state = "OPEN"

            # Adaptive timeout based on failure patterns
            if self.adaptive:
                avg_failure_interval = self._calculate_failure_interval()
                if avg_failure_interval > 0:
                    self.timeout = min(300, int(avg_failure_interval * 2))  # Max 5 min
                    logger.info(f"Adaptive timeout set to {self.timeout}s")

    def _calculate_failure_interval(self) -> float:
        """Calculate average interval between failures"""
        if len(self.failure_history) < 2:
            return 0

        intervals = []
        for i in range(1, min(10, len(self.failure_history))):
            interval = self.failure_history[-i] - self.failure_history[-(i+1)]
            intervals.append(interval)

        return sum(intervals) / len(intervals) if intervals else 0


class AsyncEventBus:
    """Advanced event-driven message bus with filtering and priority"""

    def __init__(self):
        self.subscribers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.event_queue = asyncio.PriorityQueue()
        self.event_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        self.listeners = {}  # For compatibility with get_metrics
        self.queue = self.event_queue  # Alias for compatibility

    def subscribe(self,
                  event_type: str,
                  handler: Callable,
                  priority: int = 0,
                  filter_func: Optional[Callable] = None):
        """Subscribe to an event type with priority and filtering"""
        self.subscribers[event_type].append({
            "handler": handler,
            "priority": priority,
            "filter": filter_func
        })
        logger.info(f"Subscribed handler to event: {event_type} (priority={priority})")

    async def emit(self, event_type: str, data: Any, priority: int = 0):
        """Emit an event to all subscribers with priority"""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": time.time(),
            "priority": priority
        }

        # Store in history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)

        logger.debug(f"Emitting event: {event_type} (priority={priority})")

        if event_type in self.subscribers:
            # Sort by priority (higher priority first)
            sorted_subs = sorted(
                self.subscribers[event_type],
                key=lambda x: x["priority"],
                reverse=True
            )

            tasks = []
            for sub in sorted_subs:
                # Apply filter if present
                if sub["filter"] and not sub["filter"](data):
                    continue

                tasks.append(self._safe_handle(sub["handler"], data))

            await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_handle(self, handler: Callable, data: Any):
        """Safely execute handler with error handling"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(data)
            else:
                handler(data)
        except Exception as e:
            logger.error(f"Error in event handler: {e}", exc_info=True)

    def get_event_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        event_counts = defaultdict(int)
        for event in self.event_history:
            event_counts[event["type"]] += 1

        return {
            "total_events": len(self.event_history),
            "event_types": len(event_counts),
            "event_counts": dict(event_counts),
            "subscribers": {k: len(v) for k, v in self.subscribers.items()}
        }


class PipelineMiddleware:
    """Middleware system for pipeline processing"""

    def __init__(self, name: str, handler: Callable):
        self.name = name
        self.handler = handler
        self.enabled = True
        self.metrics: Dict[str, float] = {}

    async def process(self, context: PipelineContext) -> PipelineContext:
        """Process context through middleware"""
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
    """Dynamic pipeline stage with configurable behavior"""

    def __init__(self,
                 name: str,
                 handler: Callable,
                 timeout: Optional[float] = None,
                 retry_count: int = 0,
                 required: bool = True):
        self.name = name
        self.handler = handler
        self.timeout = timeout or 30.0
        self.retry_count = retry_count
        self.required = required
        self.metrics: Dict[str, Any] = {
            "executions": 0,
            "failures": 0,
            "total_duration": 0.0,
            "avg_duration": 0.0
        }

    async def execute(self, context: PipelineContext) -> None:
        """Execute stage with retry logic"""
        attempts = 0
        last_error = None

        while attempts <= self.retry_count:
            try:
                start = time.time()

                # Execute with timeout
                await asyncio.wait_for(
                    self._run_handler(context),
                    timeout=self.timeout
                )

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
                await asyncio.sleep(2 ** attempts)  # Exponential backoff

        # All retries failed
        self._update_metrics(0, success=False)

        if self.required:
            raise Exception(last_error or f"Stage {self.name} failed")
        else:
            logger.warning(f"Non-required stage {self.name} failed, continuing...")
            context.metadata[f"stage_{self.name}_skipped"] = True

    async def _run_handler(self, context: PipelineContext):
        """Run the stage handler"""
        if asyncio.iscoroutinefunction(self.handler):
            await self.handler(context)
        else:
            self.handler(context)

    def _update_metrics(self, duration: float, success: bool):
        """Update stage metrics"""
        self.metrics["executions"] += 1
        if not success:
            self.metrics["failures"] += 1

        self.metrics["total_duration"] += duration
        self.metrics["avg_duration"] = (
            self.metrics["total_duration"] / self.metrics["executions"]
        )


class AdvancedAsyncPipeline:
    """Ultra-advanced async pipeline with dynamic configuration"""

    def __init__(self, jarvis_instance=None, config: Optional[Dict[str, Any]] = None):
        self.jarvis = jarvis_instance
        self.config = config or {}
        self.event_bus = AsyncEventBus()
        self.circuit_breaker = AdaptiveCircuitBreaker(
            initial_threshold=self.config.get("circuit_breaker_threshold", 5),
            initial_timeout=self.config.get("circuit_breaker_timeout", 60),
            adaptive=self.config.get("adaptive_circuit_breaker", True)
        )

        # Dynamic stage registry
        self.stages: Dict[str, DynamicPipelineStage] = {}
        self.middleware: List[PipelineMiddleware] = []
        self.active_commands: Dict[str, PipelineContext] = {}

        # Performance monitoring
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)

        # Initialize default stages
        self._register_default_stages()

    def _register_default_stages(self):
        """Register default pipeline stages"""
        self.register_stage(
            "validation",
            self._validate_command,
            timeout=5.0,
            required=True
        )

        self.register_stage(
            "preprocessing",
            self._preprocess_command,
            timeout=5.0,
            required=False
        )

        self.register_stage(
            "intent_analysis",
            self._analyze_intent,
            timeout=10.0,
            required=True
        )

        self.register_stage(
            "component_loading",
            self._load_components,
            timeout=15.0,
            required=False
        )

        self.register_stage(
            "processing",
            self._process_command,
            timeout=30.0,
            retry_count=2,
            required=True
        )

        self.register_stage(
            "postprocessing",
            self._postprocess_response,
            timeout=5.0,
            required=False
        )

        self.register_stage(
            "response_generation",
            self._generate_response,
            timeout=10.0,
            required=True
        )

    def register_stage(self,
                      name: str,
                      handler: Callable,
                      timeout: Optional[float] = None,
                      retry_count: int = 0,
                      required: bool = True):
        """Dynamically register a new pipeline stage"""
        stage = DynamicPipelineStage(
            name=name,
            handler=handler,
            timeout=timeout,
            retry_count=retry_count,
            required=required
        )
        self.stages[name] = stage
        logger.info(f"✅ Registered pipeline stage: {name} (timeout={timeout}s, retries={retry_count})")

    def register_middleware(self, name: str, handler: Callable):
        """Register middleware for pipeline processing"""
        middleware = PipelineMiddleware(name, handler)
        self.middleware.append(middleware)
        logger.info(f"✅ Registered middleware: {name}")

    def unregister_stage(self, name: str):
        """Remove a pipeline stage"""
        if name in self.stages:
            del self.stages[name]
            logger.info(f"Unregistered pipeline stage: {name}")

    async def process_async(self,
                           text: str,
                           user_name: str = "Sir",
                           priority: int = 0,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process command through advanced async pipeline"""

        # Create pipeline context
        command_id = f"cmd_{int(time.time() * 1000)}"
        context = PipelineContext(
            command_id=command_id,
            text=text,
            user_name=user_name,
            priority=priority,
            metadata=metadata or {}
        )

        self.active_commands[command_id] = context

        try:
            # Emit command received event
            await self.event_bus.emit("command_received", context, priority=priority)

            # Process through pipeline with circuit breaker
            result = await self.circuit_breaker.call(
                self._execute_pipeline,
                context
            )

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

    async def _execute_pipeline(self, context: PipelineContext) -> Dict[str, Any]:
        """Execute pipeline stages (only specific stage if specified in metadata)"""
        pipeline_start = time.time()

        # Execute middleware (preprocessing)
        for mw in self.middleware:
            context = await mw.process(context)

        # Check if a specific stage is requested
        specific_stage = context.metadata.get("stage")

        if specific_stage:
            # Execute only the specific stage requested
            if specific_stage in self.stages:
                stage = self.stages[specific_stage]
                context.stage = PipelineStage[specific_stage.upper()] if specific_stage.upper() in PipelineStage.__members__ else PipelineStage.PROCESSING

                await self.event_bus.emit(f"stage_{specific_stage}", context)

                try:
                    await stage.execute(context)
                except Exception as e:
                    if stage.required:
                        raise
                    else:
                        logger.warning(f"Stage {specific_stage} failed: {e}")
            else:
                logger.warning(f"Requested stage '{specific_stage}' not found")
        else:
            # Execute default pipeline stages (not custom registered ones like applescript_execution)
            default_stages = ["validation", "preprocessing", "intent_analysis",
                             "component_loading", "processing", "postprocessing",
                             "response_generation"]

            for stage_name in default_stages:
                if stage_name in self.stages:
                    stage = self.stages[stage_name]
                    context.stage = PipelineStage[stage_name.upper()] if stage_name.upper() in PipelineStage.__members__ else PipelineStage.PROCESSING

                    await self.event_bus.emit(f"stage_{stage_name}", context)

                    try:
                        await stage.execute(context)
                    except Exception as e:
                        if stage.required:
                            raise
                        else:
                            logger.warning(f"Non-critical stage {stage_name} failed: {e}")

        context.metrics["total_pipeline_duration"] = time.time() - pipeline_start

        # Return dictionary containing both response and metadata
        return {
            "response": context.response or "Task completed successfully.",
            "metadata": context.metadata,
            "success": True,
            "command_id": context.command_id,
            "metrics": context.metrics
        }

    async def _validate_command(self, context: PipelineContext):
        """Validate command input"""
        if not context.text or len(context.text.strip()) == 0:
            raise ValueError("Empty command received")

        if len(context.text) > 10000:
            raise ValueError("Command too long (max 10000 chars)")

        context.metadata["validated"] = True

    async def _preprocess_command(self, context: PipelineContext):
        """Preprocess command text"""
        # Normalize text
        context.text = context.text.strip()

        # Detect language (if needed)
        context.metadata["language"] = "en"  # Default

        # Extract entities (if needed)
        context.metadata["preprocessed"] = True

    async def _analyze_intent(self, context: PipelineContext):
        """Analyze command intent dynamically"""
        text_lower = context.text.lower()

        # Intent detection rules (extensible)
        intent_rules = {
            "monitoring": ["monitor", "watch", "track", "observe"],
            "system_control": ["open", "launch", "start", "close", "quit"],
            "document_creation": ["write", "create", "document", "draft"],
            "weather": ["weather", "temperature", "forecast", "rain"],
            "time": ["time", "date", "clock", "when"],
            "conversation": []  # Default
        }

        for intent, keywords in intent_rules.items():
            if any(kw in text_lower for kw in keywords):
                context.intent = intent
                logger.info(f"Intent detected: {intent} for command: {context.text}")
                return

        context.intent = "conversation"

    async def _load_components(self, context: PipelineContext):
        """Load required components dynamically"""
        if context.intent == "monitoring":
            context.components_loaded.append("vision")
        elif context.intent == "document_creation":
            context.components_loaded.append("document_writer")
        elif context.intent == "weather":
            context.components_loaded.append("weather_system")

        logger.info(f"Components loaded: {context.components_loaded}")

    async def _process_command(self, context: PipelineContext):
        """Process command based on intent"""
        # Check for lock/unlock commands first - these can work without JARVIS
        text_lower = context.text.lower()
        if any(phrase in text_lower for phrase in ['lock my screen', 'lock screen', 'unlock my screen', 'unlock screen', 'lock the screen', 'unlock the screen']):
            # Handle lock/unlock directly through simple handler
            logger.info(f"[PIPELINE] Processing lock/unlock command: {context.text}")
            try:
                from api.simple_unlock_handler import handle_unlock_command
                result = await handle_unlock_command(context.text, self.jarvis)

                logger.info(f"[PIPELINE] Lock/unlock result: success={result.get('success')}, action={result.get('action', 'unknown')}")

                if result.get('success'):
                    context.response = result.get('response', 'Command executed successfully.')
                    context.metadata["lock_unlock_result"] = result
                else:
                    context.response = result.get('response', 'Command failed.')
                    context.metadata["lock_unlock_error"] = result.get('error', 'Unknown error')

                context.metadata["handled_by"] = "simple_unlock_handler"
                logger.info(f"[PIPELINE] Lock/unlock response: {context.response}")
                return
            except Exception as e:
                logger.error(f"Error handling lock/unlock: {e}")
                context.metadata["lock_unlock_error"] = str(e)
                context.response = f"I couldn't execute that screen command: {str(e)}"
                return

        # For other commands, check if JARVIS is available
        if not self.jarvis:
            # Some commands don't require JARVIS - handle them here
            if context.intent == "system_control":
                # Try to handle basic system commands without JARVIS
                context.metadata["handled_without_jarvis"] = True
                context.response = "System command received but JARVIS is not fully initialized."
            else:
                context.metadata["warning"] = "No JARVIS instance available"
            return

        # Route to appropriate handler based on intent
        if context.intent == "conversation" and hasattr(self.jarvis, 'claude_chatbot'):
            try:
                response = await self.jarvis.claude_chatbot.generate_response(context.text)
                context.metadata["claude_response"] = response
            except Exception as e:
                logger.error(f"Error in Claude chatbot: {e}")
                context.metadata["claude_error"] = str(e)

        elif context.intent == "system_control" and hasattr(self.jarvis, '_handle_system_command'):
            try:
                response = await self.jarvis._handle_system_command(context.text)
                context.metadata["system_response"] = response
            except Exception as e:
                logger.error(f"Error in system control: {e}")
                context.metadata["system_error"] = str(e)

    async def _postprocess_response(self, context: PipelineContext):
        """Postprocess response before delivery"""
        # Add personalization
        if context.response:
            context.response = context.response.strip()

        context.metadata["postprocessed"] = True

    async def _generate_response(self, context: PipelineContext):
        """Generate final response from context"""
        # If response was already set (e.g., by lock/unlock handler), use it
        if context.response:
            logger.info(f"Using pre-set response: {context.response[:100]}...")
            return

        # Prioritize responses from metadata
        if "claude_response" in context.metadata:
            context.response = context.metadata["claude_response"]
        elif "system_response" in context.metadata:
            context.response = context.metadata["system_response"]
        elif "lock_unlock_result" in context.metadata:
            # Response already set in _process_command for lock/unlock
            pass
        elif context.metadata.get("warning"):
            context.response = f"Warning: {context.metadata['warning']}"
        else:
            context.response = f"I processed your command: '{context.text}', {context.user_name}."

        logger.info(f"Response generated: {context.response[:100]}...")

    def _generate_error_response(self, context: PipelineContext, error: Exception) -> Dict[str, Any]:
        """Generate user-friendly error response"""
        error_responses = {
            "TimeoutError": f"I apologize, {context.user_name}, but that's taking longer than expected. Please try again.",
            "ValueError": f"I'm sorry, {context.user_name}, but I couldn't understand that command.",
            "ConnectionError": f"I'm having trouble connecting to my services, {context.user_name}. Please check your connection.",
            "default": f"I apologize, {context.user_name}, but I encountered an error: {str(error)}"
        }

        error_type = type(error).__name__
        error_message = error_responses.get(error_type, error_responses["default"])

        return {
            "response": error_message,
            "metadata": context.metadata,
            "success": False,
            "command_id": context.command_id,
            "error": str(error),
            "error_type": error_type,
            "metrics": context.metrics
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get current pipeline metrics and statistics"""
        metrics = {
            "circuit_breaker": {
                "state": self.circuit_breaker.state,
                "failure_count": self.circuit_breaker.failure_count,
                "threshold": self.circuit_breaker.threshold,
                "success_rate": self.circuit_breaker.success_rate
            },
            "event_bus": {
                "listeners": len(self.event_bus.listeners),
                "queue_size": self.event_bus.queue.qsize() if hasattr(self.event_bus, 'queue') else 0
            },
            "stages": {
                name: {
                    "timeout": stage.timeout,
                    "retry_count": stage.retry_count,
                    "required": stage.required
                } for name, stage in self.stages.items()
            },
            "performance": {
                "average_processing_time": self._calculate_average_processing_time(),
                "total_commands_processed": self._total_commands_processed
            }
        }
        return metrics

    def _calculate_average_processing_time(self) -> float:
        """Calculate average processing time from recent commands"""
        if not hasattr(self, '_processing_times'):
            return 0.0
        if len(self._processing_times) == 0:
            return 0.0
        return sum(self._processing_times) / len(self._processing_times)

    def _record_performance(self, context: PipelineContext):
        """Record performance metrics"""
        if not hasattr(self, '_processing_times'):
            self._processing_times = []
        if not hasattr(self, '_total_commands_processed'):
            self._total_commands_processed = 0

        # Record processing time
        if 'total_time' in context.metrics:
            self._processing_times.append(context.metrics['total_time'])
            # Keep only last 100 processing times
            if len(self._processing_times) > 100:
                self._processing_times.pop(0)

        self._total_commands_processed += 1
        total_duration = context.metrics.get("total_pipeline_duration", 0)
        self.performance_metrics["total_duration"].append(total_duration)
        self.performance_metrics["intents"].append(context.intent or "unknown")

        # Keep only last 1000 metrics
        for key in self.performance_metrics:
            if len(self.performance_metrics[key]) > 1000:
                self.performance_metrics[key] = self.performance_metrics[key][-1000:]

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        total_durations = self.performance_metrics.get("total_duration", [])

        return {
            "total_commands": len(total_durations),
            "avg_duration": sum(total_durations) / len(total_durations) if total_durations else 0,
            "min_duration": min(total_durations) if total_durations else 0,
            "max_duration": max(total_durations) if total_durations else 0,
            "stages": {name: stage.metrics for name, stage in self.stages.items()},
            "middleware": {mw.name: mw.metrics for mw in self.middleware},
            "circuit_breaker": {
                "state": self.circuit_breaker.state,
                "threshold": self.circuit_breaker.threshold,
                "failures": self.circuit_breaker.failure_count,
                "successes": self.circuit_breaker.success_count
            },
            "event_bus": self.event_bus.get_event_stats(),
            "active_commands": len(self.active_commands)
        }


# Global pipeline instance
_pipeline_instance = None


def get_async_pipeline(jarvis_instance=None, config: Optional[Dict[str, Any]] = None) -> AdvancedAsyncPipeline:
    """Get or create the global async pipeline"""
    global _pipeline_instance

    if _pipeline_instance is None:
        _pipeline_instance = AdvancedAsyncPipeline(jarvis_instance, config)
        logger.info("✅ Advanced Async Command Pipeline initialized")

    return _pipeline_instance


def async_stage(name: str, timeout: float = 30.0, retry_count: int = 0, required: bool = True):
    """Decorator to register a function as a pipeline stage"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        # Register with global pipeline
        if _pipeline_instance:
            _pipeline_instance.register_stage(name, wrapper, timeout, retry_count, required)

        return wrapper
    return decorator


def async_middleware(name: str):
    """Decorator to register a function as middleware"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        # Register with global pipeline
        if _pipeline_instance:
            _pipeline_instance.register_middleware(name, wrapper)

        return wrapper
    return decorator
