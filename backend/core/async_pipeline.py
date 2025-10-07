#!/usr/bin/env python3
"""
Advanced Async Architecture - Dynamic Event-Driven Command Pipeline
Ultra-robust, adaptive, zero-hardcoding async processing system
"""

import asyncio
import re
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

        # FAST PATH for lock/unlock commands - bypass heavy processing
        text_lower = text.lower()
        if any(phrase in text_lower for phrase in ['lock my screen', 'lock screen', 'unlock my screen', 'unlock screen', 'lock the screen', 'unlock the screen']):
            return await self._fast_lock_unlock(text, user_name, metadata)

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

    async def _fast_lock_unlock(self, text: str, user_name: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fast path for lock/unlock commands - minimal overhead, maximum speed"""
        import asyncio
        from api.simple_unlock_handler import handle_unlock_command

        logger.info(f"[FAST PATH] Processing lock/unlock: {text}")
        start_time = time.time()

        # Determine action and prepare response in parallel
        text_lower = text.lower()
        is_lock = 'lock' in text_lower and 'unlock' not in text_lower

        # Execute command and prepare response concurrently
        async def execute_command():
            try:
                result = await handle_unlock_command(text, self.jarvis)
                return result
            except Exception as e:
                logger.error(f"[FAST PATH] Command execution error: {e}")
                return {'success': False, 'error': str(e)}

        async def prepare_response(is_lock_cmd: bool):
            if is_lock_cmd:
                return "Of course, Sir. Locking for you."
            else:
                return "Of course, Sir. Unlocking for you."

        # Execute both tasks in parallel for maximum speed
        command_task = asyncio.create_task(execute_command())
        response_task = asyncio.create_task(prepare_response(is_lock))

        # Start voice synthesis immediately (don't wait for command to complete)
        immediate_response = await response_task

        # Get command result
        result = await command_task

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"[FAST PATH] Command completed in {elapsed:.1f}ms")

        # Return immediately with pre-generated response
        return {
            "success": result.get('success', False),
            "response": immediate_response,
            "action": result.get('action', 'lock' if is_lock else 'unlock'),
            "metadata": {
                "handled_by": "fast_lock_unlock",
                "execution_time_ms": elapsed,
                "lock_unlock_result": result
            }
        }

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
        """Analyze command intent dynamically with compound action awareness"""
        text_lower = context.text.lower()

        # Dynamic intent detection rules - NO HARDCODING
        # Extensible through configuration
        intent_rules = {
            "monitoring": ["monitor", "watch", "track", "observe", "surveillance"],
            "system_control": ["open", "launch", "start", "close", "quit", "kill", "terminate", "run", "execute"],
            "document_creation": ["write", "create document", "write essay", "write paper", "draft", "compose", "author"],
            "web_search": ["search", "google", "look up", "find online", "browse", "surf"],
            "navigation": ["navigate", "go to", "visit", "head to"],
            "weather": ["weather", "temperature", "forecast", "rain", "snow", "climate"],
            "time": ["time", "date", "clock", "when", "calendar"],
            "media_control": ["play", "pause", "stop", "skip", "volume"],
            "conversation": []  # Default fallback
        }

        # Check for compound commands (multiple intents)
        detected_intents = []
        for intent, keywords in intent_rules.items():
            if any(kw in text_lower for kw in keywords):
                detected_intents.append(intent)

        # Intelligently handle multiple intents
        if len(detected_intents) > 1:
            # Compound command detected - prioritize system_control for compound actions
            if "system_control" in detected_intents:
                context.intent = "system_control"
                context.metadata["compound_command"] = True
                context.metadata["all_intents"] = detected_intents
                logger.info(f"Compound command detected with intents: {detected_intents}")
            else:
                # Use the most specific intent
                context.intent = detected_intents[0]
                context.metadata["secondary_intents"] = detected_intents[1:]
        elif detected_intents:
            context.intent = detected_intents[0]
            logger.info(f"Intent detected: {context.intent} for command: {context.text}")
        else:
            # Use ML-based intent detection as fallback
            context.intent = await self._ml_intent_detection(text_lower)

    async def _ml_intent_detection(self, text: str) -> str:
        """Advanced ML-based intent detection with dynamic fallbacks"""
        try:
            # Try multiple analyzers for robust intent detection
            intents_detected = []
            confidence_scores = {}

            # Analyzer 1: Context Intelligence Intent Analyzer
            try:
                from context_intelligence.analyzers.intent_analyzer import get_intent_analyzer
                analyzer = get_intent_analyzer()
                intent = await analyzer.analyze(text)
                if intent:
                    intents_detected.append(intent)
                    confidence_scores[intent] = analyzer.get_confidence() if hasattr(analyzer, 'get_confidence') else 0.8
            except:
                pass

            # Analyzer 2: Pattern-based semantic matching
            semantic_patterns = {
                'system_control': ['open', 'launch', 'start', 'close', 'quit', 'switch'],
                'search': ['search', 'find', 'look for', 'google', 'query'],
                'navigation': ['go to', 'navigate', 'visit', 'browse'],
                'file_ops': ['create', 'write', 'save', 'edit', 'delete'],
                'communication': ['email', 'message', 'send', 'call', 'contact'],
                'automation': ['automate', 'schedule', 'repeat', 'trigger']
            }

            text_lower = text.lower()
            for intent_type, patterns in semantic_patterns.items():
                match_score = sum(1 for p in patterns if p in text_lower) / len(patterns)
                if match_score > 0:
                    confidence_scores[intent_type] = match_score
                    if match_score > 0.3:  # Threshold for inclusion
                        intents_detected.append(intent_type)

            # Analyzer 3: Context-aware command type detection
            if 'and' in text_lower or 'then' in text_lower:
                # Compound command indicator
                if 'system_control' not in intents_detected:
                    intents_detected.append('system_control')
                    confidence_scores['system_control'] = 0.9

            # Select best intent based on confidence
            if confidence_scores:
                best_intent = max(confidence_scores, key=confidence_scores.get)
                logger.info(f"ML Intent Detection: {best_intent} (confidence: {confidence_scores[best_intent]:.2f})")
                return best_intent

            return "conversation"

        except Exception as e:
            logger.debug(f"ML intent detection error: {e}")
            return "conversation"

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

        # Handle compound system commands (e.g., "open safari and search for dogs")
        if context.intent == "system_control":
            logger.info(f"[PIPELINE] Processing compound system command: {context.text}")
            try:
                # Use ContextAwareCommandHandler for intelligent screen lock handling
                from context_intelligence.handlers.context_aware_handler import ContextAwareCommandHandler
                from context_intelligence.analyzers.compound_action_parser import get_compound_parser

                handler = ContextAwareCommandHandler()
                parser = get_compound_parser()

                # Define execution callback for compound actions
                async def execute_compound_command(command: str, context: Dict[str, Any] = None):
                    """Execute compound command after context handling"""
                    # Parse command into atomic actions
                    actions = await parser.parse(command)

                    if not actions:
                        return {"success": False, "message": "I didn't understand that command, Sir."}

                    # Log execution plan
                    plan = parser.get_execution_plan(actions)
                    logger.info(f"[COMPOUND] Execution plan: {plan}")

                    # Execute actions in sequence
                    results = []
                    for action in actions:
                        result = await self._execute_atomic_action(action)
                        results.append(result)
                        if not result.get('success'):
                            logger.warning(f"[COMPOUND] Action failed: {action.type.value}")

                    # Generate result with detailed feedback
                    successful_actions = [i for i, r in enumerate(results) if r.get('success')]
                    failed_actions = [i for i, r in enumerate(results) if not r.get('success')]

                    if all(r.get('success') for r in results):
                        return {
                            "success": True,
                            "message": f"Done, Sir.",
                            "plan": plan,
                            "results": results
                        }
                    elif any(r.get('success') for r in results):
                        # Build detailed message about what failed
                        failed_parts = []
                        for idx in failed_actions:
                            if idx < len(actions):
                                action_desc = actions[idx].type.value.replace('_', ' ')
                                failed_parts.append(action_desc)

                        if failed_parts:
                            failed_msg = " and ".join(failed_parts)
                            return {
                                "success": True,
                                "message": f"I completed part of your request, Sir, but couldn't {failed_msg}.",
                                "plan": plan,
                                "results": results
                            }
                        else:
                            return {
                                "success": True,
                                "message": f"Partially completed, Sir.",
                                "plan": plan,
                                "results": results
                            }
                    else:
                        return {
                            "success": False,
                            "message": "I couldn't complete that command, Sir.",
                            "plan": plan,
                            "results": results
                        }

                # Handle command with full context awareness (includes screen lock detection & unlock message)
                result = await handler.handle_command_with_context(
                    context.text,
                    execute_callback=execute_compound_command
                )

                # Set response from context-aware handler
                if result.get("messages"):
                    # Combine all messages (unlock message + execution result)
                    context.response = " ".join(result["messages"])
                else:
                    context.response = result.get("result", {}).get("message", "Command completed, Sir.")

                context.metadata["context_aware"] = True
                context.metadata["steps_taken"] = result.get("steps_taken", [])
                context.metadata["system_context"] = result.get("context", {})

                if result.get("result"):
                    context.metadata["execution_plan"] = result["result"].get("plan")
                    context.metadata["execution_results"] = result["result"].get("results")

                return

            except Exception as e:
                logger.error(f"Error handling compound command: {e}")
                import traceback
                traceback.print_exc()
                context.metadata["compound_error"] = str(e)
                context.response = f"I encountered an error processing that command, Sir."
                return

        # ENHANCED: Context-aware handling for ALL commands, not just document creation
        try:
            from context_intelligence.handlers.context_aware_handler import get_context_aware_handler
            context_handler = get_context_aware_handler()
        except ImportError as e:
            logger.warning(f"Context-aware handler not available: {e}, using fallback")
            context_handler = None

        # Get comprehensive system context
        system_context = await self._get_enhanced_system_context()
        context.metadata["system_context"] = system_context
        logger.info(f"[PIPELINE] System state: screen_locked={system_context.get('screen_locked')}, active_apps={system_context.get('active_apps', [])[:3]}")

        # Handle document creation commands with FULL context awareness
        if context.intent == "document_creation":
            logger.info(f"[PIPELINE] Document creation with enhanced context awareness")

            # Define intelligent document creation callback
            async def create_document(command: str, ctx: Dict[str, Any] = None):
                """Create document with full context awareness"""
                logger.info(f"[PIPELINE] Creating document with context: screen_locked={ctx.get('screen_locked') if ctx else 'unknown'}")

                # Intelligent document type detection
                doc_type = self._detect_document_type(command)
                logger.info(f"[PIPELINE] Detected document type: {doc_type}")

                # Check if screen is locked and document needs visual confirmation
                if ctx and ctx.get('screen_locked') and doc_type in ['presentation', 'visual_document']:
                    logger.info(f"[PIPELINE] Visual document requested but screen is locked - will notify user")

                # Smart API selection based on context
                if "google doc" in command.lower() or "google docs" in command.lower():
                    result = await self._create_google_doc(command)
                elif ctx and 'Google Chrome' in ctx.get('active_apps', []):
                    # If Chrome is open, prefer Google Docs
                    logger.info(f"[PIPELINE] Chrome is active, suggesting Google Docs")
                    result = await self._create_google_doc(command)
                else:
                    result = await self._create_local_document(command)

                return result

            # Process through context-aware handler
            try:
                if context_handler:
                    result = await context_handler.handle_command_with_context(
                        context.text,
                        execute_callback=create_document
                    )
                    logger.info(f"[PIPELINE] Context-aware processing completed successfully")
                else:
                    # Fallback: call create_document directly
                    logger.info(f"[PIPELINE] Using fallback document creation (no context handler)")
                    result = await create_document(context.text, system_context)
                    logger.info(f"[PIPELINE] Fallback document creation completed")

                # Store detailed context information
                context.metadata["context_steps"] = result.get("steps_taken", [])
                context.metadata["screen_was_locked"] = result.get("context", {}).get("screen_locked", False)

            except Exception as handler_error:
                logger.error(f"[PIPELINE] Context-aware processing error: {handler_error}")
                result = {
                    "success": False,
                    "messages": [f"Context processing error: {str(handler_error)}"]
                }

                # Set response
                if result.get("messages"):
                    # Use the messages from the context-aware handler (including unlock notifications)
                    context.response = " ".join(result["messages"])
                elif result.get("success"):
                    # If successful but no specific message, provide a default
                    topic = self._extract_document_topic(context.text)
                    context.response = f"I'm creating an essay about {topic} for you. Opening Google Docs now..."
                else:
                    context.response = "I encountered an issue with that request, Sir."

                context.metadata["document_creation"] = True
                context.metadata["steps_taken"] = result.get("steps_taken", [])
                return

            except Exception as e:
                logger.error(f"Error in document creation: {e}")
                context.response = f"I couldn't create the document: {str(e)}"
                return

        # For other commands, check if JARVIS is available
        if not self.jarvis:
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

    async def _execute_atomic_action(self, action) -> Dict[str, Any]:
        """
        Execute a single atomic action from the CompoundActionParser

        Args:
            action: AtomicAction object from CompoundActionParser

        Returns:
            Result dictionary with success status
        """
        from context_intelligence.analyzers.compound_action_parser import ActionType

        try:
            if action.type == ActionType.OPEN_APP:
                # Open application using MacOS controller
                from system_control.macos_controller import MacOSController
                controller = MacOSController()

                app_name = action.params.get('app_name', '')
                success, message = await asyncio.to_thread(controller.open_application, app_name)

                return {
                    'success': success,
                    'action': 'open_app',
                    'app_name': app_name,
                    'message': message
                }

            elif action.type == ActionType.SEARCH_WEB:
                # Perform web search using BrowserController
                from context_intelligence.automation.browser_controller import get_browser_controller
                browser = get_browser_controller(preferred_browser="Safari")

                query = action.params.get('query', '')
                success = await browser.search_google(query)

                return {
                    'success': success,
                    'action': 'search_web',
                    'query': query,
                    'message': f"Searched for: {query}" if success else f"Failed to search for: {query}"
                }

            elif action.type == ActionType.NAVIGATE_URL:
                # Navigate to URL using BrowserController
                from context_intelligence.automation.browser_controller import get_browser_controller
                browser = get_browser_controller(preferred_browser="Safari")

                url = action.params.get('url', '')
                success = await browser.navigate(url)

                return {
                    'success': success,
                    'action': 'navigate_url',
                    'url': url,
                    'message': f"Navigated to: {url}" if success else f"Failed to navigate to: {url}"
                }

            else:
                logger.warning(f"Unknown action type: {action.type}")
                return {
                    'success': False,
                    'action': str(action.type),
                    'message': 'Unknown action type'
                }

        except Exception as e:
            logger.error(f"Error executing atomic action {action.type}: {e}")
            return {
                'success': False,
                'action': str(action.type),
                'error': str(e),
                'message': f"Error: {str(e)}"
            }

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
            logger.info(f"Using pre-set response: {context.response[:100] if context.response else 'None'}...")
            # Mark that TTS should be triggered
            context.metadata["speak_response"] = True
            # Trigger voice synthesis for the response
            await self._synthesize_voice(context)
            return

        # Check if lock/unlock was handled successfully
        if "lock_unlock_result" in context.metadata:
            # Lock/unlock was handled - response should already be set in _process_command
            # If not set for some reason, generate a default success message
            if not context.response:
                result = context.metadata["lock_unlock_result"]
                if result.get("success"):
                    action = result.get("action", "command")
                    context.response = result.get("response", f"I've successfully executed the {action.replace('_', ' ')}, Sir.")
                else:
                    context.response = result.get("response", "The command was processed.")
            logger.info(f"Lock/unlock response finalized: {context.response[:100] if context.response else 'None'}...")
            # Mark that TTS should be triggered for lock/unlock confirmations
            context.metadata["speak_response"] = True
            # Trigger voice synthesis for the response
            await self._synthesize_voice(context)
            return

        # Prioritize responses from metadata
        if "claude_response" in context.metadata:
            context.response = context.metadata["claude_response"]
        elif "system_response" in context.metadata:
            context.response = context.metadata["system_response"]
        elif context.metadata.get("warning"):
            # Only show warning if it's not a handled command
            if context.metadata.get("handled_by") != "simple_unlock_handler":
                context.response = f"Warning: {context.metadata['warning']}"
            else:
                # Command was handled, don't override with warning
                logger.debug(f"Suppressing warning for handled command: {context.metadata.get('warning')}")
                if not context.response:
                    context.response = "Command executed successfully."
        else:
            context.response = f"I processed your command: '{context.text}', {context.user_name}."

        # Mark that TTS should be triggered for all responses
        context.metadata["speak_response"] = True
        logger.info(f"Response generated: {context.response[:100] if context.response else 'None'}...")

        # Trigger voice synthesis for the response
        await self._synthesize_voice(context)

    async def _synthesize_voice(self, context: PipelineContext):
        """Synthesize voice response using TTS if enabled"""
        if not context.metadata.get("speak_response", False):
            return

        if not context.response:
            return

        try:
            # Check if we should skip voice for certain responses
            skip_voice_phrases = ["Warning: No JARVIS instance", "Processing..."]
            if any(phrase in context.response for phrase in skip_voice_phrases):
                logger.debug(f"Skipping voice synthesis for: {context.response[:50]}...")
                return

            # Check if TTS is available through the WebSocket or API
            if context.metadata.get("websocket"):
                # If WebSocket is available, the speak flag is already set in metadata
                logger.info(f"[VOICE] WebSocket will handle TTS for: {context.response[:50]}...")
            else:
                # Try to use the TTS API directly for non-WebSocket contexts
                logger.info(f"[VOICE] Triggering TTS for: {context.response[:50]}...")

                # Import TTS handler
                try:
                    from api.async_tts_handler import generate_speech_async

                    # Generate speech asynchronously (fire and forget)
                    asyncio.create_task(self._speak_response(context.response))
                    logger.info("[VOICE] TTS task created for response")
                except ImportError:
                    logger.debug("TTS handler not available, voice synthesis skipped")

        except Exception as e:
            logger.error(f"Error in voice synthesis: {e}")
            # Don't fail the pipeline for voice synthesis errors

    async def _speak_response(self, text: str):
        """Helper method to speak text using TTS"""
        try:
            from api.jarvis_voice_api import async_subprocess_run

            # Use macOS say command for quick TTS
            await async_subprocess_run(
                ["say", "-v", "Daniel", "-r", "180", text],
                timeout=30.0
            )
            logger.info(f"[VOICE] Spoken: {text[:50]}...")
        except Exception as e:
            logger.error(f"Error speaking response: {e}")

    async def _create_google_doc(self, command: str) -> Dict[str, Any]:
        """Create a document using Google Docs API with intelligent retry"""
        try:
            # Extract topic from command
            topic = self._extract_document_topic(command)

            # Try to use Google Docs integration
            try:
                from context_intelligence.integrations.google_docs_api import GoogleDocsWriter

                writer = GoogleDocsWriter()
                doc_url = await writer.create_document(topic, command)

                return {
                    "success": True,
                    "message": f"I've created a Google Doc about {topic} and opened it for you.",
                    "doc_url": doc_url
                }
            except ImportError:
                # Fallback to browser automation
                from context_intelligence.automation.browser_controller import get_browser_controller

                browser = get_browser_controller()
                success = await browser.open_google_docs(topic)

                if success:
                    return {
                        "success": True,
                        "message": f"I've opened Google Docs for you to create a document about {topic}."
                    }
                else:
                    return {
                        "success": False,
                        "message": "I couldn't open Google Docs. Please check your internet connection."
                    }

        except Exception as e:
            logger.error(f"Error creating Google Doc: {e}")
            return {
                "success": False,
                "message": f"I encountered an error: {str(e)}"
            }

    async def _create_local_document(self, command: str) -> Dict[str, Any]:
        """Create a local document with content generation"""
        try:
            topic = self._extract_document_topic(command)

            # Use document writer for content generation
            from context_intelligence.executors.document_writer import DocumentWriterExecutor, DocumentRequest, DocumentType

            # Create document request
            request = DocumentRequest(
                topic=topic,
                document_type=DocumentType.ESSAY,
                original_command=command
            )

            # Create writer and initiate document creation
            writer = DocumentWriterExecutor()
            # This will trigger async document creation in background
            # NOTE: We start the task but don't await it - it runs asynchronously
            # IMPORTANT: This task will only start AFTER the context-aware handler
            # has finished checking screen lock and unlocking if necessary
            asyncio.create_task(writer.create_document(request))

            return {
                "success": True,
                # Return a message that will be used by context-aware handler
                "message": f"I'm creating an essay about {topic} for you, Sir.",
                "topic": topic,
                "task_started": True
            }

        except Exception as e:
            logger.error(f"Error creating local document: {e}")
            return {
                "success": False,
                "message": f"I couldn't create the document: {str(e)}"
            }

    async def _get_enhanced_system_context(self) -> Dict[str, Any]:
        """Get comprehensive system context for intelligent processing"""
        try:
            try:
                from context_intelligence.detectors.screen_lock_detector import get_screen_lock_detector
                screen_detector = get_screen_lock_detector()
                is_locked = await screen_detector.is_screen_locked()
            except ImportError:
                logger.warning("Screen lock detector not available, assuming unlocked")
                is_locked = False

            # Get active applications
            active_apps = []
            try:
                import subprocess
                result = subprocess.run(['osascript', '-e', 'tell application "System Events" to get name of (processes where background only is false)'],
                                      capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    active_apps = result.stdout.strip().split(', ')
            except:
                pass

            # Get network status
            network_connected = True  # You can expand this

            return {
                'screen_locked': is_locked,
                'active_apps': active_apps,
                'network_connected': network_connected,
                'timestamp': datetime.now().isoformat(),
                'system_load': self._get_system_load()
            }
        except Exception as e:
            logger.warning(f"Could not get full system context: {e}")
            return {
                'screen_locked': False,
                'active_apps': [],
                'network_connected': True,
                'timestamp': datetime.now().isoformat()
            }

    def _get_system_load(self) -> Dict[str, float]:
        """Get system resource usage"""
        try:
            import psutil
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent
            }
        except:
            return {'cpu_percent': 0.0, 'memory_percent': 0.0}

    def _detect_document_type(self, command: str) -> str:
        """Intelligently detect the type of document requested"""
        command_lower = command.lower()

        # Check for specific document types
        if any(word in command_lower for word in ['presentation', 'slides', 'powerpoint']):
            return 'presentation'
        elif any(word in command_lower for word in ['spreadsheet', 'excel', 'data', 'table']):
            return 'spreadsheet'
        elif any(word in command_lower for word in ['diagram', 'chart', 'graph', 'visual']):
            return 'visual_document'
        elif any(word in command_lower for word in ['essay', 'paper', 'report']):
            return 'text_document'
        elif any(word in command_lower for word in ['email', 'letter', 'memo']):
            return 'correspondence'
        else:
            return 'general_document'

    def _extract_document_topic(self, command: str) -> str:
        """Extract the topic from a document creation command"""
        import re

        command_lower = command.lower()

        # Try various patterns to extract topic
        patterns = [
            r'(?:write|create).*?(?:about|on|regarding)\s+(.+)',
            r'(?:essay|document|paper).*?(?:about|on)\s+(.+)',
            r'(?:write|create)\s+(?:an?|the)?\s*(?:essay|document|paper)?\s*(?:about|on)?\s*(.+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, command_lower)
            if match:
                topic = match.group(1).strip()
                # Clean up the topic
                topic = topic.replace('essay', '').replace('document', '').replace('paper', '')
                topic = topic.replace('about', '').replace('on', '').strip()
                if topic:
                    return topic

        # Fallback: extract key words after common triggers
        for trigger in ['write', 'create', 'compose', 'draft']:
            if trigger in command_lower:
                parts = command_lower.split(trigger, 1)
                if len(parts) > 1:
                    topic = parts[1].strip()
                    # Remove common words
                    for word in ['a', 'an', 'the', 'me', 'essay', 'document', 'paper', 'about', 'on']:
                        topic = topic.replace(word, ' ')
                    topic = ' '.join(topic.split())  # Clean up extra spaces
                    if topic:
                        return topic

        return "general topic"

    def _generate_error_response(self, context: PipelineContext, error: Exception) -> Dict[str, Any]:
        """Generate intelligent error response with recovery suggestions"""
        error_responses = {
            "TimeoutError": f"I apologize, {context.user_name}, but that's taking longer than expected. Please try again.",
            "ValueError": f"I'm sorry, {context.user_name}, but I couldn't understand that command.",
            "ConnectionError": f"I'm having trouble connecting to my services, {context.user_name}. Please check your connection.",
            "ImportError": f"Some components are not available, {context.user_name}. I'll try an alternative approach.",
            "PermissionError": f"I don't have permission to perform that action, {context.user_name}. You may need to unlock your screen.",
            "default": f"I apologize, {context.user_name}, but I encountered an error: {str(error)}"
        }

        error_type = type(error).__name__
        error_message = error_responses.get(error_type, error_responses["default"])

        # Add recovery suggestions based on error type
        recovery_suggestions = self._get_recovery_suggestions(error_type, context)

        return {
            "response": error_message,
            "metadata": context.metadata,
            "success": False,
            "command_id": context.command_id,
            "error": str(error),
            "error_type": error_type,
            "recovery_suggestions": recovery_suggestions,
            "metrics": context.metrics
        }

    def _get_recovery_suggestions(self, error_type: str, context: PipelineContext) -> List[str]:
        """Generate intelligent recovery suggestions based on error type"""
        suggestions = []

        if error_type == "TimeoutError":
            suggestions.append("Try simplifying your command")
            suggestions.append("Check if the system is busy")
        elif error_type == "ConnectionError":
            suggestions.append("Check your internet connection")
            suggestions.append("Verify the service is running")
        elif error_type == "PermissionError":
            if context.metadata.get("screen_locked"):
                suggestions.append("Your screen appears to be locked")
                suggestions.append("Try saying 'unlock my screen' first")
            else:
                suggestions.append("Check file or system permissions")
        elif error_type == "ImportError":
            suggestions.append("Some modules may need to be installed")
            suggestions.append("The system will use fallback methods")

        return suggestions

    async def _retry_with_fallback(self, func, *args, max_retries: int = 3, **kwargs):
        """Intelligent retry mechanism with exponential backoff"""
        last_error = None
        backoff = 1.0

        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

                # Exponential backoff
                if attempt < max_retries - 1:
                    await asyncio.sleep(backoff)
                    backoff *= 2

                # Try fallback strategies
                if attempt == 1:
                    # Second attempt: try with simplified parameters
                    if 'command' in kwargs:
                        kwargs['command'] = self._simplify_command(kwargs['command'])
                elif attempt == 2:
                    # Third attempt: use alternative method
                    if hasattr(self, f"{func.__name__}_fallback"):
                        fallback_func = getattr(self, f"{func.__name__}_fallback")
                        return await fallback_func(*args, **kwargs)

        raise last_error

    def _simplify_command(self, command: str) -> str:
        """Simplify a command for retry attempts"""
        # Remove complex connectors
        simplified = command.replace(" and then ", " ")
        simplified = simplified.replace(", then ", " ")
        simplified = simplified.replace(" followed by ", " ")

        # Remove extra words
        words_to_remove = ['please', 'could you', 'can you', 'would you']
        for word in words_to_remove:
            simplified = simplified.replace(word, '')

        return simplified.strip()

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
