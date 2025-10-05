#!/usr/bin/env python3
"""
Complete Async Architecture - Event-Driven Command Pipeline
Fixes "Processing..." stuck issue with fully non-blocking operations
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline processing stages"""
    RECEIVED = "received"
    INTENT_ANALYSIS = "intent_analysis"
    COMPONENT_LOADING = "component_loading"
    PROCESSING = "processing"
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


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = None

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                logger.info("Circuit breaker: transitioning to HALF_OPEN")
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")

        try:
            result = await func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e

    def on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            logger.info("Circuit breaker: transitioning to CLOSED")
            self.state = "CLOSED"

    def on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            logger.warning(f"Circuit breaker: OPEN (failures: {self.failure_count})")
            self.state = "OPEN"


class AsyncEventBus:
    """Event-driven message bus for pipeline communication"""

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_queue = asyncio.Queue()

    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to an event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
        logger.info(f"Subscribed handler to event: {event_type}")

    async def emit(self, event_type: str, data: Any):
        """Emit an event to all subscribers"""
        logger.debug(f"Emitting event: {event_type}")

        if event_type in self.subscribers:
            tasks = []
            for handler in self.subscribers[event_type]:
                tasks.append(self._safe_handle(handler, data))

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


class AsyncCommandPipeline:
    """Fully async, non-blocking command processing pipeline"""

    def __init__(self, jarvis_instance=None):
        self.jarvis = jarvis_instance
        self.event_bus = AsyncEventBus()
        self.circuit_breaker = CircuitBreaker()
        self.active_commands: Dict[str, PipelineContext] = {}

        # Pipeline stages as async functions
        self.stages = {
            PipelineStage.INTENT_ANALYSIS: self._analyze_intent,
            PipelineStage.COMPONENT_LOADING: self._load_components,
            PipelineStage.PROCESSING: self._process_command,
            PipelineStage.RESPONSE_GENERATION: self._generate_response,
        }

    async def process_async(self, text: str, user_name: str = "Sir") -> str:
        """Process command through async pipeline with immediate acknowledgment"""

        # Create pipeline context
        command_id = f"cmd_{int(time.time() * 1000)}"
        context = PipelineContext(
            command_id=command_id,
            text=text,
            user_name=user_name
        )

        self.active_commands[command_id] = context

        try:
            # Emit command received event
            await self.event_bus.emit("command_received", context)

            # Process through pipeline stages with circuit breaker
            result = await self.circuit_breaker.call(
                self._execute_pipeline,
                context
            )

            # Emit command completed event
            context.stage = PipelineStage.COMPLETED
            await self.event_bus.emit("command_completed", context)

            return result

        except Exception as e:
            logger.error(f"Pipeline error for command {command_id}: {e}", exc_info=True)
            context.stage = PipelineStage.FAILED
            context.error = str(e)
            await self.event_bus.emit("command_failed", context)

            return f"I apologize, {user_name}, but I encountered an error processing your request: {str(e)}"

        finally:
            # Cleanup after processing
            if command_id in self.active_commands:
                del self.active_commands[command_id]

    async def _execute_pipeline(self, context: PipelineContext) -> str:
        """Execute all pipeline stages in sequence"""

        for stage, handler in self.stages.items():
            context.stage = stage
            await self.event_bus.emit(f"stage_{stage.value}", context)

            try:
                # Execute stage with timeout to prevent hanging
                await asyncio.wait_for(
                    handler(context),
                    timeout=30.0  # 30 second timeout per stage
                )
            except asyncio.TimeoutError:
                raise Exception(f"Pipeline stage {stage.value} timed out after 30s")
            except Exception as e:
                raise Exception(f"Error in stage {stage.value}: {str(e)}")

        return context.response or "Task completed successfully."

    async def _analyze_intent(self, context: PipelineContext):
        """Analyze command intent (non-blocking)"""
        text_lower = context.text.lower()

        # Quick intent detection
        if any(kw in text_lower for kw in ["monitor", "watch", "track"]):
            context.intent = "monitoring"
        elif any(kw in text_lower for kw in ["open", "launch", "start"]):
            context.intent = "system_control"
        elif any(kw in text_lower for kw in ["write", "create", "document"]):
            context.intent = "document_creation"
        elif any(kw in text_lower for kw in ["weather", "temperature", "forecast"]):
            context.intent = "weather"
        else:
            context.intent = "conversation"

        logger.info(f"Intent detected: {context.intent} for command: {context.text}")

    async def _load_components(self, context: PipelineContext):
        """Load required components asynchronously"""
        # This would integrate with dynamic_component_manager
        if context.intent == "monitoring":
            context.components_loaded.append("vision")
        elif context.intent == "document_creation":
            context.components_loaded.append("document_writer")

        logger.info(f"Components loaded: {context.components_loaded}")

    async def _process_command(self, context: PipelineContext):
        """Process command based on intent (non-blocking)"""

        if not self.jarvis:
            context.metadata["warning"] = "No JARVIS instance available"
            return

        # Route to appropriate handler based on intent
        if context.intent == "conversation" and hasattr(self.jarvis, 'claude_chatbot'):
            try:
                # Use Claude chatbot for conversation (ensure it's async)
                if hasattr(self.jarvis.claude_chatbot, 'generate_response'):
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

    async def _generate_response(self, context: PipelineContext):
        """Generate final response"""

        # Prioritize responses from metadata
        if "claude_response" in context.metadata:
            context.response = context.metadata["claude_response"]
        elif "system_response" in context.metadata:
            context.response = context.metadata["system_response"]
        elif context.metadata.get("warning"):
            context.response = f"Warning: {context.metadata['warning']}"
        else:
            context.response = f"I processed your command: '{context.text}', {context.user_name}."

        logger.info(f"Response generated: {context.response[:100]}...")


class StreamingResponseHandler:
    """Handle streaming responses to prevent UI blocking"""

    def __init__(self):
        self.response_queue = asyncio.Queue()
        self.active_streams: Dict[str, asyncio.Queue] = {}

    async def create_stream(self, command_id: str) -> asyncio.Queue:
        """Create a new response stream"""
        stream = asyncio.Queue()
        self.active_streams[command_id] = stream
        return stream

    async def stream_response(self, command_id: str, chunk: str):
        """Stream a response chunk"""
        if command_id in self.active_streams:
            await self.active_streams[command_id].put(chunk)

    async def close_stream(self, command_id: str):
        """Close a response stream"""
        if command_id in self.active_streams:
            await self.active_streams[command_id].put(None)  # Signal end
            del self.active_streams[command_id]


# Global pipeline instance
_pipeline_instance = None


def get_async_pipeline(jarvis_instance=None) -> AsyncCommandPipeline:
    """Get or create the global async pipeline"""
    global _pipeline_instance

    if _pipeline_instance is None:
        _pipeline_instance = AsyncCommandPipeline(jarvis_instance)
        logger.info("✅ Async Command Pipeline initialized")

    return _pipeline_instance
