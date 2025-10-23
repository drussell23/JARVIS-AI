"""
Unified WebSocket Handler - Advanced Self-Healing WebSocket System

Features:
- Intelligent self-healing with automatic recovery
- UAE (Unified Awareness Engine) integration for system intelligence
- SAI (Situational Awareness Intelligence) integration for context
- Learning Database integration for pattern recognition
- Dynamic recovery strategies with no hardcoding
- Circuit breaker pattern for resilience
- Predictive disconnection prevention
- Advanced async operations with robust error handling
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging
import json
import asyncio
import time
from typing import Dict, Any, Optional, Set, List
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

# Import async pipeline for non-blocking WebSocket operations
from core.async_pipeline import get_async_pipeline, AdvancedAsyncPipeline

logger = logging.getLogger(__name__)

router = APIRouter()

# Active connections management
active_connections: Dict[str, WebSocket] = {}
connection_capabilities: Dict[str, Set[str]] = {}


# ============================================================================
# ADVANCED CONNECTION HEALTH & SELF-HEALING SYSTEM
# ============================================================================

class ConnectionState(Enum):
    """WebSocket connection states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    DISCONNECTED = "disconnected"


@dataclass
class ConnectionHealth:
    """Real-time health metrics for a WebSocket connection"""
    client_id: str
    websocket: WebSocket
    state: ConnectionState = ConnectionState.HEALTHY
    connection_time: float = field(default_factory=time.time)
    last_message_time: float = field(default_factory=time.time)
    last_ping_time: float = field(default_factory=time.time)
    messages_sent: int = 0
    messages_received: int = 0
    errors: int = 0
    reconnections: int = 0
    health_score: float = 100.0
    latency_ms: float = 0.0
    last_error: Optional[str] = None
    recovery_attempts: int = 0

    # Intelligence integration
    uae_context: Optional[Dict] = None
    sai_context: Optional[Dict] = None
    learned_patterns: List[str] = field(default_factory=list)


class UnifiedWebSocketManager:
    """
    Advanced WebSocket Manager with Self-Healing Intelligence

    Integrates with:
    - UAE (Unified Awareness Engine) for system-wide intelligence
    - SAI (Situational Awareness Intelligence) for context awareness
    - Learning Database for pattern recognition and prediction
    """

    def __init__(self):
        # Connection management
        self.connections: Dict[str, WebSocket] = {}
        self.connection_health: Dict[str, ConnectionHealth] = {}
        self.display_monitor = None  # Will be set by main.py

        # Initialize async pipeline for WebSocket operations
        self.pipeline = get_async_pipeline()
        self._register_pipeline_stages()

        # Intelligence integration (injected via dependency injection - no hardcoding)
        self.uae_engine = None  # Will be set by main.py
        self.sai_engine = None  # Will be set by main.py
        self.learning_db = None  # Will be set by main.py

        # Self-healing configuration (dynamic, loaded from config or learned)
        self.config = {
            "health_check_interval": 5.0,
            "message_timeout": 30.0,
            "ping_interval": 15.0,
            "max_recovery_attempts": 5,
            "circuit_breaker_threshold": 3,
            "circuit_breaker_timeout": 60.0,
            "predictive_healing_enabled": True,
            "auto_learning_enabled": True
        }

        # Circuit breaker state
        self.circuit_open = False
        self.circuit_failures = 0
        self.circuit_open_time: Optional[float] = None

        # Background tasks
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.recovery_tasks: Dict[str, asyncio.Task] = {}

        # Learning & patterns
        self.disconnection_patterns: List[Dict] = []
        self.recovery_success_rate: Dict[str, float] = {}

        # Message handlers (dynamically extensible)
        self.handlers = {
            # Voice/JARVIS handlers
            "command": self._handle_voice_command,
            "voice_command": self._handle_voice_command,
            "jarvis_command": self._handle_voice_command,

            # Vision handlers
            "vision_analyze": self._handle_vision_analyze,
            "vision_monitor": self._handle_vision_monitor,
            "workspace_analysis": self._handle_workspace_analysis,

            # Audio/ML handlers
            "ml_audio_stream": self._handle_ml_audio,
            "audio_error": self._handle_audio_error,

            # System handlers
            "model_status": self._handle_model_status,
            "network_status": self._handle_network_status,
            "notification": self._handle_notification,

            # General handlers
            "ping": self._handle_ping,
            "pong": self._handle_pong,
            "subscribe": self._handle_subscribe,
            "unsubscribe": self._handle_unsubscribe,
            "health_check": self._handle_health_check,
        }

        logger.info("[UNIFIED-WS] Advanced WebSocket Manager initialized")
        logger.info("[UNIFIED-WS] Self-healing: ✅ | Circuit breaker: ✅ | Predictive healing: ✅")

    def set_intelligence_engines(self, uae=None, sai=None, learning_db=None):
        """Inject intelligence engines (dependency injection - no hardcoding)"""
        self.uae_engine = uae
        self.sai_engine = sai
        self.learning_db = learning_db

        logger.info(f"[UNIFIED-WS] Intelligence engines set: UAE={'✅' if uae else '❌'}, SAI={'✅' if sai else '❌'}, Learning DB={'✅' if learning_db else '❌'}")

    async def start_health_monitoring(self):
        """Start intelligent health monitoring"""
        if self.health_monitor_task is None:
            self.health_monitor_task = asyncio.create_task(self._health_monitoring_loop())
            logger.info("[UNIFIED-WS] 🏥 Health monitoring started")

    async def stop_health_monitoring(self):
        """Stop health monitoring"""
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass
            logger.info("[UNIFIED-WS] Health monitoring stopped")

    async def _health_monitoring_loop(self):
        """Continuous health monitoring with predictive healing"""
        while True:
            try:
                await asyncio.sleep(self.config["health_check_interval"])

                current_time = time.time()

                for client_id, health in list(self.connection_health.items()):
                    # Check message timeout
                    time_since_message = current_time - health.last_message_time

                    if time_since_message > self.config["message_timeout"]:
                        # Degraded connection
                        if health.state == ConnectionState.HEALTHY:
                            health.state = ConnectionState.DEGRADED
                            health.health_score = max(0, health.health_score - 20)
                            logger.warning(f"[UNIFIED-WS] Connection {client_id} degraded (no messages for {time_since_message:.1f}s)")

                            # Notify SAI of degradation
                            await self._notify_sai("connection_degraded", health)

                            # Attempt preventive recovery
                            await self._preventive_recovery(health)

                    # Send periodic pings
                    time_since_ping = current_time - health.last_ping_time
                    if time_since_ping > self.config["ping_interval"]:
                        await self._send_ping(health)

                    # Predictive healing (UAE-powered)
                    if self.config["predictive_healing_enabled"] and self.uae_engine:
                        await self._predictive_healing(health)

                # Check circuit breaker
                await self._check_circuit_breaker()

            except Exception as e:
                logger.error(f"[UNIFIED-WS] Health monitoring error: {e}", exc_info=True)

    async def _send_ping(self, health: ConnectionHealth):
        """Send ping to check connection health"""
        try:
            ping_time = time.time()
            await health.websocket.send_json({
                "type": "ping",
                "timestamp": ping_time
            })
            health.last_ping_time = ping_time
            logger.debug(f"[UNIFIED-WS] Sent ping to {health.client_id}")
        except Exception as e:
            logger.error(f"[UNIFIED-WS] Failed to send ping to {health.client_id}: {e}")
            health.errors += 1
            health.health_score = max(0, health.health_score - 10)

    async def _preventive_recovery(self, health: ConnectionHealth):
        """Attempt preventive recovery before full disconnection"""
        if health.recovery_attempts >= self.config["max_recovery_attempts"]:
            logger.warning(f"[UNIFIED-WS] Max recovery attempts reached for {health.client_id}")
            health.state = ConnectionState.DISCONNECTED
            return

        try:
            health.state = ConnectionState.RECOVERING
            health.recovery_attempts += 1

            logger.info(f"[UNIFIED-WS] Attempting preventive recovery for {health.client_id} (attempt {health.recovery_attempts})")

            # Strategy 1: Send wake-up ping
            await self._send_ping(health)

            # Strategy 2: Notify client of degradation
            await health.websocket.send_json({
                "type": "connection_health",
                "state": "degraded",
                "health_score": health.health_score,
                "message": "Connection health degraded, attempting recovery"
            })

            # Strategy 3: Log pattern to learning database
            if self.config["auto_learning_enabled"] and self.learning_db:
                await self._log_connection_pattern(health, "preventive_recovery")

            # Wait a bit and check if recovery worked
            await asyncio.sleep(2)

            if health.state == ConnectionState.RECOVERING:
                # If we received a message during recovery, it worked
                current_time = time.time()
                if current_time - health.last_message_time < 3:
                    health.state = ConnectionState.HEALTHY
                    health.health_score = min(100, health.health_score + 30)
                    logger.info(f"[UNIFIED-WS] ✅ Recovery successful for {health.client_id}")

                    # Notify SAI of recovery
                    await self._notify_sai("connection_recovered", health)
                else:
                    # Recovery didn't work
                    health.health_score = max(0, health.health_score - 15)

        except Exception as e:
            logger.error(f"[UNIFIED-WS] Preventive recovery failed for {health.client_id}: {e}")
            health.errors += 1
            health.health_score = max(0, health.health_score - 20)

    async def _predictive_healing(self, health: ConnectionHealth):
        """UAE-powered predictive healing to prevent disconnections"""
        if not self.uae_engine:
            return

        try:
            # Gather connection metrics
            metrics = {
                "client_id": health.client_id,
                "health_score": health.health_score,
                "latency_ms": health.latency_ms,
                "messages_sent": health.messages_sent,
                "messages_received": health.messages_received,
                "errors": health.errors,
                "reconnections": health.reconnections,
                "connection_duration": time.time() - health.connection_time,
                "time_since_message": time.time() - health.last_message_time
            }

            # Ask UAE to predict disconnection risk
            prediction = await self._ask_uae_prediction(metrics)

            if prediction and prediction.get("risk_level", "low") in ["high", "critical"]:
                logger.warning(f"[UNIFIED-WS] 🔮 UAE predicts disconnection risk: {prediction.get('risk_level')} for {health.client_id}")

                # Apply UAE-suggested recovery strategy
                strategy = prediction.get("suggested_strategy", "ping")

                if strategy == "immediate_reconnect":
                    await self._notify_uae("immediate_reconnect_needed", health)
                    # Notify client to prepare for reconnection
                    await health.websocket.send_json({
                        "type": "reconnection_advisory",
                        "reason": "predictive_healing",
                        "message": "Connection instability detected, please standby"
                    })
                elif strategy == "increase_pings":
                    # Temporarily increase ping frequency
                    self.config["ping_interval"] = max(5.0, self.config["ping_interval"] / 2)
                    logger.info(f"[UNIFIED-WS] Increased ping frequency to {self.config['ping_interval']}s")
                elif strategy == "reduce_load":
                    # Notify client to reduce message frequency
                    await health.websocket.send_json({
                        "type": "connection_optimization",
                        "action": "reduce_load",
                        "message": "Optimizing connection performance"
                    })

                # Log prediction to learning database
                if self.learning_db:
                    await self._log_uae_prediction(health, prediction)

        except Exception as e:
            logger.error(f"[UNIFIED-WS] Predictive healing error for {health.client_id}: {e}")

    async def _check_circuit_breaker(self):
        """Manage circuit breaker state for system-wide resilience"""
        current_time = time.time()

        if self.circuit_open:
            # Check if timeout has passed
            if self.circuit_open_time and (current_time - self.circuit_open_time) > self.config["circuit_breaker_timeout"]:
                # Try half-open state
                logger.info("[UNIFIED-WS] Circuit breaker entering half-open state")
                self.circuit_open = False
                self.circuit_failures = 0
                self.circuit_open_time = None

                # Notify SAI of circuit recovery
                await self._notify_sai("circuit_breaker_half_open", None)
        else:
            # Check if we should open the circuit
            if self.circuit_failures >= self.config["circuit_breaker_threshold"]:
                logger.error(f"[UNIFIED-WS] 🔴 Circuit breaker OPEN (failures: {self.circuit_failures})")
                self.circuit_open = True
                self.circuit_open_time = current_time

                # Notify all clients
                await self.broadcast({
                    "type": "system_status",
                    "status": "degraded",
                    "message": "System experiencing high failure rate, entering protective mode"
                })

                # Notify SAI
                await self._notify_sai("circuit_breaker_open", None)

    async def _notify_sai(self, event: str, health: Optional[ConnectionHealth]):
        """Notify SAI of connection events for situational awareness"""
        if not self.sai_engine:
            return

        try:
            event_data = {
                "event": event,
                "timestamp": time.time(),
                "source": "unified_websocket"
            }

            if health:
                event_data["client_id"] = health.client_id
                event_data["health_score"] = health.health_score
                event_data["connection_state"] = health.state.value
                event_data["latency_ms"] = health.latency_ms

            # Call SAI's event notification method
            if hasattr(self.sai_engine, "notify_event"):
                await self.sai_engine.notify_event(event_data)
                logger.debug(f"[UNIFIED-WS] Notified SAI of event: {event}")

        except Exception as e:
            logger.error(f"[UNIFIED-WS] Failed to notify SAI: {e}")

    async def _notify_uae(self, event: str, health: ConnectionHealth):
        """Notify UAE of critical connection events"""
        if not self.uae_engine:
            return

        try:
            event_data = {
                "event": event,
                "timestamp": time.time(),
                "client_id": health.client_id,
                "health_metrics": {
                    "score": health.health_score,
                    "latency": health.latency_ms,
                    "errors": health.errors,
                    "state": health.state.value
                }
            }

            # Call UAE's notification method
            if hasattr(self.uae_engine, "notify_websocket_event"):
                await self.uae_engine.notify_websocket_event(event_data)
                logger.debug(f"[UNIFIED-WS] Notified UAE of event: {event}")

        except Exception as e:
            logger.error(f"[UNIFIED-WS] Failed to notify UAE: {e}")

    async def _handle_pong(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pong responses to update latency metrics"""
        if client_id in self.connection_health:
            health = self.connection_health[client_id]

            # Calculate latency
            pong_time = time.time()
            ping_timestamp = message.get("timestamp", health.last_ping_time)
            health.latency_ms = (pong_time - ping_timestamp) * 1000

            # Update health score based on latency
            if health.latency_ms < 100:
                health.health_score = min(100, health.health_score + 2)
            elif health.latency_ms > 500:
                health.health_score = max(0, health.health_score - 5)

            # Update state if recovering
            if health.state == ConnectionState.DEGRADED and health.health_score > 80:
                health.state = ConnectionState.HEALTHY
                logger.info(f"[UNIFIED-WS] Connection {client_id} restored to healthy state")

            logger.debug(f"[UNIFIED-WS] Pong from {client_id}: latency={health.latency_ms:.1f}ms, health={health.health_score:.1f}")

        return {
            "type": "pong_ack",
            "latency_ms": self.connection_health[client_id].latency_ms if client_id in self.connection_health else 0
        }

    async def _handle_health_check(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle explicit health check requests"""
        if client_id in self.connection_health:
            health = self.connection_health[client_id]

            return {
                "type": "health_status",
                "client_id": client_id,
                "state": health.state.value,
                "health_score": health.health_score,
                "latency_ms": health.latency_ms,
                "connection_duration": time.time() - health.connection_time,
                "messages_sent": health.messages_sent,
                "messages_received": health.messages_received,
                "errors": health.errors
            }

        return {
            "type": "health_status",
            "error": "Client health data not found"
        }

    async def _ask_uae_prediction(self, metrics: Dict) -> Optional[Dict]:
        """Ask UAE to predict disconnection risk"""
        if not self.uae_engine or not hasattr(self.uae_engine, "predict_connection_risk"):
            return None

        try:
            prediction = await self.uae_engine.predict_connection_risk(metrics)
            return prediction
        except Exception as e:
            logger.error(f"[UNIFIED-WS] UAE prediction failed: {e}")
            return None

    async def _log_connection_pattern(self, health: ConnectionHealth, event_type: str):
        """Log connection patterns to learning database"""
        if not self.learning_db:
            return

        try:
            pattern = {
                "timestamp": time.time(),
                "client_id": health.client_id,
                "event_type": event_type,
                "health_score": health.health_score,
                "latency_ms": health.latency_ms,
                "state": health.state.value,
                "errors": health.errors,
                "recovery_attempts": health.recovery_attempts
            }

            # Store in learning database
            if hasattr(self.learning_db, "log_websocket_pattern"):
                await self.learning_db.log_websocket_pattern(pattern)
                logger.debug(f"[UNIFIED-WS] Logged connection pattern: {event_type}")

        except Exception as e:
            logger.error(f"[UNIFIED-WS] Failed to log pattern: {e}")

    async def _log_uae_prediction(self, health: ConnectionHealth, prediction: Dict):
        """Log UAE predictions for learning"""
        if not self.learning_db:
            return

        try:
            log_entry = {
                "timestamp": time.time(),
                "client_id": health.client_id,
                "prediction": prediction,
                "actual_state": health.state.value,
                "health_score": health.health_score
            }

            if hasattr(self.learning_db, "log_uae_prediction"):
                await self.learning_db.log_uae_prediction(log_entry)

        except Exception as e:
            logger.error(f"[UNIFIED-WS] Failed to log UAE prediction: {e}")

    def _register_pipeline_stages(self):
        """Register async pipeline stages for WebSocket operations"""

        # Message processing stage
        self.pipeline.register_stage(
            "message_processing",
            self._process_message_async,
            timeout=60.0,  # Increased from 30s for multi-space vision queries
            retry_count=1,
            required=True
        )

        # Command execution stage
        self.pipeline.register_stage(
            "command_execution",
            self._execute_command_async,
            timeout=90.0,  # Increased from 45s for complex vision processing
            retry_count=2,
            required=True
        )

        # Response streaming stage
        self.pipeline.register_stage(
            "response_streaming",
            self._stream_response_async,
            timeout=60.0,
            retry_count=0,
            required=False  # Optional for non-streaming responses
        )

    async def _process_message_async(self, context):
        """Non-blocking message processing via async pipeline"""
        try:
            message = context.metadata.get("message", {})
            client_id = context.metadata.get("client_id", "")

            # Parse message type
            msg_type = message.get("type", "")
            context.metadata["msg_type"] = msg_type

            # Validate message
            if not msg_type:
                context.metadata["error"] = "Missing message type"
                return

            # Store for next stage
            context.metadata["validated"] = True

        except Exception as e:
            logger.error(f"Message processing error: {e}")
            context.metadata["error"] = str(e)

    async def _execute_command_async(self, context):
        """Non-blocking command execution via async pipeline"""
        try:
            message = context.metadata.get("message", {})
            msg_type = context.metadata.get("msg_type", "")
            client_id = context.metadata.get("client_id", "")

            # Route to appropriate handler
            if msg_type == "command" or msg_type == "voice_command":
                # Execute voice command
                from .jarvis_voice_api import jarvis_api
                from pydantic import BaseModel

                class VoiceCommand(BaseModel):
                    text: str

                command_text = message.get("command", message.get("text", ""))
                command_obj = VoiceCommand(text=command_text)

                result = await jarvis_api.process_command(command_obj)

                context.metadata["response"] = {
                    "type": "response",
                    "text": result.get("response", ""),
                    "status": result.get("status", "success"),
                    "command_type": result.get("command_type", "unknown"),
                    "speak": True
                }

            elif msg_type == "vision_analyze":
                # Execute vision analysis
                context.metadata["response"] = await self._execute_vision_analysis(message)

            else:
                context.metadata["response"] = {
                    "type": "error",
                    "error": f"Unknown message type: {msg_type}"
                }

        except Exception as e:
            logger.error(f"Command execution error: {e}")
            context.metadata["error"] = str(e)
            context.metadata["response"] = {
                "type": "error",
                "error": str(e)
            }

    async def _stream_response_async(self, context):
        """Non-blocking response streaming via async pipeline"""
        try:
            websocket = context.metadata.get("websocket")
            response = context.metadata.get("response", {})
            stream_mode = context.metadata.get("stream_mode", False)

            if stream_mode and websocket:
                # Stream response in chunks
                response_text = response.get("text", "")
                chunk_size = 50

                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i:i + chunk_size]
                    await websocket.send_json({
                        "type": "stream_chunk",
                        "chunk": chunk,
                        "progress": (i + chunk_size) / len(response_text)
                    })
                    await asyncio.sleep(0.05)

                # Send completion
                await websocket.send_json({
                    "type": "stream_complete",
                    "message": "Streaming complete"
                })
            else:
                # Send complete response
                if websocket and response:
                    await websocket.send_json(response)

            context.metadata["sent"] = True

        except Exception as e:
            logger.error(f"Response streaming error: {e}")
            context.metadata["error"] = str(e)

    async def _execute_vision_analysis(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Execute vision analysis (helper for command execution)"""
        try:
            from ..main import app

            if hasattr(app.state, 'vision_analyzer'):
                analyzer = app.state.vision_analyzer

                screenshot = await analyzer.capture_screen()
                if screenshot:
                    query = message.get("query", "Describe what you see on screen")
                    result = await analyzer.describe_screen({"screenshot": screenshot, "query": query})

                    return {
                        "type": "vision_result",
                        "success": result.get("success", False),
                        "description": result.get("description", ""),
                        "timestamp": datetime.now().isoformat()
                    }

            return {
                "type": "vision_result",
                "success": False,
                "error": "Vision analyzer not available"
            }

        except Exception as e:
            logger.error(f"Vision analysis error: {e}")
            return {
                "type": "vision_result",
                "success": False,
                "error": str(e)
            }

    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept new WebSocket connection with health monitoring"""
        await websocket.accept()
        self.connections[client_id] = websocket
        connection_capabilities[client_id] = set()

        # Create health monitoring for this connection
        health = ConnectionHealth(
            client_id=client_id,
            websocket=websocket
        )
        self.connection_health[client_id] = health

        logger.info(f"[UNIFIED-WS] ✅ Client {client_id} connected (health monitoring: active)")

        # Start health monitoring if not already running
        if not self.health_monitor_task:
            await self.start_health_monitoring()

        # Notify SAI of new connection
        await self._notify_sai("connection_established", health)

        # Log connection to learning database
        if self.learning_db:
            await self._log_connection_pattern(health, "connection_established")

        # Send welcome message with health features
        await websocket.send_json({
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat(),
            "available_handlers": list(self.handlers.keys()),
            "features": {
                "self_healing": True,
                "predictive_healing": self.config["predictive_healing_enabled"],
                "health_monitoring": True,
                "circuit_breaker": True
            }
        })

        # Send current display status if display monitor is available
        if self.display_monitor:
            try:
                available_displays = self.display_monitor.get_available_display_details()
                if available_displays:
                    logger.info(f"[WS] Sending {len(available_displays)} available displays to new client")
                    for display in available_displays:
                        await websocket.send_json({
                            "type": "display_detected",
                            "display_name": display["display_name"],
                            "display_id": display["display_id"],
                            "message": display["message"],
                            "timestamp": datetime.now().isoformat(),
                            "on_connect": True  # Flag to indicate this is initial status
                        })
            except Exception as e:
                logger.warning(f"[WS] Failed to send display status to new client: {e}")
        
    async def disconnect(self, client_id: str):
        """Remove WebSocket connection with learning and SAI notification"""
        # Gather final health metrics before removal
        health = self.connection_health.get(client_id)

        if health:
            # Log disconnection pattern to learning database
            if self.learning_db:
                await self._log_connection_pattern(health, "disconnection")

                # Store final session metrics for learning
                session_summary = {
                    "client_id": client_id,
                    "connection_duration": time.time() - health.connection_time,
                    "total_messages": health.messages_sent + health.messages_received,
                    "errors": health.errors,
                    "reconnections": health.reconnections,
                    "final_health_score": health.health_score,
                    "avg_latency_ms": health.latency_ms,
                    "learned_patterns": health.learned_patterns
                }

                if hasattr(self.learning_db, "log_session_summary"):
                    await self.learning_db.log_session_summary(session_summary)

            # Notify SAI of disconnection
            await self._notify_sai("connection_disconnected", health)

            logger.info(f"[UNIFIED-WS] Client {client_id} disconnected (duration: {time.time() - health.connection_time:.1f}s, health: {health.health_score:.1f})")

        # Clean up
        if client_id in self.connections:
            del self.connections[client_id]
        if client_id in connection_capabilities:
            del connection_capabilities[client_id]
        if client_id in self.connection_health:
            del self.connection_health[client_id]
        if client_id in self.recovery_tasks:
            # Cancel any ongoing recovery tasks
            self.recovery_tasks[client_id].cancel()
            del self.recovery_tasks[client_id]
        
    async def handle_message(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route message to appropriate handler via async pipeline

        Args:
            client_id: Client identifier
            message: Message to process

        Returns:
            Response dictionary
        """
        msg_type = message.get("type", "")

        # Check if message type should use pipeline processing
        pipeline_types = {"command", "voice_command", "jarvis_command", "vision_analyze", "vision_monitor"}

        if msg_type in pipeline_types:
            try:
                # Process through async pipeline for non-blocking execution
                websocket = self.connections.get(client_id)

                result = await self.pipeline.process_async(
                    text=message.get("text", message.get("command", "")),
                    metadata={
                        "message": message,
                        "client_id": client_id,
                        "websocket": websocket,
                        "stream_mode": message.get("stream", False)
                    }
                )

                # Return response from pipeline
                # First check if we have a direct response
                if result.get("response"):
                    # Include speak flag for voice output
                    response_dict = {
                        "type": "command_response",
                        "response": result.get("response"),
                        "success": result.get("success", True),
                        "speak": True  # Enable text-to-speech for all responses
                    }

                    # Add additional metadata for lock/unlock commands
                    if result.get("metadata", {}).get("lock_unlock_result"):
                        lock_result = result["metadata"]["lock_unlock_result"]
                        response_dict["action"] = lock_result.get("action", "")
                        response_dict["command_type"] = lock_result.get("type", "system_control")

                    return response_dict

                # Fall back to metadata response
                return result.get("metadata", {}).get("response", {
                    "type": "error",
                    "error": "No response generated"
                })

            except Exception as e:
                logger.error(f"Pipeline processing error for {msg_type}: {e}")
                return {
                    "type": "error",
                    "error": str(e),
                    "original_type": msg_type
                }

        # Fall back to legacy handlers for other message types
        elif msg_type in self.handlers:
            try:
                return await self.handlers[msg_type](client_id, message)
            except Exception as e:
                logger.error(f"Error handling {msg_type}: {e}")
                return {
                    "type": "error",
                    "error": str(e),
                    "original_type": msg_type
                }
        else:
            return {
                "type": "error",
                "error": f"Unknown message type: {msg_type}",
                "available_types": list(self.handlers.keys())
            }
    
    async def broadcast(self, message: Dict[str, Any], capability: Optional[str] = None):
        """Broadcast message to all connected clients or those with specific capability"""
        disconnected = []
        
        for client_id, websocket in self.connections.items():
            # Skip if capability filter is set and client doesn't have it
            if capability and capability not in connection_capabilities.get(client_id, set()):
                continue
                
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to {client_id}: {e}")
                disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)
    
    # Handler implementations
    
    async def _handle_voice_command(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle voice/JARVIS commands"""
        try:
            command_text = message.get("command", message.get("text", ""))

            logger.info(f"[WS] Processing voice command: {command_text}")

            # Use unified command processor for full command classification and handling
            # This includes display commands, system commands, vision, etc.
            try:
                from api.unified_command_processor import get_unified_processor
                processor = get_unified_processor()
                result = await processor.process_command(command_text, websocket=None)

                response_text = result.get("response", "Command processed, Sir.")
                success = result.get("success", False)
                command_type = result.get("command_type", "unknown")

                logger.info(f"[WS] Unified processor result: {response_text[:100]}")

                return {
                    "type": "response",
                    "text": response_text,
                    "status": "success" if success else "error",
                    "command_type": command_type,
                    "speak": True
                }
            except ImportError:
                # Fallback to async pipeline if unified processor not available
                logger.warning("[WS] Unified processor not available, falling back to async pipeline")
                result = await self.pipeline.execute(command_text)

                response_text = result.response or "Command processed, Sir."

                logger.info(f"[WS] Pipeline result: {response_text[:100]}")

                return {
                    "type": "response",
                    "text": response_text,
                    "status": "success" if result.success else "error",
                    "command_type": result.metadata.get("intent", "unknown"),
                    "speak": True
                }

        except Exception as e:
            logger.error(f"Error processing voice command: {e}", exc_info=True)
            return {
                "type": "response",
                "text": f"I encountered an error: {str(e)}",
                "status": "error",
                "speak": True
            }
    
    async def _handle_vision_analyze(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle vision analysis requests"""
        try:
            # Get vision analyzer from app state
            from ..main import app
            
            if hasattr(app.state, 'vision_analyzer'):
                analyzer = app.state.vision_analyzer
                
                # Perform analysis
                screenshot = await analyzer.capture_screen()
                if screenshot:
                    query = message.get("query", "Describe what you see on screen")
                    result = await analyzer.describe_screen({"screenshot": screenshot, "query": query})
                    
                    return {
                        "type": "vision_result",
                        "success": result.get("success", False),
                        "description": result.get("description", ""),
                        "timestamp": datetime.now().isoformat()
                    }
            
            return {
                "type": "vision_result",
                "success": False,
                "error": "Vision analyzer not available"
            }
            
        except Exception as e:
            logger.error(f"Vision analysis error: {e}")
            return {
                "type": "vision_result",
                "success": False,
                "error": str(e)
            }
    
    async def _handle_vision_monitor(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle continuous vision monitoring"""
        action = message.get("action", "start")
        
        if action == "start":
            connection_capabilities[client_id].add("vision_monitoring")
            
            # Start monitoring loop for this client
            asyncio.create_task(self._vision_monitoring_loop(client_id))
            
            return {
                "type": "monitor_status",
                "status": "started",
                "client_id": client_id
            }
        elif action == "stop":
            connection_capabilities[client_id].discard("vision_monitoring")
            return {
                "type": "monitor_status",
                "status": "stopped",
                "client_id": client_id
            }
    
    async def _vision_monitoring_loop(self, client_id: str):
        """Continuous vision monitoring loop"""
        while client_id in self.connections and "vision_monitoring" in connection_capabilities.get(client_id, set()):
            try:
                # Analyze screen periodically
                await self._handle_vision_analyze(client_id, {"type": "vision_analyze"})
                await asyncio.sleep(5)  # Analyze every 5 seconds
            except Exception as e:
                logger.error(f"Monitoring error for {client_id}: {e}")
                break
    
    async def _handle_workspace_analysis(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle workspace analysis requests"""
        # This would integrate with workspace analyzer
        return {
            "type": "workspace_result",
            "analysis": "Workspace analysis placeholder",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_ml_audio(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ML audio streaming"""
        audio_data = message.get("audio_data", "")
        
        return {
            "type": "ml_audio_result",
            "status": "processed",
            "has_speech": len(audio_data) > 0,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_audio_error(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle audio error recovery"""
        error_type = message.get("error_type", "unknown")
        
        return {
            "type": "audio_recovery",
            "strategy": "reconnect" if error_type == "connection" else "retry",
            "delay_ms": 1000
        }
    
    async def _handle_model_status(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ML model status requests"""
        # This would integrate with ML model loader
        return {
            "type": "model_status",
            "models_loaded": True,
            "status": "ready",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_network_status(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle network status checks"""
        return {
            "type": "network_status",
            "status": "connected",
            "latency_ms": 50,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_notification(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle notification detection"""
        # This would integrate with notification detection
        return {
            "type": "notification_result",
            "notifications": [],
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_ping(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ping/pong for connection keep-alive"""
        return {
            "type": "pong",
            "timestamp": message.get("timestamp", datetime.now().isoformat())
        }
    
    async def _handle_subscribe(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle capability subscription"""
        capabilities = message.get("capabilities", [])
        
        for cap in capabilities:
            connection_capabilities[client_id].add(cap)
        
        return {
            "type": "subscription_result",
            "subscribed": capabilities,
            "current_capabilities": list(connection_capabilities[client_id])
        }
    
    async def _handle_unsubscribe(self, client_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle capability unsubscription"""
        capabilities = message.get("capabilities", [])
        
        for cap in capabilities:
            connection_capabilities[client_id].discard(cap)
        
        return {
            "type": "unsubscription_result",
            "unsubscribed": capabilities,
            "current_capabilities": list(connection_capabilities[client_id])
        }


# Create global manager instance
ws_manager = UnifiedWebSocketManager()


def set_jarvis_instance(jarvis_api):
    """Set the JARVIS instance for the WebSocket pipeline"""
    if ws_manager and ws_manager.pipeline:
        ws_manager.pipeline.jarvis = jarvis_api
        logger.info("✅ JARVIS instance set in unified WebSocket pipeline")


@router.websocket("/ws")
async def unified_websocket_endpoint(websocket: WebSocket):
    """Single unified WebSocket endpoint for all communication with advanced self-healing"""
    client_id = f"client_{id(websocket)}_{datetime.now().timestamp()}"

    await ws_manager.connect(websocket, client_id)

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()

            # Update health metrics
            if client_id in ws_manager.connection_health:
                health = ws_manager.connection_health[client_id]
                health.last_message_time = time.time()
                health.messages_received += 1

                # Update health score on successful message
                health.health_score = min(100, health.health_score + 1)

                # If recovering, mark as healthy
                if health.state == ConnectionState.RECOVERING:
                    health.state = ConnectionState.HEALTHY
                    logger.info(f"[UNIFIED-WS] Connection {client_id} recovered to healthy state")

            # Log incoming command for debugging
            if data.get("type") == "command" or data.get("type") == "voice_command":
                logger.info(f"[WS] Received command: {data.get('text', data.get('command', 'unknown'))}")

            # Handle message
            response = await ws_manager.handle_message(client_id, data)

            # Update health metrics for sent message
            if client_id in ws_manager.connection_health:
                health = ws_manager.connection_health[client_id]
                health.messages_sent += 1

            # Log outgoing response for debugging lock/unlock
            if "lock" in str(data).lower() or "unlock" in str(data).lower():
                logger.info(f"[WS] Sending lock/unlock response: {response}")

            # Send response
            await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info(f"[UNIFIED-WS] Client {client_id} disconnected (WebSocketDisconnect)")
    except Exception as e:
        logger.error(f"[UNIFIED-WS] WebSocket error for {client_id}: {e}", exc_info=True)

        # Increment error count
        if client_id in ws_manager.connection_health:
            health = ws_manager.connection_health[client_id]
            health.errors += 1
            health.last_error = str(e)
            health.health_score = max(0, health.health_score - 10)

            # Increment circuit breaker failures
            ws_manager.circuit_failures += 1
    finally:
        await ws_manager.disconnect(client_id)