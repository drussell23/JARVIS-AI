"""
Unified WebSocket Handler - Single endpoint for all WebSocket communication
Consolidates all WebSocket functionality into one clean interface
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging
import json
import asyncio
from typing import Dict, Any, Optional, Set
from datetime import datetime

# Import async pipeline for non-blocking WebSocket operations
from core.async_pipeline import get_async_pipeline, AdvancedAsyncPipeline

logger = logging.getLogger(__name__)

router = APIRouter()

# Active connections management
active_connections: Dict[str, WebSocket] = {}
connection_capabilities: Dict[str, Set[str]] = {}


class UnifiedWebSocketManager:
    """Manages all WebSocket connections and message routing"""

    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}

        # Initialize async pipeline for WebSocket operations
        self.pipeline = get_async_pipeline()
        self._register_pipeline_stages()

        self.handlers = {
            # Voice/JARVIS handlers
            "command": self._handle_voice_command,  # Main command handler
            "voice_command": self._handle_voice_command,
            "jarvis_command": self._handle_voice_command,  # Alias

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
            "subscribe": self._handle_subscribe,
            "unsubscribe": self._handle_unsubscribe,
        }

    def _register_pipeline_stages(self):
        """Register async pipeline stages for WebSocket operations"""

        # Message processing stage
        self.pipeline.register_stage(
            "message_processing",
            self._process_message_async,
            timeout=30.0,
            retry_count=1,
            required=True
        )

        # Command execution stage
        self.pipeline.register_stage(
            "command_execution",
            self._execute_command_async,
            timeout=45.0,
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
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.connections[client_id] = websocket
        connection_capabilities[client_id] = set()
        logger.info(f"Client {client_id} connected to unified WebSocket")
        
        # Send welcome message
        await websocket.send_json({
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat(),
            "available_handlers": list(self.handlers.keys())
        })
        
    def disconnect(self, client_id: str):
        """Remove WebSocket connection"""
        if client_id in self.connections:
            del self.connections[client_id]
        if client_id in connection_capabilities:
            del connection_capabilities[client_id]
        logger.info(f"Client {client_id} disconnected")
        
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
            # Import JARVIS API directly
            from .jarvis_voice_api import jarvis_api
            from pydantic import BaseModel
            
            # Create command object
            class VoiceCommand(BaseModel):
                text: str
            
            command_text = message.get("command", message.get("text", ""))
            command_obj = VoiceCommand(text=command_text)
            
            # Process through JARVIS
            result = await jarvis_api.process_command(command_obj)

            return {
                "type": "response",
                "text": result.get("response", ""),
                "status": result.get("status", "success"),
                "command_type": result.get("command_type", "unknown"),
                "speak": True
            }
            
        except Exception as e:
            logger.error(f"Error processing voice command: {e}")
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


@router.websocket("/ws")
async def unified_websocket_endpoint(websocket: WebSocket):
    """Single unified WebSocket endpoint for all communication"""
    client_id = f"client_{id(websocket)}_{datetime.now().timestamp()}"

    await ws_manager.connect(websocket, client_id)

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()

            # Log incoming command for debugging
            if data.get("type") == "command" or data.get("type") == "voice_command":
                logger.info(f"[WS] Received command: {data.get('text', data.get('command', 'unknown'))}")

            # Handle message
            response = await ws_manager.handle_message(client_id, data)

            # Log outgoing response for debugging lock/unlock
            if "lock" in str(data).lower() or "unlock" in str(data).lower():
                logger.info(f"[WS] Sending lock/unlock response: {response}")

            # Send response
            await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
    finally:
        ws_manager.disconnect(client_id)