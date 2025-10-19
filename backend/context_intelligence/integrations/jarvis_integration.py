"""
JARVIS Integration Module
========================

Integrates the new Context Intelligence System with existing JARVIS components.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime

from ..core.context_manager import get_context_manager, ExecutionState
from ..core.command_queue import CommandPriority
from ..analyzers.intent_analyzer import get_intent_analyzer

logger = logging.getLogger(__name__)


class JARVISContextIntegration:
    """
    Integration layer between JARVIS and the Context Intelligence System
    """
    
    def __init__(self):
        """Initialize JARVIS integration"""
        self.context_manager = get_context_manager()
        self.intent_analyzer = get_intent_analyzer()
        
        # WebSocket connections for real-time updates
        self.websocket_connections: List[Any] = []
        
        # Callback registration
        self._register_callbacks()
        
        # Integration state
        self.initialized = False
        
    async def initialize(self):
        """Initialize the integration"""
        if self.initialized:
            return
            
        logger.info("Initializing JARVIS Context Integration...")
        
        # Initialize context manager
        await self.context_manager.initialize()
        
        self.initialized = True
        logger.info("JARVIS Context Integration initialized")
        
    def _register_callbacks(self):
        """Register callbacks with context manager"""
        # Progress updates
        self.context_manager.add_progress_callback(self._handle_progress)
        
        # State changes
        self.context_manager.add_state_callback(
            ExecutionState.AWAITING_UNLOCK,
            self._handle_awaiting_unlock
        )
        self.context_manager.add_state_callback(
            ExecutionState.EXECUTING,
            self._handle_executing
        )
        self.context_manager.add_state_callback(
            ExecutionState.COMPLETED,
            self._handle_completed
        )
        self.context_manager.add_state_callback(
            ExecutionState.FAILED,
            self._handle_failed
        )
        
    async def process_voice_command(self, command: str, 
                                  voice_context: Optional[Dict[str, Any]] = None,
                                  websocket = None) -> Dict[str, Any]:
        """
        Process a voice command through the context intelligence system
        
        Args:
            command: The voice command text
            voice_context: Optional voice analysis context (tone, urgency, etc.)
            websocket: WebSocket connection for real-time updates
            
        Returns:
            Response dictionary
        """
        if not self.initialized:
            await self.initialize()
            
        # Register WebSocket if provided
        if websocket and websocket not in self.websocket_connections:
            self.websocket_connections.append(websocket)
            
        try:
            # Analyze intent
            full_context = {
                "voice_context": voice_context or {},
                "timestamp": datetime.now().isoformat(),
                "source": "voice"
            }
            
            intent = await self.intent_analyzer.analyze(command, full_context)

            # Check if this is a predictive query - handle differently
            if intent.type.value == "predictive_query":
                logger.info(f"[JARVIS-INTEGRATION] Routing predictive query to specialized handler")
                from ..handlers.context_aware_handler import get_context_aware_handler
                handler = get_context_aware_handler()
                result = await handler.handle_command_with_context(
                    command,
                    execute_callback=None,
                    intent_type="predictive_query"
                )
                # Add intent information
                result["intent"] = {
                    "type": intent.type.value,
                    "entities": intent.entities,
                    "confidence": intent.confidence,
                    "requires_screen": intent.requires_screen
                }
                return result

            # Determine priority based on voice context
            priority = self._determine_priority(voice_context)

            # Process through context manager
            result = await self.context_manager.process_command(
                command_text=command,
                intent={
                    "type": intent.type.value,
                    "confidence": intent.confidence,
                    "entities": intent.entities,
                    "requires_screen": intent.requires_screen
                },
                priority=priority,
                metadata={
                    "voice_tone": voice_context.get("tone") if voice_context else None,
                    "urgency": voice_context.get("urgency") if voice_context else "normal",
                    "original_metadata": intent.metadata
                }
            )
            
            # Add intent information to result
            result["intent"] = {
                "type": intent.type.value,
                "entities": intent.entities,
                "confidence": intent.confidence,
                "requires_screen": intent.requires_screen
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing voice command: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": "I encountered an error processing your request."
            }
            
    def _determine_priority(self, voice_context: Optional[Dict[str, Any]]) -> CommandPriority:
        """Determine command priority from voice context"""
        if not voice_context:
            return CommandPriority.NORMAL
            
        urgency = voice_context.get("urgency", "normal")
        
        if urgency == "urgent" or voice_context.get("tone") == "stressed":
            return CommandPriority.URGENT
        elif urgency == "high":
            return CommandPriority.HIGH
        elif urgency == "low" or voice_context.get("tone") == "relaxed":
            return CommandPriority.LOW
        else:
            return CommandPriority.NORMAL
            
    async def _handle_progress(self, context, message: str):
        """Handle progress updates"""
        # Send to WebSocket connections
        update = {
            "type": "progress",
            "command_id": context.command.command_id,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        await self._broadcast_update(update)
        
    async def _handle_awaiting_unlock(self, context, old_state, new_state):
        """Handle awaiting unlock state"""
        update = {
            "type": "state_change",
            "command_id": context.command.command_id,
            "state": "awaiting_unlock",
            "message": "Your screen is locked. Preparing to unlock...",
            "timestamp": datetime.now().isoformat()
        }
        
        await self._broadcast_update(update)
        
    async def _handle_executing(self, context, old_state, new_state):
        """Handle executing state"""
        update = {
            "type": "state_change",
            "command_id": context.command.command_id,
            "state": "executing",
            "message": f"Executing: {context.command.original_text}",
            "timestamp": datetime.now().isoformat()
        }
        
        await self._broadcast_update(update)
        
    async def _handle_completed(self, context, old_state, new_state):
        """Handle completed state"""
        update = {
            "type": "completion",
            "command_id": context.command.command_id,
            "success": True,
            "message": "Command completed successfully",
            "execution_time": context.execution_time,
            "timestamp": datetime.now().isoformat()
        }
        
        await self._broadcast_update(update)
        
    async def _handle_failed(self, context, old_state, new_state):
        """Handle failed state"""
        error_msg = "; ".join(context.error_messages) if context.error_messages else "Unknown error"
        
        update = {
            "type": "completion",
            "command_id": context.command.command_id,
            "success": False,
            "message": f"Command failed: {error_msg}",
            "execution_time": context.execution_time,
            "timestamp": datetime.now().isoformat()
        }
        
        await self._broadcast_update(update)
        
    async def _broadcast_update(self, update: Dict[str, Any]):
        """Broadcast update to all connected WebSockets"""
        # Remove disconnected connections
        active_connections = []
        
        for ws in self.websocket_connections:
            try:
                if hasattr(ws, 'send_json'):
                    await ws.send_json(update)
                elif hasattr(ws, 'send'):
                    import json
                    await ws.send(json.dumps(update))
                active_connections.append(ws)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                
        self.websocket_connections = active_connections
        
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        queue_stats = await self.context_manager.command_queue.get_statistics()
        pending = await self.context_manager.command_queue.get_pending_commands(limit=10)
        
        return {
            "statistics": queue_stats,
            "pending_commands": [
                {
                    "id": cmd.command_id,
                    "command": cmd.original_text,
                    "priority": cmd.priority.name,
                    "status": cmd.status.value,
                    "created_at": cmd.created_at.isoformat(),
                    "age_minutes": cmd.age_minutes
                }
                for cmd in pending
            ]
        }
        
    async def cancel_command(self, command_id: str) -> Dict[str, Any]:
        """Cancel a pending command"""
        try:
            await self.context_manager.command_queue.cancel_command(
                command_id, 
                reason="User requested cancellation"
            )
            
            return {
                "success": True,
                "message": f"Command {command_id} cancelled"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return await self.context_manager.get_status()
        
    def disconnect_websocket(self, websocket):
        """Remove a WebSocket connection"""
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)
            
    async def shutdown(self):
        """Shutdown the integration"""
        if self.initialized:
            await self.context_manager.shutdown()
            self.initialized = False


# Global integration instance
_integration: Optional[JARVISContextIntegration] = None


def get_jarvis_integration() -> JARVISContextIntegration:
    """Get or create JARVIS integration"""
    global _integration
    if _integration is None:
        _integration = JARVISContextIntegration()
    return _integration


# API endpoint handlers for easy integration

async def handle_voice_command(command: str, voice_context: Dict[str, Any] = None,
                             websocket = None) -> Dict[str, Any]:
    """Handle voice command with context awareness"""
    integration = get_jarvis_integration()
    return await integration.process_voice_command(command, voice_context, websocket)


async def handle_queue_status() -> Dict[str, Any]:
    """Get queue status"""
    integration = get_jarvis_integration()
    return await integration.get_queue_status()


async def handle_cancel_command(command_id: str) -> Dict[str, Any]:
    """Cancel a command"""
    integration = get_jarvis_integration()
    return await integration.cancel_command(command_id)


async def handle_system_status() -> Dict[str, Any]:
    """Get system status"""
    integration = get_jarvis_integration()
    return await integration.get_system_status()