"""
Unified Vision WebSocket Handler
Consolidates all vision WebSocket endpoints into a single, non-conflicting handler
Works with TypeScript WebSocket Router for seamless integration
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict
import traceback

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

# Import all vision systems
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from vision.vision_system_v2 import VisionSystemV2
except ImportError:
    VisionSystemV2 = None

try:
    from vision.dynamic_response_composer import DynamicResponseComposer
except ImportError:
    DynamicResponseComposer = None

try:
    from vision.natural_responses import NaturalResponseGenerator
except ImportError:
    NaturalResponseGenerator = None

try:
    from vision.performance_optimizer import PerformanceOptimizer
except ImportError:
    PerformanceOptimizer = None

# Import existing modules with correct paths
try:
    from vision.workspace_analyzer import WorkspaceAnalyzer
except ImportError:
    WorkspaceAnalyzer = None

try:
    from autonomy.action_executor import ActionExecutor as AutonomousActionExecutor
except ImportError:
    AutonomousActionExecutor = None

logger = logging.getLogger(__name__)

@dataclass
class VisionContext:
    """Context for vision operations"""

    client_id: str
    capabilities: List[str]
    metadata: Dict[str, Any]
    monitoring_active: bool = False
    monitoring_interval: float = 2.0
    autonomous_mode: bool = False

class MessageRouter:
    """Dynamic message routing system"""

    def __init__(self):
        self.handlers: Dict[str, Callable] = {}
        self.middleware: List[Callable] = []
        self.fallback_handler: Optional[Callable] = None

    def register(self, message_type: str, handler: Callable):
        """Register a handler for a specific message type"""
        self.handlers[message_type] = handler

    def use(self, middleware: Callable):
        """Add middleware to the processing chain"""
        self.middleware.append(middleware)

    def set_fallback(self, handler: Callable):
        """Set fallback handler for unknown message types"""
        self.fallback_handler = handler

    async def route(self, message: Dict[str, Any], context: VisionContext) -> Any:
        """Route message to appropriate handler"""
        message_type = message.get("type", "unknown")

        # Process through middleware
        for mw in self.middleware:
            message = await mw(message, context)
            if message is None:
                return None

        # Find and execute handler
        handler = self.handlers.get(message_type, self.fallback_handler)
        if handler:
            return await handler(message, context)
        else:
            raise ValueError(f"No handler for message type: {message_type}")

class UnifiedVisionHandler:
    """Unified handler for all vision WebSocket operations"""

    def __init__(self):
        # Initialize available components
        self.vision_system = VisionSystemV2() if VisionSystemV2 else None
        self.response_composer = (
            DynamicResponseComposer() if DynamicResponseComposer else None
        )
        self.natural_responder = (
            NaturalResponseGenerator() if NaturalResponseGenerator else None
        )
        self.performance_optimizer = (
            PerformanceOptimizer() if PerformanceOptimizer else None
        )

        # These modules don't exist, so set to None
        self.claude_ai = None
        self.notification_system = None
        self.pattern_learner = None

        # Initialize existing modules
        self.action_executor = (
            AutonomousActionExecutor() if AutonomousActionExecutor else None
        )
        self.workspace_analyzer = WorkspaceAnalyzer() if WorkspaceAnalyzer else None

        self.router = MessageRouter()
        self.contexts: Dict[str, VisionContext] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}

        self._setup_routes()
        self._setup_middleware()

    def _setup_routes(self):
        """Setup message routing"""
        # Configuration messages
        self.router.register("set_monitoring_interval", self.handle_monitoring_interval)
        self.router.register("set_autonomous_mode", self.handle_autonomous_mode)

        # Vision operations
        self.router.register(
            "request_workspace_analysis", self.handle_workspace_analysis
        )
        self.router.register("vision_command", self.handle_vision_command)
        self.router.register("claude_vision", self.handle_claude_vision)

        # Action execution
        self.router.register("execute_action", self.handle_action_execution)
        self.router.register("autonomous_action", self.handle_autonomous_action)

        # Learning and patterns
        self.router.register("learn_pattern", self.handle_pattern_learning)
        self.router.register("get_learned_patterns", self.handle_get_patterns)

        # System status
        self.router.register("get_status", self.handle_get_status)
        self.router.register("ping", self.handle_ping)

        # Set fallback
        self.router.set_fallback(self.handle_unknown_message)

    def _setup_middleware(self):
        """Setup processing middleware"""

        async def logging_middleware(
            message: Dict[str, Any], context: VisionContext
        ) -> Dict[str, Any]:
            """Log all incoming messages"""
            logger.debug(
                f"Received message from {context.client_id}: {message.get('type', 'unknown')}"
            )
            return message

        async def validation_middleware(
            message: Dict[str, Any], context: VisionContext
        ) -> Dict[str, Any]:
            """Validate message structure"""
            if not isinstance(message, dict):
                raise ValueError("Message must be a dictionary")
            if "type" not in message:
                raise ValueError("Message must have a 'type' field")
            return message

        async def performance_middleware(
            message: Dict[str, Any], context: VisionContext
        ) -> Dict[str, Any]:
            """Track performance metrics"""
            message["_received_at"] = datetime.now().isoformat()
            return message

        self.router.use(logging_middleware)
        self.router.use(validation_middleware)
        self.router.use(performance_middleware)

    async def handle_monitoring_interval(
        self, message: Dict[str, Any], context: VisionContext
    ) -> Dict[str, Any]:
        """Handle monitoring interval update"""
        interval = message.get("interval", 2.0)
        context.monitoring_interval = max(0.5, min(10.0, float(interval)))

        # Restart monitoring if active
        if context.monitoring_active:
            await self._restart_monitoring(context)

        return {
            "type": "config_updated",
            "monitoring_interval": context.monitoring_interval,
            "status": "success",
        }

    async def handle_autonomous_mode(
        self, message: Dict[str, Any], context: VisionContext
    ) -> Dict[str, Any]:
        """Handle autonomous mode toggle"""
        enabled = message.get("enabled", False)
        context.autonomous_mode = bool(enabled)

        return {
            "type": "config_updated",
            "autonomous_mode": context.autonomous_mode,
            "status": "success",
        }

    async def handle_workspace_analysis(
        self, message: Dict[str, Any], context: VisionContext
    ) -> Dict[str, Any]:
        """Handle workspace analysis request"""
        try:
            # Capture current screen
            screenshot = await self.vision_system.capture_screenshot()

            # Analyze workspace
            analysis = await self.workspace_analyzer.analyze_workspace()

            # Get Claude's insights if available
            claude_insights = None
            if self.claude_ai.is_available():
                claude_insights = await self.claude_ai.analyze_workspace(screenshot)

            # Compose natural response
            natural_response = self.natural_responder.generate_workspace_summary(
                analysis, claude_insights
            )

            return {
                "type": "workspace_analysis",
                "timestamp": datetime.now().isoformat(),
                "analysis": {
                    "focused_task": analysis.focused_task,
                    "context": analysis.workspace_context,
                    "notifications": analysis.important_notifications,
                    "suggestions": analysis.suggestions,
                    "confidence": analysis.confidence,
                    "natural_response": natural_response,
                    "claude_insights": claude_insights,
                },
            }
        except Exception as e:
            logger.error(f"Workspace analysis error: {e}")
            return {
                "type": "error",
                "message": f"Failed to analyze workspace: {str(e)}",
            }

    async def handle_vision_command(
        self, message: Dict[str, Any], context: VisionContext
    ) -> Dict[str, Any]:
        """Handle general vision commands"""
        command = message.get("command", "")

        try:
            # Process command through vision system
            result = await self.vision_system.process_command(command)

            # Enhance with natural language
            enhanced_result = self.response_composer.compose_response(command, result)

            return {
                "type": "vision_result",
                "command": command,
                "result": enhanced_result,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Vision command error: {e}")
            return {
                "type": "error",
                "message": f"Failed to process vision command: {str(e)}",
            }

    async def handle_claude_vision(
        self, message: Dict[str, Any], context: VisionContext
    ) -> Dict[str, Any]:
        """Handle Claude-specific vision requests"""
        query = message.get("query", "")

        try:
            if not self.claude_ai.is_available():
                return {"type": "error", "message": "Claude AI is not available"}

            # Capture and analyze with Claude
            screenshot = await self.vision_system.capture_screenshot()
            claude_response = await self.claude_ai.analyze_with_query(screenshot, query)

            return {
                "type": "claude_vision_result",
                "query": query,
                "response": claude_response,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Claude vision error: {e}")
            return {
                "type": "error",
                "message": f"Failed to process Claude vision: {str(e)}",
            }

    async def handle_action_execution(
        self, message: Dict[str, Any], context: VisionContext
    ) -> Dict[str, Any]:
        """Handle action execution requests"""
        action = message.get("action", {})

        try:
            # Validate action
            if not action or "type" not in action:
                raise ValueError("Invalid action format")

            # Execute action
            result = await self.action_executor.execute(action)

            # Learn from execution if successful
            if result.success:
                await self.pattern_learner.learn_from_action(action, result)

            return {
                "type": "action_result",
                "action": action,
                "result": asdict(result),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Action execution error: {e}")
            return {"type": "error", "message": f"Failed to execute action: {str(e)}"}

    async def handle_autonomous_action(
        self, message: Dict[str, Any], context: VisionContext
    ) -> Dict[str, Any]:
        """Handle autonomous action requests"""
        if not context.autonomous_mode:
            return {"type": "error", "message": "Autonomous mode is not enabled"}

        try:
            # Analyze current state and determine action
            screenshot = await self.vision_system.capture_screenshot()
            workspace_state = await self.workspace_analyzer.analyze_workspace()

            # Get AI-recommended action
            recommended_action = await self.claude_ai.suggest_action(
                screenshot, workspace_state
            )

            if recommended_action:
                # Execute the action
                result = await self.action_executor.execute(recommended_action)

                return {
                    "type": "autonomous_action_result",
                    "action": recommended_action,
                    "result": asdict(result),
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                return {
                    "type": "autonomous_action_result",
                    "message": "No action recommended at this time",
                    "timestamp": datetime.now().isoformat(),
                }
        except Exception as e:
            logger.error(f"Autonomous action error: {e}")
            return {
                "type": "error",
                "message": f"Failed to execute autonomous action: {str(e)}",
            }

    async def handle_pattern_learning(
        self, message: Dict[str, Any], context: VisionContext
    ) -> Dict[str, Any]:
        """Handle pattern learning requests"""
        pattern_data = message.get("pattern", {})

        try:
            # Learn the pattern
            learning_result = await self.pattern_learner.learn_pattern(pattern_data)

            return {
                "type": "pattern_learned",
                "pattern": pattern_data,
                "result": learning_result,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Pattern learning error: {e}")
            return {"type": "error", "message": f"Failed to learn pattern: {str(e)}"}

    async def handle_get_patterns(
        self, message: Dict[str, Any], context: VisionContext
    ) -> Dict[str, Any]:
        """Get learned patterns"""
        try:
            patterns = await self.pattern_learner.get_patterns()

            return {
                "type": "learned_patterns",
                "patterns": patterns,
                "count": len(patterns),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Get patterns error: {e}")
            return {"type": "error", "message": f"Failed to get patterns: {str(e)}"}

    async def handle_get_status(
        self, message: Dict[str, Any], context: VisionContext
    ) -> Dict[str, Any]:
        """Get system status"""
        try:
            status = {
                "vision_system": self.vision_system.get_status(),
                "claude_ai": {
                    "available": self.claude_ai.is_available(),
                    "model": "Claude Opus 4",
                },
                "monitoring": {
                    "active": context.monitoring_active,
                    "interval": context.monitoring_interval,
                },
                "autonomous_mode": context.autonomous_mode,
                "performance": self.performance_optimizer.get_metrics(),
                "learned_patterns": await self.pattern_learner.get_pattern_count(),
            }

            return {
                "type": "system_status",
                "status": status,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Get status error: {e}")
            return {"type": "error", "message": f"Failed to get status: {str(e)}"}

    async def handle_ping(
        self, message: Dict[str, Any], context: VisionContext
    ) -> Dict[str, Any]:
        """Handle ping messages"""
        return {
            "type": "pong",
            "timestamp": datetime.now().isoformat(),
            "client_id": context.client_id,
        }

    async def handle_unknown_message(
        self, message: Dict[str, Any], context: VisionContext
    ) -> Dict[str, Any]:
        """Handle unknown message types"""
        logger.warning(f"Unknown message type: {message.get('type', 'undefined')}")
        return {
            "type": "error",
            "message": f"Unknown message type: {message.get('type', 'undefined')}",
            "supported_types": list(self.router.handlers.keys()),
        }

    async def _restart_monitoring(self, context: VisionContext):
        """Restart monitoring for a client"""
        # Cancel existing task
        if context.client_id in self.monitoring_tasks:
            self.monitoring_tasks[context.client_id].cancel()

        # Start new monitoring task
        task = asyncio.create_task(self._monitor_workspace(context))
        self.monitoring_tasks[context.client_id] = task

    async def _monitor_workspace(self, context: VisionContext):
        """Monitor workspace continuously"""
        while context.monitoring_active:
            try:
                # Analyze workspace
                analysis = await self.workspace_analyzer.analyze_workspace()

                # Check for important changes
                if analysis.important_notifications:
                    # Send notification update
                    await self._send_to_client(
                        context.client_id,
                        {
                            "type": "workspace_update",
                            "notifications": analysis.important_notifications,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )

                # Sleep for interval
                await asyncio.sleep(context.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(context.monitoring_interval)

    async def _send_to_client(self, client_id: str, message: Dict[str, Any]):
        """Send message to a specific client"""
        # This would be implemented by the WebSocket connection manager
        # For now, just log it
        logger.info(f"Would send to {client_id}: {message}")

    def create_context(
        self, client_id: str, capabilities: List[str], metadata: Dict[str, Any]
    ) -> VisionContext:
        """Create a new client context"""
        context = VisionContext(
            client_id=client_id, capabilities=capabilities, metadata=metadata
        )
        self.contexts[client_id] = context
        return context

    def remove_context(self, client_id: str):
        """Remove client context and cleanup"""
        if client_id in self.contexts:
            # Cancel monitoring task
            if client_id in self.monitoring_tasks:
                self.monitoring_tasks[client_id].cancel()
                del self.monitoring_tasks[client_id]

            del self.contexts[client_id]

    async def handle_websocket_message(
        self, message: Dict[str, Any], **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Main entry point for WebSocket messages from TypeScript router"""
        try:
            # Extract context from kwargs
            client_id = kwargs.get("client_id", "unknown")
            context = self.contexts.get(client_id)

            if not context:
                # Create temporary context
                context = VisionContext(
                    client_id=client_id,
                    capabilities=kwargs.get("capabilities", []),
                    metadata=kwargs.get("context", {}),
                )

            # Route the message
            result = await self.router.route(message, context)

            return result

        except Exception as e:
            logger.error(
                f"WebSocket message handling error: {e}\n{traceback.format_exc()}"
            )
            return {
                "type": "error",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }

# Global handler instance
unified_handler = UnifiedVisionHandler()

# Export handler functions for TypeScript bridge
async def handle_monitoring_interval(
    message: Dict[str, Any], **kwargs
) -> Dict[str, Any]:
    """Handle monitoring interval updates"""
    return await unified_handler.handle_websocket_message(message, **kwargs)

async def handle_workspace_analysis(
    message: Dict[str, Any], **kwargs
) -> Dict[str, Any]:
    """Handle workspace analysis requests"""
    return await unified_handler.handle_websocket_message(message, **kwargs)

async def handle_action_execution(message: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Handle action execution requests"""
    return await unified_handler.handle_websocket_message(message, **kwargs)

async def handle_vision_command(message: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Handle vision commands"""
    return await unified_handler.handle_websocket_message(message, **kwargs)

async def handle_claude_vision(message: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Handle Claude vision requests"""
    return await unified_handler.handle_websocket_message(message, **kwargs)

async def handle_autonomous_action(message: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Handle autonomous actions"""
    return await unified_handler.handle_websocket_message(message, **kwargs)

async def handle_pattern_learning(message: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Handle pattern learning"""
    return await unified_handler.handle_websocket_message(message, **kwargs)
