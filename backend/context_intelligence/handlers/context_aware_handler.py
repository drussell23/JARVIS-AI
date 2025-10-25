"""
Context-Aware Command Handler for JARVIS
=======================================

Handles commands with full context awareness, including screen lock state
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

from context_intelligence.detectors.screen_lock_detector import get_screen_lock_detector
from context_intelligence.analyzers.intent_analyzer import IntentType
from context_intelligence.handlers.predictive_query_handler import (
    get_predictive_handler,
    initialize_predictive_handler,
    PredictiveQueryRequest
)
from context_intelligence.handlers.action_query_handler import (
    get_action_query_handler,
    initialize_action_query_handler,
    ActionQueryResponse
)

logger = logging.getLogger(__name__)


class ContextAwareCommandHandler:
    """
    Handles commands with context awareness
    """

    def __init__(self):
        self.screen_lock_detector = get_screen_lock_detector()
        self.execution_steps = []
        self.predictive_handler = None  # Lazy initialize
        self.action_handler = None  # Lazy initialize

    async def handle_command_with_context(
        self, command: str, execute_callback=None, intent_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle a command with full context awareness

        Args:
            command: The command to execute
            execute_callback: Callback to execute the actual command
            intent_type: Optional intent type for routing

        Returns:
            Response dict with status and messages
        """
        logger.info(f"[CONTEXT AWARE] Starting context-aware handling for: {command}")
        self.execution_steps = []
        response = {
            "success": True,
            "command": command,
            "messages": [],
            "steps_taken": [],
            "context": {},
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Check if this is an action query (takes priority)
            if intent_type == "action_query" or self._is_action_query(command):
                logger.info("[CONTEXT AWARE] Detected action query - routing to action handler")
                return await self._handle_action_query(command)

            # Check if this is a predictive query
            if intent_type == "predictive_query" or self._is_predictive_query(command):
                logger.info("[CONTEXT AWARE] Detected predictive query - routing to predictive handler")
                return await self._handle_predictive_query(command)

            # Step 1: Get system context
            logger.info("[CONTEXT AWARE] Getting system context...")
            system_context = await self._get_system_context()
            response["context"] = system_context
            logger.info(
                f"[CONTEXT AWARE] System context: screen_locked={system_context.get('screen_locked', False)}"
            )

            # Step 2: Check screen lock context
            is_locked = system_context.get("screen_locked", False)
            logger.info(
                f"[CONTEXT AWARE] Screen is {'LOCKED' if is_locked else 'UNLOCKED'}"
            )

            if is_locked:
                self._add_step("Detected locked screen", {"screen_locked": True})
                logger.warning(
                    f"[CONTEXT AWARE] ⚠️  SCREEN IS LOCKED - Command requires unlocked screen"
                )

                # Check if command requires unlocked screen
                screen_context = await self.screen_lock_detector.check_screen_context(
                    command
                )

                if screen_context["requires_unlock"]:
                    # IMPORTANT: Speak to user FIRST about screen being locked
                    unlock_notification = screen_context["unlock_message"]
                    self._add_step("Screen unlock required", screen_context)

                    # Log prominently for debugging
                    logger.warning(f"[CONTEXT AWARE] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                    logger.warning(
                        f"[CONTEXT AWARE] 🔓 SCREEN LOCKED - UNLOCK REQUIRED"
                    )
                    logger.warning(f"[CONTEXT AWARE] 📝 Command: {command}")
                    logger.warning(
                        f"[CONTEXT AWARE] 📢 Unlock Message: '{unlock_notification}'"
                    )
                    logger.warning(f"[CONTEXT AWARE] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

                    # Speak the unlock message immediately with emphasis
                    logger.warning(
                        f"[CONTEXT AWARE] 🎤 Speaking unlock notification NOW..."
                    )
                    await self._speak_message(unlock_notification, priority="high")
                    logger.warning(f"[CONTEXT AWARE] ✅ Unlock notification spoken")

                    # Add a longer delay to ensure user hears the message before unlock happens
                    logger.info(
                        f"[CONTEXT AWARE] ⏱️  Waiting 3 seconds for user to hear notification..."
                    )
                    await asyncio.sleep(3.0)
                    logger.info(
                        f"[CONTEXT AWARE] ⏱️  Wait complete, proceeding with unlock..."
                    )

                    # Now perform the actual unlock
                    logger.info(f"[CONTEXT AWARE] 🔓 Now unlocking screen...")
                    unlock_success, unlock_message = (
                        await self.screen_lock_detector.handle_screen_lock_context(
                            command
                        )
                    )

                    if unlock_success:
                        self._add_step(
                            "Screen unlocked successfully", {"unlocked": True}
                        )
                        logger.info(f"[CONTEXT AWARE] ✅ Screen unlocked successfully")
                        # Don't add unlock message to response since we already spoke it
                    else:
                        self._add_step(
                            "Screen unlock failed", {"error": unlock_message}
                        )
                        logger.error(
                            f"[CONTEXT AWARE] ❌ Screen unlock failed: {unlock_message}"
                        )
                        response["success"] = False
                        response["messages"].append(
                            unlock_message or "Failed to unlock screen"
                        )
                        return self._finalize_response(response)

            # Step 3: Execute the actual command
            if execute_callback:
                self._add_step("Executing command", {"command": command})

                try:
                    # Execute with context
                    exec_result = await execute_callback(
                        command, context=system_context
                    )

                    if isinstance(exec_result, dict):
                        if exec_result.get("success", True):
                            self._add_step("Command executed successfully", exec_result)
                            # Only add message if one was provided (not None)
                            message = exec_result.get("message")
                            if message:
                                response["messages"].append(message)
                            elif exec_result.get("task_started"):
                                # For document creation, provide appropriate message
                                topic = exec_result.get("topic", "the requested topic")
                                response["messages"].append(
                                    f"I'm creating an essay about {topic} for you, Sir."
                                )
                            else:
                                response["messages"].append(
                                    "Command completed successfully"
                                )
                            response["result"] = exec_result
                        else:
                            self._add_step("Command execution failed", exec_result)
                            response["success"] = False
                            response["messages"].append(
                                exec_result.get("message", "Command failed")
                            )
                    else:
                        # Simple success
                        self._add_step(
                            "Command completed", {"result": str(exec_result)}
                        )
                        response["messages"].append("Command completed successfully")

                except Exception as e:
                    self._add_step("Command execution error", {"error": str(e)})
                    response["success"] = False
                    response["messages"].append(f"Error executing command: {str(e)}")

            # Step 4: Provide confirmation
            if response["success"]:
                confirmation = self._generate_confirmation(
                    command, self.execution_steps
                )
                response["messages"].append(confirmation)

        except Exception as e:
            logger.error(f"Error in context-aware command handling: {e}")
            response["success"] = False
            response["messages"].append(f"An error occurred: {str(e)}")
            self._add_step("Error occurred", {"error": str(e)})

        return self._finalize_response(response)

    def _is_action_query(self, command: str) -> bool:
        """Check if command is an action query"""
        action_keywords = [
            "switch to space", "close", "fix", "run tests", "run build",
            "move", "focus", "launch", "quit", "restart", "open http",
            "fix the", "fix it", "close it", "close that"
        ]
        command_lower = command.lower()
        return any(keyword in command_lower for keyword in action_keywords)

    async def _handle_action_query(self, command: str) -> Dict[str, Any]:
        """Handle an action query using the action handler"""
        try:
            # Lazy initialize action handler
            if self.action_handler is None:
                # Need implicit resolver for reference resolution!
                from core.nlp.implicit_reference_resolver import get_implicit_resolver

                self.action_handler = get_action_query_handler()
                if self.action_handler is None:
                    implicit_resolver = get_implicit_resolver()
                    self.action_handler = initialize_action_query_handler(
                        context_graph=None,  # Could integrate context graph here
                        implicit_resolver=implicit_resolver  # ⭐ KEY INTEGRATION!
                    )

            logger.info(f"[CONTEXT AWARE] Processing action query: {command}")

            # Execute action query
            result: ActionQueryResponse = await self.action_handler.handle_action_query(
                command,
                context={}
            )

            # Format response for JARVIS
            response = {
                "success": result.success,
                "command": command,
                "messages": [result.message] if result.message else [],
                "steps_taken": [
                    {
                        "step": 1,
                        "description": f"Executed action: {result.action_type}",
                        "details": {
                            "action_type": result.action_type,
                            "requires_confirmation": result.requires_confirmation,
                            "execution_status": result.execution_result.status.value if result.execution_result else "none"
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                ],
                "context": {
                    "action_type": result.action_type,
                    "resolved_references": result.metadata.get("resolved_references", {}),
                    "safety_level": result.metadata.get("safety_level", "unknown")
                },
                "timestamp": result.timestamp.isoformat(),
                "result": {
                    "execution": result.execution_result.__dict__ if result.execution_result else None,
                    "plan": {
                        "steps": [s.__dict__ for s in result.plan.steps] if result.plan else [],
                        "safety_level": result.plan.safety_level.value if result.plan else "unknown"
                    } if result.plan else None
                }
            }

            if response["success"]:
                response["summary"] = result.message
            else:
                response["summary"] = f"Action failed: {result.message}"

            logger.info(f"[CONTEXT AWARE] Action query completed: success={result.success}")

            return response

        except Exception as e:
            logger.error(f"[CONTEXT AWARE] Error handling action query: {e}", exc_info=True)
            return {
                "success": False,
                "command": command,
                "messages": [f"Error processing action: {str(e)}"],
                "steps_taken": [],
                "context": {},
                "timestamp": datetime.now().isoformat(),
                "summary": f"Error: {str(e)}"
            }

    def _is_predictive_query(self, command: str) -> bool:
        """Check if command is a predictive/analytical query"""
        predictive_keywords = [
            "making progress", "am i doing", "my progress",
            "what should i", "what to do next", "next steps",
            "any bugs", "any errors", "any issues", "potential bugs",
            "explain", "what does", "how does",
            "what patterns", "analyze patterns",
            "improve my workflow", "optimize", "work more efficiently",
            "code quality"
        ]
        command_lower = command.lower()
        return any(keyword in command_lower for keyword in predictive_keywords)

    async def _handle_predictive_query(self, command: str) -> Dict[str, Any]:
        """Handle a predictive/analytical query"""
        try:
            # Lazy initialize predictive handler
            if self.predictive_handler is None:
                self.predictive_handler = get_predictive_handler()
                if self.predictive_handler is None:
                    self.predictive_handler = initialize_predictive_handler()

            logger.info(f"[CONTEXT AWARE] Processing predictive query: {command}")

            # Determine if visual analysis is needed
            use_vision = any(keyword in command.lower() for keyword in ["explain", "code", "this", "that"])

            # Create request
            request = PredictiveQueryRequest(
                query=command,
                use_vision=use_vision,
                capture_screen=use_vision,
                repo_path=".",
                additional_context={}
            )

            # Execute query
            result = await self.predictive_handler.handle_query(request)

            # Format response
            response = {
                "success": result.success,
                "command": command,
                "messages": [result.response_text] if result.response_text else [],
                "steps_taken": [
                    {
                        "step": 1,
                        "description": "Analyzed query with predictive engine",
                        "details": {
                            "query_type": result.analytics.query_type.value if result.analytics else "unknown",
                            "confidence": result.confidence,
                            "used_vision": result.vision_analysis is not None
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                ],
                "context": {
                    "query_type": result.analytics.query_type.value if result.analytics else "unknown",
                    "confidence": result.confidence,
                    "insights": result.analytics.insights if result.analytics else []
                },
                "timestamp": result.timestamp.isoformat(),
                "result": {
                    "analytics": {
                        "metrics": result.analytics.metrics.__dict__ if result.analytics and result.analytics.metrics else None,
                        "bug_patterns": [bp.__dict__ for bp in result.analytics.bug_patterns] if result.analytics else [],
                        "recommendations": [rec.__dict__ for rec in result.analytics.recommendations] if result.analytics else []
                    } if result.analytics else None,
                    "vision_analysis": result.vision_analysis
                }
            }

            if response["success"]:
                response["summary"] = result.response_text if result.response_text else "Analysis complete"
            else:
                response["summary"] = "Predictive query failed"

            logger.info(f"[CONTEXT AWARE] Predictive query completed: success={result.success}, confidence={result.confidence:.2%}")

            return response

        except Exception as e:
            logger.error(f"[CONTEXT AWARE] Error handling predictive query: {e}", exc_info=True)
            return {
                "success": False,
                "command": command,
                "messages": [f"Error processing predictive query: {str(e)}"],
                "steps_taken": [],
                "context": {},
                "timestamp": datetime.now().isoformat(),
                "summary": f"Error: {str(e)}"
            }

    async def _get_system_context(self) -> Dict[str, Any]:
        """Get current system context"""
        logger.info("[CONTEXT AWARE] _get_system_context called")
        # Get key system states
        states_to_check = [
            "screen_locked",
            "active_apps",
            "network_connected",
            "active_window",
        ]

        # Get context without system_monitor (simplified for now)
        logger.info("[CONTEXT AWARE] Checking if screen is locked...")
        is_locked = await self.screen_lock_detector.is_screen_locked()
        logger.info(f"[CONTEXT AWARE] Screen lock detector returned: {is_locked}")
        context = {
            "screen_locked": is_locked,
            "active_apps": [],
            "network_connected": True,
            "active_window": None,
        }

        # Add summary
        context["summary"] = {
            "screen_accessible": not context.get("screen_locked", True),
            "apps_running": len(context.get("active_apps", [])),
            "network_available": context.get("network_connected", False),
        }

        return context

    async def _speak_message(self, message: str, priority: str = "normal"):
        """Speak a message immediately using JARVIS voice through WebSocket and macOS say"""
        try:
            logger.info(
                f"[CONTEXT AWARE] 📢 Speaking message (priority={priority}): {message}"
            )

            # Use macOS say command FIRST (more reliable, especially when screen is locked)
            # This ensures the user hears the message even if WebSocket fails
            say_success = False
            try:
                import subprocess

                # Run say command synchronously for immediate feedback
                # Use slower speech rate for unlock messages to ensure clarity
                speech_rate = (
                    "160" if priority == "high" else "190"
                )  # Slower for unlock messages
                process = await asyncio.create_subprocess_exec(
                    "say",
                    "-v",
                    "Daniel",
                    "-r",
                    speech_rate,
                    message,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                returncode = await process.wait()
                say_success = returncode == 0
                if say_success:
                    logger.info(
                        f"[CONTEXT AWARE] ✅ Spoke via macOS say command successfully"
                    )
                else:
                    logger.warning(
                        f"[CONTEXT AWARE] ⚠️  Say command returned non-zero: {returncode}"
                    )
            except Exception as e:
                logger.error(f"[CONTEXT AWARE] ❌ Say command failed: {e}")

            # Also try WebSocket broadcast as secondary method
            try:
                from api.unified_websocket import broadcast_message

                # Broadcast the notification via WebSocket
                await broadcast_message(
                    {"type": "speak", "text": message, "priority": priority}
                )
                logger.info(f"[CONTEXT AWARE] 📡 Broadcasted via WebSocket")
            except Exception as e:
                logger.debug(
                    f"[CONTEXT AWARE] WebSocket broadcast failed (this is OK if no clients): {e}"
                )

            # If both methods failed, log an error
            if not say_success:
                logger.error(
                    f"[CONTEXT AWARE] ⚠️  WARNING: Could not speak message reliably!"
                )

        except Exception as e:
            logger.error(f"[CONTEXT AWARE] ❌ Failed to speak message: {e}")

    def _add_step(self, description: str, details: Dict[str, Any]):
        """Add an execution step for tracking"""
        self.execution_steps.append(
            {
                "step": len(self.execution_steps) + 1,
                "description": description,
                "details": details,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def _generate_confirmation(self, command: str, steps: List[Dict]) -> str:
        """Generate accurate confirmation message based on actual execution results"""
        if not steps:
            return "Command completed."

        # Check execution results from steps
        executed_successfully = False
        for step in steps:
            if step["description"] == "Command executed successfully":
                executed_successfully = True
                if "result" in step.get("details", {}):
                    result = step["details"]["result"]
                    if isinstance(result, dict) and "message" in result:
                        return result["message"]

        # If command was executed, don't generate generic message
        if executed_successfully:
            return ""  # Let the execution result message be used instead

        # Fallback for non-execution confirmations
        return "Task completed."

    def _finalize_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize the response with steps taken"""
        response["steps_taken"] = self.execution_steps
        response["step_count"] = len(self.execution_steps)

        # Create a summary message if multiple messages
        if len(response["messages"]) > 1:
            response["summary"] = " ".join(response["messages"])
        elif len(response["messages"]) == 1:
            response["summary"] = response["messages"][0]
        else:
            response["summary"] = "Command processed"

        return response


# Global instance
_handler = None


def get_context_aware_handler() -> ContextAwareCommandHandler:
    """Get or create context-aware handler instance"""
    global _handler
    if _handler is None:
        _handler = ContextAwareCommandHandler()
    return _handler
