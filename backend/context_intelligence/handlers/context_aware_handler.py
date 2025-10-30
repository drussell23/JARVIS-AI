"""
Context-Aware Command Handler for JARVIS
=======================================

This module provides intelligent command handling with full context awareness,
including screen lock state detection, speaker verification, and adaptive
response generation. It integrates multiple AI services to provide seamless
voice-controlled interactions.

The handler supports:
- Screen lock detection and automatic unlocking
- Speaker verification and personalized responses
- Action queries for system operations
- Predictive queries for code analysis
- Context-aware message generation

Example:
    >>> handler = get_context_aware_handler()
    >>> result = await handler.handle_command_with_context(
    ...     "open the terminal",
    ...     execute_callback=my_callback,
    ...     audio_data=audio_bytes
    ... )
    >>> print(result["success"])
    True
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Awaitable

from context_intelligence.detectors.screen_lock_detector import get_screen_lock_detector
from context_intelligence.handlers.action_query_handler import (
    ActionQueryResponse,
    get_action_query_handler,
    initialize_action_query_handler,
)
from context_intelligence.handlers.predictive_query_handler import (
    PredictiveQueryRequest,
    get_predictive_handler,
    initialize_predictive_handler,
)

logger = logging.getLogger(__name__)


class ContextAwareCommandHandler:
    """
    Handles commands with comprehensive context awareness and intelligent routing.
    
    This class provides the main interface for processing voice commands with
    full context awareness, including screen state, speaker verification,
    and adaptive response generation. It routes commands to specialized
    handlers based on intent detection.
    
    Attributes:
        screen_lock_detector: Service for detecting and managing screen lock state
        execution_steps: List of execution steps for tracking command processing
        predictive_handler: Handler for analytical and predictive queries
        action_handler: Handler for system action commands
        message_generator: Service for generating contextual messages
        speaker_verification: Service for verifying speaker identity
        voice_unlock_integration: Integrated voice-based unlock service
    
    Example:
        >>> handler = ContextAwareCommandHandler()
        >>> result = await handler.handle_command_with_context(
        ...     "close the terminal",
        ...     execute_callback=terminal_callback
        ... )
        >>> print(result["success"])
        True
    """

    def __init__(self) -> None:
        """
        Initialize the context-aware command handler.
        
        Sets up the screen lock detector and initializes lazy-loaded services
        to None for efficient resource management.
        """
        self.screen_lock_detector = get_screen_lock_detector()
        self.execution_steps: List[Dict[str, Any]] = []
        self.predictive_handler = None  # Lazy initialize
        self.action_handler = None  # Lazy initialize
        self.message_generator = None  # Lazy initialize
        self.speaker_verification = None  # Lazy initialize
        self.voice_unlock_integration = None  # Lazy initialize - new integrated unlock service

    async def handle_command_with_context(
        self,
        command: str,
        execute_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[Any]]] = None,
        intent_type: Optional[str] = None,
        audio_data: Optional[bytes] = None,
        speaker_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle a command with full context awareness and intelligent routing.

        This is the main entry point for processing voice commands. It performs
        context analysis, intent detection, screen lock handling, speaker
        verification, and command execution with appropriate feedback.

        Args:
            command: The voice command text to process
            execute_callback: Optional callback function to execute the actual command.
                            Should accept (command: str, context: Dict) and return result
            intent_type: Optional explicit intent type for routing
                        ('action_query', 'predictive_query', or None for auto-detection)
            audio_data: Raw audio bytes for speaker verification (optional)
            speaker_name: Known speaker name for personalization (optional)

        Returns:
            Dict containing:
                - success (bool): Whether command processing succeeded
                - command (str): The original command
                - messages (List[str]): Response messages for the user
                - steps_taken (List[Dict]): Detailed execution steps
                - context (Dict): System context at time of execution
                - timestamp (str): ISO timestamp of processing
                - result (Dict, optional): Detailed execution results
                - summary (str): Concise summary of the operation

        Raises:
            Exception: For unexpected errors during command processing
            
        Example:
            >>> async def my_callback(cmd, ctx):
            ...     return {"success": True, "message": "Done"}
            >>> 
            >>> result = await handler.handle_command_with_context(
            ...     "open terminal",
            ...     execute_callback=my_callback,
            ...     audio_data=audio_bytes,
            ...     speaker_name="John"
            ... )
            >>> print(result["messages"][0])
            "Opening terminal for you, John."
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
                logger.info(
                    "[CONTEXT AWARE] Detected predictive query - routing to predictive handler"
                )
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
            logger.info(f"[CONTEXT AWARE] Screen is {'LOCKED' if is_locked else 'UNLOCKED'}")

            if is_locked:
                self._add_step("Detected locked screen", {"screen_locked": True})
                logger.warning(
                    f"[CONTEXT AWARE] ⚠️  SCREEN IS LOCKED - Command requires unlocked screen"
                )

                # Check if command requires unlocked screen (pass speaker for personalization)
                screen_context = await self.screen_lock_detector.check_screen_context(
                    command, speaker_name=speaker_name
                )

                if screen_context["requires_unlock"]:
                    # Generate contextual unlock message
                    await self._lazy_initialize_services()

                    # Perform speaker verification if audio data available
                    speaker_verified = False
                    verification_confidence = 0.0
                    identified_speaker = speaker_name or "Sir"

                    if audio_data and self.speaker_verification:
                        try:
                            verification_result = await self.speaker_verification.verify_speaker(
                                audio_data, speaker_name
                            )
                            speaker_verified = verification_result.get("verified", False)
                            verification_confidence = verification_result.get("confidence", 0.0)
                            identified_speaker = verification_result.get(
                                "speaker_name", speaker_name
                            )
                            is_owner = verification_result.get("is_owner", False)

                            logger.info(
                                f"[CONTEXT AWARE] 🔐 Speaker: {identified_speaker}, "
                                f"Verified: {speaker_verified}, "
                                f"Confidence: {verification_confidence:.1%}, "
                                f"Owner: {is_owner}"
                            )
                        except Exception as e:
                            logger.warning(f"[CONTEXT AWARE] Speaker verification failed: {e}")

                    # Generate intelligent, contextual unlock message
                    if self.message_generator:
                        unlock_notification = (
                            await self.message_generator.generate_screen_unlock_message(
                                speaker_name=identified_speaker,
                                original_command=command,
                                is_owner=speaker_verified,
                            )
                        )
                    else:
                        # Fallback to screen context message
                        unlock_notification = screen_context["unlock_message"]

                    self._add_step("Screen unlock required", screen_context)

                    # Log prominently for debugging
                    logger.warning(f"[CONTEXT AWARE] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                    logger.warning(f"[CONTEXT AWARE] 🔓 SCREEN LOCKED - UNLOCK REQUIRED")
                    logger.warning(f"[CONTEXT AWARE] 📝 Command: {command}")
                    logger.warning(
                        f"[CONTEXT AWARE] 👤 Speaker: {identified_speaker} (verified: {speaker_verified})"
                    )
                    logger.warning(f"[CONTEXT AWARE] 📢 Unlock Message: '{unlock_notification}'")
                    logger.warning(f"[CONTEXT AWARE] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

                    # Speak the unlock message immediately with emphasis
                    logger.warning(f"[CONTEXT AWARE] 🎤 Speaking unlock notification NOW...")
                    await self._speak_message(unlock_notification, priority="high")
                    logger.warning(f"[CONTEXT AWARE] ✅ Unlock notification spoken")

                    # Add a longer delay to ensure user hears the message before unlock happens
                    logger.info(
                        f"[CONTEXT AWARE] ⏱️  Waiting 3 seconds for user to hear notification..."
                    )
                    await asyncio.sleep(3.0)
                    logger.info(f"[CONTEXT AWARE] ⏱️  Wait complete, proceeding with unlock...")

                    # Now perform the actual unlock with voice authentication
                    logger.info(
                        f"[CONTEXT AWARE] 🔓 Now unlocking screen with voice verification..."
                    )

                    # Use new integrated voice unlock service if available
                    if audio_data and self.voice_unlock_integration:
                        try:
                            unlock_success, unlock_message, unlock_confidence = (
                                await self.voice_unlock_integration.verify_and_unlock(
                                    audio_data=audio_data,
                                    speaker_name=identified_speaker,
                                    command_text=command,
                                )
                            )

                            logger.info(
                                f"[CONTEXT AWARE] Voice unlock result: success={unlock_success}, "
                                f"confidence={unlock_confidence:.2%}, message={unlock_message}"
                            )
                        except Exception as e:
                            logger.error(
                                f"[CONTEXT AWARE] Voice unlock integration error: {e}",
                                exc_info=True,
                            )
                            # Fallback to old method
                            unlock_success, unlock_message = (
                                await self.screen_lock_detector.handle_screen_lock_context(
                                    command, audio_data=audio_data, speaker_name=speaker_name
                                )
                            )
                    else:
                        # Fallback to old method if audio data not available or service not initialized
                        unlock_success, unlock_message = (
                            await self.screen_lock_detector.handle_screen_lock_context(
                                command, audio_data=audio_data, speaker_name=speaker_name
                            )
                        )

                    if unlock_success:
                        self._add_step("Screen unlocked successfully", {"unlocked": True})
                        logger.info(f"[CONTEXT AWARE] ✅ Screen unlocked successfully")
                        # Don't add unlock message to response since we already spoke it
                    else:
                        self._add_step("Screen unlock failed", {"error": unlock_message})
                        logger.error(f"[CONTEXT AWARE] ❌ Screen unlock failed: {unlock_message}")
                        response["success"] = False
                        response["messages"].append(unlock_message or "Failed to unlock screen")
                        return self._finalize_response(response)

            # Step 3: Execute the actual command
            if execute_callback:
                self._add_step("Executing command", {"command": command})

                try:
                    # Execute with context
                    exec_result = await execute_callback(command, context=system_context)

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
                                response["messages"].append("Command completed successfully")
                            response["result"] = exec_result
                        else:
                            self._add_step("Command execution failed", exec_result)
                            response["success"] = False
                            response["messages"].append(
                                exec_result.get("message", "Command failed")
                            )
                    else:
                        # Simple success
                        self._add_step("Command completed", {"result": str(exec_result)})
                        response["messages"].append("Command completed successfully")

                except Exception as e:
                    self._add_step("Command execution error", {"error": str(e)})
                    response["success"] = False
                    response["messages"].append(f"Error executing command: {str(e)}")

            # Step 4: Provide confirmation
            if response["success"]:
                confirmation = self._generate_confirmation(command, self.execution_steps)
                response["messages"].append(confirmation)

        except Exception as e:
            logger.error(f"Error in context-aware command handling: {e}")
            response["success"] = False
            response["messages"].append(f"An error occurred: {str(e)}")
            self._add_step("Error occurred", {"error": str(e)})

        return self._finalize_response(response)

    def _is_action_query(self, command: str) -> bool:
        """
        Check if a command is an action query requiring system operations.
        
        Action queries are commands that involve direct system manipulation
        like switching applications, closing windows, or running processes.
        
        Args:
            command: The command text to analyze
            
        Returns:
            bool: True if the command is classified as an action query
            
        Example:
            >>> handler._is_action_query("switch to terminal")
            True
            >>> handler._is_action_query("what is the weather")
            False
        """
        action_keywords = [
            "switch to space",
            "close",
            "fix",
            "run tests",
            "run build",
            "move",
            "focus",
            "launch",
            "quit",
            "restart",
            "open http",
            "fix the",
            "fix it",
            "close it",
            "close that",
        ]
        command_lower = command.lower()
        return any(keyword in command_lower for keyword in action_keywords)

    async def _handle_action_query(self, command: str) -> Dict[str, Any]:
        """
        Handle an action query using the specialized action handler.
        
        Routes system operation commands to the action query handler which
        can perform tasks like application switching, window management,
        and process control with safety checks and confirmation.
        
        Args:
            command: The action command to execute
            
        Returns:
            Dict containing execution results with standardized response format
            
        Raises:
            Exception: If action handler initialization or execution fails
            
        Example:
            >>> result = await handler._handle_action_query("close terminal")
            >>> print(result["success"])
            True
        """
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
                        implicit_resolver=implicit_resolver,  # ⭐ KEY INTEGRATION!
                    )

            logger.info(f"[CONTEXT AWARE] Processing action query: {command}")

            # Execute action query
            result: ActionQueryResponse = await self.action_handler.handle_action_query(
                command, context={}
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
                            "execution_status": (
                                result.execution_result.status.value
                                if result.execution_result
                                else "none"
                            ),
                        },
                        "timestamp": datetime.now().isoformat(),
                    }
                ],
                "context": {
                    "action_type": result.action_type,
                    "resolved_references": result.metadata.get("resolved_references", {}),
                    "safety_level": result.metadata.get("safety_level", "unknown"),
                },
                "timestamp": result.timestamp.isoformat(),
                "result": {
                    "execution": (
                        result.execution_result.__dict__ if result.execution_result else None
                    ),
                    "plan": (
                        {
                            "steps": [s.__dict__ for s in result.plan.steps] if result.plan else [],
                            "safety_level": (
                                result.plan.safety_level.value if result.plan else "unknown"
                            ),
                        }
                        if result.plan
                        else None
                    ),
                },
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
                "summary": f"Error: {str(e)}",
            }

    def _is_predictive_query(self, command: str) -> bool:
        """
        Check if a command is a predictive/analytical query.
        
        Predictive queries are commands that require analysis, explanation,
        or insights about code, progress, patterns, or system state.
        
        Args:
            command: The command text to analyze
            
        Returns:
            bool: True if the command is classified as a predictive query
            
        Example:
            >>> handler._is_predictive_query("analyze my code quality")
            True
            >>> handler._is_predictive_query("open terminal")
            False
        """
        predictive_keywords = [
            "making progress",
            "am i doing",
            "my progress",
            "what should i",
            "what to do next",
            "next steps",
            "any bugs",
            "any errors",
            "any issues",
            "potential bugs",
            "explain",
            "what does",
            "how does",
            "what patterns",
            "analyze patterns",
            "improve my workflow",
            "optimize",
            "work more efficiently",
            "code quality",
        ]
        command_lower = command.lower()
        return any(keyword in command_lower for keyword in predictive_keywords)

    async def _handle_predictive_query(self, command: str) -> Dict[str, Any]:
        """
        Handle a predictive/analytical query using the specialized predictive handler.
        
        Routes analytical commands to the predictive query handler which can
        perform code analysis, progress tracking, bug detection, and provide
        insights with optional visual analysis of the current screen.
        
        Args:
            command: The predictive query to process
            
        Returns:
            Dict containing analysis results with insights and recommendations
            
        Raises:
            Exception: If predictive handler initialization or execution fails
            
        Example:
            >>> result = await handler._handle_predictive_query("analyze my code")
            >>> print(result["context"]["confidence"])
            0.85
        """
        try:
            # Lazy initialize predictive handler
            if self.predictive_handler is None:
                self.predictive_handler = get_predictive_handler()
                if self.predictive_handler is None:
                    self.predictive_handler = initialize_predictive_handler()

            logger.info(f"[CONTEXT AWARE] Processing predictive query: {command}")

            # Determine if visual analysis is needed
            use_vision = any(
                keyword in command.lower() for keyword in ["explain", "code", "this", "that"]
            )

            # Create request
            request = PredictiveQueryRequest(
                query=command,
                use_vision=use_vision,
                capture_screen=use_vision,
                repo_path=".",
                additional_context={},
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
                            "query_type": (
                                result.analytics.query_type.value if result.analytics else "unknown"
                            ),
                            "confidence": result.confidence,
                            "used_vision": result.vision_analysis is not None,
                        },
                        "timestamp": datetime.now().isoformat(),
                    }
                ],
                "context": {
                    "query_type": (
                        result.analytics.query_type.value if result.analytics else "unknown"
                    ),
                    "confidence": result.confidence,
                    "insights": result.analytics.insights if result.analytics else [],
                },
                "timestamp": result.timestamp.isoformat(),
                "result": {
                    "analytics": (
                        {
                            "metrics": (
                                result.analytics.metrics.__dict__
                                if result.analytics and result.analytics.metrics
                                else None
                            ),
                            "bug_patterns": (
                                [bp.__dict__ for bp in result.analytics.bug_patterns]
                                if result.analytics
                                else []
                            ),
                            "recommendations": (
                                [rec.__dict__ for rec in result.analytics.recommendations]
                                if result.analytics
                                else []
                            ),
                        }
                        if result.analytics
                        else None
                    ),
                    "vision_analysis": result.vision_analysis,
                },
            }

            if response["success"]:
                response["summary"] = (
                    result.response_text if result.response_text else "Analysis complete"
                )
            else:
                response["summary"] = "Predictive query failed"

            logger.info(
                f"[CONTEXT AWARE] Predictive query completed: success={result.success}, confidence={result.confidence:.2%}"
            )

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
                "summary": f"Error: {str(e)}",
            }

    async def _get_system_context(self) -> Dict[str, Any]:
        """
        Get current system context including screen state and application status.
        
        Collects key system information needed for context-aware command
        processing, including screen lock state, active applications,
        network connectivity, and active window information.
        
        Returns:
            Dict containing:
                - screen_locked (bool): Whether screen is currently locked
                - active_apps (List): List of currently running applications
                - network_connected (bool): Network connectivity status
                - active_window (str): Currently focused window/application
                - summary (Dict): High-level context summary
                
        Example:
            >>> context = await handler._get_system_context()
            >>> print(context["screen_locked"])
            False
        """
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

    async def _lazy_initialize_services(self) -> None:
        """
        Lazy initialize optional services for enhanced functionality.
        
        Initializes speaker verification, message generation, and voice unlock
        services only when needed to optimize resource usage and startup time.
        Services that fail to initialize are logged but don't prevent operation.
        
        Raises:
            Exception: Individual service initialization failures are caught and logged
            
        Example:
            >>> await handler._lazy_initialize_services()
            # Services are now available if initialization succeeded
        """
        if self.speaker_verification is None:
            try:
                from voice.speaker_verification import SpeakerVerificationService

                self.speaker_verification = SpeakerVerificationService()
                await self.speaker_verification.initialize()
                logger.info("[CONTEXT AWARE] ✅ Speaker verification service initialized")
            except Exception as e:
                logger.warning(f"[CONTEXT AWARE] Speaker verification not available: {e}")

        if self.message_generator is None:
            try:
                from voice.contextual_message_generator import get_message_generator

                self.message_generator = get_message_generator()
                await self.message_generator.initialize()
                logger.info("[CONTEXT AWARE] ✅ Contextual message generator initialized")
            except Exception as e:
                logger.warning(f"[CONTEXT AWARE] Message generator not available: {e}")

        if self.voice_unlock_integration is None:
            try:
                from voice.voice_unlock_integration import get_voice_unlock_integration

                self.voice_unlock_integration = await get_voice_unlock_integration()
                logger.info("[CONTEXT AWARE] ✅ Voice unlock integration initialized")
            except Exception as e:
                logger.warning(f"[CONTEXT AWARE] Voice unlock integration not available: {e}")

    async def _speak_message(self, message: str, priority: str = "normal") -> None:
        """
        Speak a message immediately using multiple output methods for reliability.
        
        Uses both macOS 'say' command and WebSocket broadcasting to ensure
        message delivery even when screen is locked or WebSocket clients
        are unavailable. High priority messages use slower speech rates.
        
        Args:
            message: The text message to speak
            priority: Priority level ('normal' or 'high') affecting speech rate
            
        Raises:
            Exception: Logged but doesn't prevent operation if all methods fail
            
        Example:
            >>> await handler._speak_message("Screen unlocked", priority="high")
            # Message is spoken via available audio methods
        """
        try:
            logger.info(f"[CONTEXT AWARE] 📢 Speaking message (priority={priority}): {message}")

            # Use macOS say command FIRST (more reliable, especially when screen is locked)
            # This ensures the user hears the message even if WebSocket fails
            say_success = False
            try:
                pass

                # Run say command synchronously for immediate feedback
                # Use slower speech rate for unlock messages to ensure clarity
                speech_rate = "160" if priority == "high" else "190"  # Slower for unlock messages
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
                    logger.info(f"[CONTEXT AWARE] ✅ Spoke via macOS say command successfully")
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
                await broadcast_message({"type": "speak", "text": message, "priority": priority})
                logger.info(f"[CONTEXT AWARE] 📡 Broadcasted via WebSocket")
            except Exception as e:
                logger.debug(
                    f"[CONTEXT AWARE] WebSocket broadcast failed (this is OK if no clients): {e}"
                )

            # If both methods failed, log an error
            if not say_success:
                logger.error(f"[CONTEXT AWARE] ⚠️  WARNING: Could not speak message reliably!")

        except Exception as e:
            logger.error(f"[CONTEXT AWARE] ❌ Failed to speak message: {e}")

    def _add_step(self, description: str, details: Dict[str,