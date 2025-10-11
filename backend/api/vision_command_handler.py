"""
Vision Command Handler for JARVIS - Pure Intelligence Version
NO TEMPLATES. NO HARDCODING. PURE CLAUDE VISION INTELLIGENCE.

Every response is generated fresh by Claude based on what it sees.
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime

from .pure_vision_intelligence import (
    PureVisionIntelligence,
    ProactiveIntelligence,
    WorkflowIntelligence,
    ConversationContext,
)
from .proactive_monitoring_handler import get_monitoring_handler
from .activity_reporting_commands import is_activity_reporting_command

# Import new monitoring system components
try:
    from vision.monitoring_command_classifier import (
        classify_monitoring_command,
        CommandType,
        MonitoringAction,
    )
    from vision.monitoring_state_manager import (
        get_state_manager,
        MonitoringState,
        MonitoringCapability,
    )
    from vision.macos_indicator_controller import get_indicator_controller

    monitoring_system_available = True
except ImportError as e:
    logger.warning(f"Monitoring system components not available: {e}")
    monitoring_system_available = False

logger = logging.getLogger(__name__)


class WebSocketLogger:
    """Logger that sends logs to WebSocket for browser console"""

    def __init__(self):
        self.websocket_callback: Optional[Callable] = None

    def set_websocket_callback(self, callback: Callable):
        """Set callback to send logs through WebSocket"""
        self.websocket_callback = callback

    async def log(self, message: str, level: str = "info"):
        """Send log message through WebSocket"""
        if self.websocket_callback:
            try:
                await self.websocket_callback(
                    {
                        "type": "debug_log",
                        "message": f"[VISION] {message}",
                        "level": level,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            except Exception as e:
                logger.error(f"Failed to send WebSocket log: {e}")

        # Also log to server console
        if level == "error":
            logger.error(message)
        else:
            logger.info(message)


# Global WebSocket logger instance
ws_logger = WebSocketLogger()


class VisionCommandHandler:
    """
    Handles vision commands using pure Claude intelligence.
    Zero templates, zero hardcoded responses.
    """

    def __init__(self):
        self.vision_manager = None
        self.intelligence = None
        self.proactive = None
        self.workflow = None
        self.monitoring_active = False
        self.jarvis_api = None  # For voice integration

    async def initialize_intelligence(self, api_key: str = None):
        """Initialize pure vision intelligence system"""
        if not self.intelligence:
            # Try to get existing vision analyzer from app state
            vision_analyzer = None
            try:
                from api.jarvis_factory import get_app_state

                app_state = get_app_state()
                if app_state and hasattr(app_state, "vision_analyzer"):
                    vision_analyzer = app_state.vision_analyzer
                    logger.info(
                        "[PURE VISION] Using existing vision analyzer from app state"
                    )
            except Exception as e:
                logger.debug(f"Could not get vision analyzer from app state: {e}")

            # If no app state analyzer and we have an API key, create one
            if not vision_analyzer and api_key:
                try:
                    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer

                    vision_analyzer = ClaudeVisionAnalyzer(api_key)
                    logger.info(
                        "[PURE VISION] Created new vision analyzer with API key"
                    )
                except Exception as e:
                    logger.error(f"Failed to create vision analyzer: {e}")

            # Store the vision analyzer reference for later use
            self.vision_analyzer = vision_analyzer

            # If no existing analyzer, create a wrapper for the API
            if vision_analyzer:
                # Create a Claude client wrapper that uses the existing vision analyzer
                class ClaudeVisionWrapper:
                    def __init__(self, analyzer):
                        self.analyzer = analyzer

                    async def analyze_image_with_prompt(
                        self, image: Any, prompt: str, max_tokens: int = 500
                    ) -> Dict[str, Any]:
                        """Wrapper to use existing vision analyzer"""
                        try:
                            # Use the existing analyzer's analyze method
                            result = await self.analyzer.analyze_image_with_prompt(
                                image=image, prompt=prompt
                            )

                            # Extract the response text
                            if isinstance(result, dict):
                                # First check for 'content' key (from analyze_image_with_prompt)
                                if "content" in result:
                                    return {"content": result["content"]}
                                # Then check for description or response
                                return {
                                    "content": result.get(
                                        "description",
                                        result.get("response", str(result)),
                                    )
                                }
                            else:
                                return {"content": str(result)}
                        except Exception as e:
                            logger.error(f"Vision analyzer error: {e}")
                            raise

                    async def analyze_multiple_images_with_prompt(
                        self,
                        images: List[Dict[str, Any]],
                        prompt: str,
                        max_tokens: int = 1000,
                    ) -> Dict[str, Any]:
                        """Wrapper for multi-space analysis"""
                        try:
                            # Use the analyzer's multi-image method
                            if hasattr(
                                self.analyzer, "analyze_multiple_images_with_prompt"
                            ):
                                result = await self.analyzer.analyze_multiple_images_with_prompt(
                                    images=images, prompt=prompt, max_tokens=max_tokens
                                )
                                return result
                            else:
                                # Fallback: analyze first image only
                                logger.warning(
                                    "Analyzer doesn't support multi-image analysis, using first image only"
                                )
                                if images:
                                    first_image = images[0]["image"]
                                    return await self.analyze_image_with_prompt(
                                        first_image, prompt, max_tokens
                                    )
                                else:
                                    return {
                                        "content": "No images provided for analysis",
                                        "success": False,
                                    }
                        except Exception as e:
                            logger.error(f"Multi-image vision analyzer error: {e}")
                            raise

                claude_client = ClaudeVisionWrapper(vision_analyzer)
            else:
                # No vision analyzer available - use mock
                logger.warning(
                    "[PURE VISION] No vision analyzer available, using mock responses"
                )
                claude_client = None

            # Initialize intelligence systems with multi-space enabled
            self.intelligence = PureVisionIntelligence(
                claude_client, enable_multi_space=True
            )
            self.proactive = ProactiveIntelligence(self.intelligence)
            self.workflow = WorkflowIntelligence(self.intelligence)

            logger.info("[PURE VISION] Intelligence systems initialized")

    async def handle_command(self, command_text: str) -> Dict[str, Any]:
        """
        Handle any vision command with pure intelligence.
        No pattern matching, no templates - Claude understands intent.
        """
        logger.info(f"[VISION] Handling command: {command_text}")
        await ws_logger.log(f"Processing vision command: {command_text}")

        # IMPORTANT: Check if this is a lock/unlock screen command - should NOT be handled by vision
        command_lower = command_text.lower()
        if ("lock" in command_lower and "screen" in command_lower) or (
            "unlock" in command_lower and "screen" in command_lower
        ):
            logger.info(
                f"[VISION] Lock/unlock screen command detected, not handling as vision"
            )
            return {
                "handled": False,
                "reason": "Lock/unlock screen commands are system commands, not vision",
            }

        # Check if this is a proactive monitoring command
        monitoring_handler = get_monitoring_handler(self)
        monitoring_result = await monitoring_handler.handle_monitoring_request(
            command_text
        )
        if monitoring_result.get("handled"):
            return monitoring_result

        # Ensure intelligence is initialized
        if not self.intelligence:
            await self.initialize_intelligence()

        # Check if this is a multi-space query first
        needs_multi_space = False
        if self.intelligence and hasattr(self.intelligence, "_should_use_multi_space"):
            needs_multi_space = self.intelligence._should_use_multi_space(command_text)
            logger.info(f"[VISION] Multi-space query detected: {needs_multi_space}")

        # Capture screen(s) based on query type
        if needs_multi_space:
            # Capture multiple spaces for comprehensive analysis
            screenshot = await self._capture_screen(multi_space=True)
            logger.info(
                f"[VISION] Captured {len(screenshot) if isinstance(screenshot, dict) else 1} space(s)"
            )
        else:
            # Single space capture
            screenshot = await self._capture_screen()

        if not screenshot:
            # Even error messages come from Claude
            return await self._get_error_response("screenshot_failed", command_text)

        # Use new classification system if available
        if monitoring_system_available:
            # Get current monitoring state
            state_manager = get_state_manager()
            current_state = state_manager.is_monitoring_active()

            # Classify the command
            command_context = classify_monitoring_command(command_text, current_state)
            logger.info(
                f"[VISION] Command classified as: {command_context['type'].value} with confidence {command_context['confidence']:.2f}"
            )

            # Route based on command type
            if command_context["type"] == CommandType.MONITORING_CONTROL:
                return await self._handle_monitoring_control(
                    command_text, command_context, screenshot
                )
            elif command_context["type"] == CommandType.MONITORING_STATUS:
                return await self._handle_monitoring_status(
                    command_text, command_context, screenshot
                )
            elif command_context["type"] == CommandType.VISION_QUERY:
                # Pure vision query - let Claude see and respond
                response = await self.intelligence.understand_and_respond(
                    screenshot, command_text
                )
                return {
                    "handled": True,
                    "response": response,
                    "pure_intelligence": True,
                    "monitoring_active": self.monitoring_active,
                    "context": self.intelligence.context.get_temporal_context(),
                }
            else:
                # Ambiguous command - use Claude to determine intent
                is_monitoring_command = await self._is_monitoring_command(
                    command_text, screenshot
                )
                if is_monitoring_command:
                    return await self._handle_monitoring_command(
                        command_text, screenshot
                    )
                else:
                    # Pure vision query
                    response = await self.intelligence.understand_and_respond(
                        screenshot, command_text
                    )
                    return {
                        "handled": True,
                        "response": response,
                        "pure_intelligence": True,
                        "monitoring_active": self.monitoring_active,
                        "context": self.intelligence.context.get_temporal_context(),
                    }
        else:
            # Fallback to old logic
            # First check if it's an activity reporting command (faster than Claude)
            if is_activity_reporting_command(command_text):
                is_monitoring_command = True
            # Quick check for common monitoring phrases
            elif any(
                phrase in command_text.lower()
                for phrase in [
                    "start monitoring",
                    "enable monitoring",
                    "monitor my screen",
                    "enable screen monitoring",
                    "monitoring capabilities",
                    "turn on monitoring",
                    "activate monitoring",
                    "begin monitoring",
                ]
            ):
                is_monitoring_command = True
                logger.info(f"Quick match: '{command_text}' is a monitoring command")
            else:
                # Determine if this is a monitoring command through Claude
                is_monitoring_command = await self._is_monitoring_command(
                    command_text, screenshot
                )

            if is_monitoring_command:
                return await self._handle_monitoring_command(command_text, screenshot)
            else:
                # Pure vision query - let Claude see and respond
                response = await self.intelligence.understand_and_respond(
                    screenshot, command_text
                )

                return {
                    "handled": True,
                    "response": response,
                    "pure_intelligence": True,
                    "monitoring_active": self.monitoring_active,
                    "context": self.intelligence.context.get_temporal_context(),
                }

    async def analyze_screen(self, command_text: str) -> Dict[str, Any]:
        """Analyze screen with enhanced multi-space intelligence"""
        logger.info(f"[VISION] analyze_screen called with: {command_text}")

        # Use the same logic as handle_command but for screen analysis
        try:
            # Check if this is a multi-space query
            needs_multi_space = False
            if self.intelligence and hasattr(
                self.intelligence, "_should_use_multi_space"
            ):
                needs_multi_space = self.intelligence._should_use_multi_space(
                    command_text
                )
                logger.info(f"[VISION] Multi-space query detected: {needs_multi_space}")

            # Capture screen(s) based on query type
            if needs_multi_space:
                # Capture multiple spaces for comprehensive analysis
                screenshot = await self._capture_screen(multi_space=True)
                logger.info(
                    f"[VISION] Captured {len(screenshot) if isinstance(screenshot, dict) else 1} space(s)"
                )
            else:
                # Single space capture
                screenshot = await self._capture_screen()

            if not screenshot:
                # Even error messages come from Claude
                return await self._get_error_response("screenshot_failed", command_text)

            # Use enhanced multi-space intelligence if available
            if (
                self.intelligence
                and hasattr(self.intelligence, "multi_space_extension")
                and hasattr(
                    self.intelligence.multi_space_extension,
                    "analyze_comprehensive_workspace",
                )
            ):

                # Get workspace data
                window_data = await self.intelligence._gather_multi_space_data()

                # Use enhanced analysis
                enhanced_response = self.intelligence.multi_space_extension.analyze_comprehensive_workspace(
                    command_text, window_data
                )

                logger.info(
                    f"[ENHANCED VISION] Generated enhanced response: {len(enhanced_response)} chars"
                )
                return {
                    "handled": True,
                    "response": enhanced_response,
                    "pure_intelligence": True,
                    "monitoring_active": self.monitoring_active,
                    "context": self.intelligence.context.get_temporal_context(),
                    "enhanced_analysis": True,
                }
            else:
                # Fallback to basic Claude analysis
                response = await self.intelligence.understand_and_respond(
                    screenshot, command_text
                )

                return {
                    "handled": True,
                    "response": response,
                    "pure_intelligence": True,
                    "monitoring_active": self.monitoring_active,
                    "context": self.intelligence.context.get_temporal_context(),
                }

        except Exception as e:
            logger.error(f"[VISION] Error in analyze_screen: {e}")
            import traceback

            traceback.print_exc()

            # Return error response
            return {
                "handled": False,
                "response": f"I encountered an error while analyzing your screen: {str(e)}",
                "error": str(e),
                "monitoring_active": self.monitoring_active,
            }

    async def _is_monitoring_command(self, command: str, screenshot: Any) -> bool:
        """Let Claude determine if this is a monitoring command"""
        prompt = f"""Look at the screen and the user's command: "{command}"

Is this command asking to start or stop screen monitoring/watching?
Respond with just "YES" or "NO".

Examples of monitoring commands:
- "start monitoring my screen"
- "stop watching"
- "activate vision monitoring"
- "enable screen monitoring"
- "enable screen monitoring capabilities"
- "turn on monitoring"

Examples of non-monitoring commands:
- "what do you see?"
- "what's my battery?"
- "analyze this screen"
"""

        response = await self.intelligence._get_claude_vision_response(
            screenshot, prompt
        )
        return response.get("response", "").strip().upper() == "YES"

    async def _handle_monitoring_command(
        self, command: str, screenshot: Any
    ) -> Dict[str, Any]:
        """Handle monitoring commands with natural responses"""

        # Check if this is an activity reporting command
        if is_activity_reporting_command(command):
            monitoring_handler = get_monitoring_handler(self)
            return await monitoring_handler.enable_change_reporting()

        # Let Claude understand if this is start or stop
        intent_prompt = f"""The user said: "{command}"

Are they asking to START or STOP monitoring?
Respond with just "START" or "STOP".
"""

        response = await self.intelligence._get_claude_vision_response(
            screenshot, intent_prompt
        )
        intent = response.get("response", "").strip().upper()

        if intent == "START":
            self.monitoring_active = True
            self.proactive.monitoring_active = True

            # Start multi-space monitoring with purple indicator
            monitoring_success = False
            if hasattr(self.intelligence, "start_multi_space_monitoring"):
                monitoring_started = (
                    await self.intelligence.start_multi_space_monitoring()
                )
                if monitoring_started:
                    logger.info("Multi-space monitoring started with purple indicator")
                    monitoring_success = True
                else:
                    logger.warning("Failed to start multi-space monitoring")
                    monitoring_success = False

            # Update vision status if monitoring started successfully
            if monitoring_success:
                try:
                    from vision.vision_status_manager import get_vision_status_manager

                    vision_status_manager = get_vision_status_manager()
                    await vision_status_manager.update_vision_status(True)
                    logger.info("✅ Vision status updated to connected (old flow)")
                except Exception as e:
                    logger.error(f"Failed to update vision status: {e}")

            # Get natural response for starting monitoring
            if monitoring_success:
                start_prompt = f"""The user asked: "{command}"

You're JARVIS. The screen monitoring is now ACTIVE with the macOS purple indicator visible.

Give a BRIEF confirmation (1-2 sentences max) that includes:
1. Monitoring is now active
2. The purple indicator is visible in the menu bar
3. You can see their screen

Example: "Screen monitoring is now active, Sir. The purple indicator is visible in your menu bar, and I can see your desktop."

BE CONCISE. Do not explain technical details or list options.
"""
            else:
                start_prompt = f"""The user asked: "{command}"

You're JARVIS. Screen monitoring FAILED to start due to permissions.

Give a BRIEF response (1-2 sentences) explaining:
1. Monitoring couldn't start
2. They need to grant screen recording permission

Example: "I couldn't start screen monitoring, Sir. Please grant screen recording permission in System Preferences."

BE CONCISE.
"""
            response = await self.intelligence._get_claude_vision_response(
                screenshot, start_prompt
            )

            # Start proactive monitoring
            asyncio.create_task(self._proactive_monitoring_loop())

        else:  # STOP
            self.monitoring_active = False
            self.proactive.monitoring_active = False

            # Stop multi-space monitoring and remove purple indicator
            if hasattr(self.intelligence, "stop_multi_space_monitoring"):
                await self.intelligence.stop_multi_space_monitoring()
                logger.info("Multi-space monitoring stopped, purple indicator removed")

            # Update vision status to disconnected
            try:
                from vision.vision_status_manager import get_vision_status_manager

                vision_status_manager = get_vision_status_manager()
                await vision_status_manager.update_vision_status(False)
                logger.info("✅ Vision status updated to disconnected (old flow)")
            except Exception as e:
                logger.error(f"Failed to update vision status: {e}")

            # Get natural response for stopping monitoring
            stop_prompt = f"""The user asked: "{command}"

You're JARVIS. Screen monitoring has been STOPPED and the purple indicator is gone.

Give a BRIEF confirmation (1 sentence) that monitoring has stopped.

Example: "Screen monitoring has been disabled, Sir."

BE CONCISE. No technical details.
"""
            response = await self.intelligence._get_claude_vision_response(
                screenshot, stop_prompt
            )

        return {
            "handled": True,
            "response": response.get("response"),
            "monitoring_active": self.monitoring_active,
            "pure_intelligence": True,
        }

    async def _handle_monitoring_control(
        self, command: str, context: Dict[str, Any], screenshot: Any
    ) -> Dict[str, Any]:
        """Handle monitoring control commands with new system"""
        state_manager = get_state_manager()
        indicator_controller = get_indicator_controller()

        action = context["action"]

        if action == MonitoringAction.START:
            # Check if we can start monitoring
            can_start, reason = state_manager.can_start_monitoring()
            if not can_start:
                return {
                    "handled": True,
                    "response": f"I cannot start monitoring right now, Sir. {reason}",
                    "monitoring_active": state_manager.is_monitoring_active(),
                    "pure_intelligence": True,
                }

            # Transition to activating state
            await state_manager.transition_to(MonitoringState.ACTIVATING)

            # Activate macOS indicator
            indicator_result = await indicator_controller.activate_indicator()

            if indicator_result["success"]:
                # Update state manager
                state_manager.update_component_status("macos_indicator", True)
                state_manager.add_capability(MonitoringCapability.MACOS_INDICATOR)

                # Start multi-space monitoring
                monitoring_started = False
                if hasattr(self.intelligence, "start_multi_space_monitoring"):
                    monitoring_started = (
                        await self.intelligence.start_multi_space_monitoring()
                    )
                    if monitoring_started:
                        state_manager.update_component_status("multi_space", True)
                        state_manager.add_capability(MonitoringCapability.MULTI_SPACE)

                # Update monitoring active flag
                self.monitoring_active = True
                self.proactive.monitoring_active = True
                state_manager.update_component_status("vision_intelligence", True)

                # Transition to active state
                await state_manager.transition_to(MonitoringState.ACTIVE)

                # Update vision status to connected
                try:
                    from vision.vision_status_manager import get_vision_status_manager

                    vision_status_manager = get_vision_status_manager()
                    await vision_status_manager.update_vision_status(True)
                    logger.info("✅ Vision status updated to connected")
                except Exception as e:
                    logger.error(f"Failed to update vision status: {e}")

                # Start proactive monitoring
                asyncio.create_task(self._proactive_monitoring_loop())

                return {
                    "handled": True,
                    "response": "Screen monitoring is now active, Sir. The purple indicator is visible in your menu bar, and I can see your entire workspace.",
                    "monitoring_active": True,
                    "pure_intelligence": True,
                    "indicator_active": True,
                }
            else:
                # Indicator activation failed
                await state_manager.transition_to(
                    MonitoringState.ERROR,
                    {
                        "error": "Failed to activate macOS indicator",
                        "details": indicator_result,
                    },
                )

                return {
                    "handled": True,
                    "response": "I couldn't activate screen monitoring, Sir. Please ensure screen recording permission is granted in System Preferences.",
                    "monitoring_active": False,
                    "pure_intelligence": True,
                    "error": indicator_result.get("error"),
                }

        elif action == MonitoringAction.STOP:
            # Check if we can stop monitoring
            can_stop, reason = state_manager.can_stop_monitoring()
            if not can_stop:
                return {
                    "handled": True,
                    "response": f"I cannot stop monitoring right now, Sir. {reason}",
                    "monitoring_active": state_manager.is_monitoring_active(),
                    "pure_intelligence": True,
                }

            # Transition to deactivating state
            await state_manager.transition_to(MonitoringState.DEACTIVATING)

            # Stop monitoring components
            self.monitoring_active = False
            self.proactive.monitoring_active = False

            # Stop multi-space monitoring
            if hasattr(self.intelligence, "stop_multi_space_monitoring"):
                await self.intelligence.stop_multi_space_monitoring()

            # Deactivate macOS indicator
            indicator_result = await indicator_controller.deactivate_indicator()

            # Clear capabilities
            state_manager.remove_capability(MonitoringCapability.MACOS_INDICATOR)
            state_manager.remove_capability(MonitoringCapability.MULTI_SPACE)

            # Update component status
            state_manager.update_component_status("macos_indicator", False)
            state_manager.update_component_status("multi_space", False)
            state_manager.update_component_status("vision_intelligence", False)

            # Transition to inactive state
            await state_manager.transition_to(MonitoringState.INACTIVE)

            # Update vision status to disconnected
            try:
                from vision.vision_status_manager import get_vision_status_manager

                vision_status_manager = get_vision_status_manager()
                await vision_status_manager.update_vision_status(False)
                logger.info("✅ Vision status updated to disconnected")
            except Exception as e:
                logger.error(f"Failed to update vision status: {e}")

            return {
                "handled": True,
                "response": "Screen monitoring has been disabled, Sir.",
                "monitoring_active": False,
                "pure_intelligence": True,
                "indicator_active": False,
            }

        return {
            "handled": True,
            "response": "I'm not sure how to handle that monitoring command, Sir.",
            "monitoring_active": self.monitoring_active,
            "pure_intelligence": True,
        }

    async def _handle_monitoring_status(
        self, command: str, context: Dict[str, Any], screenshot: Any
    ) -> Dict[str, Any]:
        """Handle monitoring status queries"""
        state_manager = get_state_manager()
        state_info = state_manager.get_state_info()

        # Build status response
        if state_info["is_active"]:
            capabilities = state_info["active_capabilities"]
            cap_text = f"with {', '.join(capabilities)}" if capabilities else ""
            response = f"Yes Sir, monitoring is currently active {cap_text}. The purple indicator should be visible in your menu bar."
        elif state_info["is_transitioning"]:
            response = f"Monitoring is currently {state_info['current_state'].replace('_', ' ')}, Sir."
        else:
            response = "No Sir, monitoring is not active. Would you like me to start monitoring your screen?"

        return {
            "handled": True,
            "response": response,
            "monitoring_active": state_info["is_active"],
            "pure_intelligence": True,
            "state_info": state_info,
        }

    async def _proactive_monitoring_loop(self):
        """Proactive monitoring with pure intelligence"""
        logger.info("[VISION] Starting proactive monitoring loop")

        while self.monitoring_active:
            try:
                # Wait before next check
                await asyncio.sleep(5)

                if not self.monitoring_active:
                    break

                # Capture screen and check for important changes
                screenshot = await self._capture_screen()
                if screenshot:
                    proactive_message = await self.proactive.observe_and_communicate(
                        screenshot
                    )

                    if proactive_message and self.jarvis_api:
                        # Send proactive message through JARVIS voice
                        try:
                            await self.jarvis_api.speak_proactive(proactive_message)
                        except Exception as e:
                            logger.error(f"Failed to speak proactive message: {e}")

            except Exception as e:
                logger.error(f"Proactive monitoring error: {e}")
                await asyncio.sleep(5)

    async def _capture_screen(
        self, multi_space=False, space_number=None
    ) -> Optional[Any]:
        """
        Capture screen(s) with multi-space support

        Args:
            multi_space: If True, capture all desktop spaces
            space_number: If provided, capture specific space

        Returns:
            Single screenshot or Dict[int, screenshot] for multi-space
        """
        try:
            # Initialize vision manager if needed
            await self._ensure_vision_manager()

            if (
                self.vision_manager
                and hasattr(self.vision_manager, "vision_analyzer")
                and self.vision_manager.vision_analyzer
            ):
                # Use enhanced capture with multi-space support
                screenshot = await self.vision_manager.vision_analyzer.capture_screen(
                    multi_space=multi_space, space_number=space_number
                )
                return screenshot
            else:
                # Try to capture screen directly as fallback
                logger.info("[VISION] Attempting direct screen capture...")
                try:
                    # Try macOS screencapture
                    import subprocess
                    import tempfile
                    from PIL import Image

                    with tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False
                    ) as tmp:
                        tmp_path = tmp.name

                    # Capture screen
                    result = subprocess.run(
                        ["screencapture", "-x", tmp_path],
                        capture_output=True,
                        text=True,
                    )

                    if result.returncode == 0:
                        # Load and return image
                        screenshot = Image.open(tmp_path)
                        os.unlink(tmp_path)  # Clean up
                        logger.info("[VISION] Direct screen capture successful")
                        return screenshot
                    else:
                        logger.error(f"screencapture failed: {result.stderr}")

                except Exception as e:
                    logger.error(f"Direct capture failed: {e}")

        except Exception as e:
            logger.error(f"Screen capture error: {e}")

        return None

    async def _ensure_vision_manager(self):
        """Initialize vision manager if not already done"""
        if not self.vision_manager:
            try:
                logger.info("[VISION INIT] Attempting to import vision_manager...")
                try:
                    from api.vision_websocket import vision_manager
                except ImportError:
                    from .vision_websocket import vision_manager

                self.vision_manager = vision_manager
                logger.info(f"[VISION INIT] Vision manager imported: {vision_manager}")

                # Check if vision_analyzer needs initialization
                if hasattr(vision_manager, "vision_analyzer"):
                    if vision_manager.vision_analyzer is None:
                        logger.info(
                            "[VISION INIT] Vision analyzer is None, checking app state..."
                        )
                        # Try to get from app state
                        try:
                            import sys
                            import os

                            sys.path.append(
                                os.path.dirname(
                                    os.path.dirname(os.path.abspath(__file__))
                                )
                            )
                            from main import app

                            if hasattr(app.state, "vision_analyzer"):
                                vision_manager.vision_analyzer = (
                                    app.state.vision_analyzer
                                )
                                logger.info(
                                    "[VISION INIT] Set vision analyzer from app state"
                                )
                        except Exception as e:
                            logger.error(
                                f"[VISION INIT] Failed to get vision analyzer from app state: {e}"
                            )

                        # If still None and we have our own vision analyzer, use that
                        if (
                            vision_manager.vision_analyzer is None
                            and hasattr(self, "vision_analyzer")
                            and self.vision_analyzer
                        ):
                            vision_manager.vision_analyzer = self.vision_analyzer
                            logger.info(
                                "[VISION INIT] Set vision analyzer from handler"
                            )
                    else:
                        logger.info("[VISION INIT] Vision analyzer already set")

            except Exception as e:
                logger.error(f"Failed to initialize vision manager: {e}")

    async def _get_error_response(
        self, error_type: str, command: str, details: str = ""
    ) -> Dict[str, Any]:
        """Even errors are communicated naturally by Claude"""
        error_prompt = f"""The user asked: "{command}"

An error occurred: {error_type}
{f"Details: {details}" if details else ""}

You're JARVIS. Respond naturally to explain the issue and suggest a solution.
Be helpful and specific, but keep it conversational.
Never use generic error messages or technical jargon.
"""

        # Use mock response if no Claude client
        if self.intelligence and self.intelligence.claude:
            response = await self.intelligence._get_claude_vision_response(
                None, error_prompt
            )
            error_message = response.get("response")
        else:
            # Natural fallback
            if error_type == "screenshot_failed":
                error_message = "I'm having trouble accessing your screen right now, Sir. Let me check the vision system configuration."
            elif error_type == "intelligence_error":
                error_message = "I encountered an issue processing that request. Let me recalibrate the vision systems."
            else:
                error_message = f"Something went wrong with that request, Sir. {details if details else 'Let me investigate.'}"

        return {
            "handled": True,
            "response": error_message,
            "error": True,
            "pure_intelligence": True,
            "monitoring_active": self.monitoring_active,
        }

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        if self.intelligence and self.intelligence.context:
            return {
                "conversation_length": len(self.intelligence.context.history),
                "monitoring_active": self.monitoring_active,
                "workflow_state": self.intelligence.context.workflow_state,
                "emotional_state": (
                    self.intelligence.context.emotional_context.value
                    if self.intelligence.context.emotional_context
                    else "neutral"
                ),
            }
        return {
            "conversation_length": 0,
            "monitoring_active": False,
            "workflow_state": "unknown",
            "emotional_state": "neutral",
        }


# Singleton instance
vision_command_handler = VisionCommandHandler()
