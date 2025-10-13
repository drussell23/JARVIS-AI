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

logger = logging.getLogger(__name__)

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

# Import enhanced multi-space system
try:
    from vision.enhanced_multi_space_integration import EnhancedMultiSpaceSystem
    enhanced_system_available = True
except ImportError as e:
    logger.warning(f"Enhanced multi-space system not available: {e}")
    enhanced_system_available = False

# Import workspace name processor
try:
    from vision.workspace_name_processor import process_jarvis_response, update_workspace_names
    workspace_processor_available = True
except ImportError as e:
    logger.warning(f"Workspace name processor not available: {e}")
    workspace_processor_available = False
    process_jarvis_response = lambda x, y=None: x  # Fallback to identity function
    update_workspace_names = lambda x: None

# Import workspace name detector for better name detection
try:
    from vision.workspace_name_detector import process_response_with_workspace_names, get_current_workspace_names
    workspace_detector_available = True
except ImportError as e:
    logger.warning(f"Workspace name detector not available: {e}")
    workspace_detector_available = False
    process_response_with_workspace_names = lambda x, y=None: x
    get_current_workspace_names = lambda: {}

# Import Yabai-based multi-space intelligence system
try:
    from vision.yabai_space_detector import YabaiSpaceDetector, YabaiStatus
    from vision.workspace_analyzer import WorkspaceAnalyzer
    from vision.space_response_generator import SpaceResponseGenerator
    yabai_system_available = True
    logger.info("[VISION] ✅ Yabai multi-space intelligence system loaded")
except ImportError as e:
    logger.warning(f"Yabai multi-space system not available: {e}")
    yabai_system_available = False

# Import Intelligent Query Classification System
try:
    from vision.intelligent_query_classifier import (
        QueryIntent,
        ClassificationResult,
        get_query_classifier
    )
    from vision.smart_query_router import get_smart_router
    from vision.query_context_manager import get_context_manager
    from vision.adaptive_learning_system import get_learning_system
    from vision.performance_monitor import get_performance_monitor
    intelligent_system_available = True
    logger.info("[VISION] ✅ Intelligent query classification system loaded")
except ImportError as e:
    logger.warning(f"Intelligent classification system not available: {e}")
    intelligent_system_available = False

# Import Proactive Suggestions System
try:
    from vision.proactive_suggestions import get_proactive_system, ProactiveSuggestion
    proactive_suggestions_available = True
    logger.info("[VISION] ✅ Proactive suggestions system loaded")
except ImportError as e:
    logger.warning(f"Proactive suggestions system not available: {e}")
    proactive_suggestions_available = False


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

        # Initialize enhanced multi-space system if available
        self.enhanced_system = None
        if enhanced_system_available:
            try:
                self.enhanced_system = EnhancedMultiSpaceSystem()
                logger.info("[VISION] Enhanced multi-space system initialized")
            except Exception as e:
                logger.warning(f"[VISION] Could not initialize enhanced system: {e}")

        # Initialize Yabai-based multi-space intelligence system
        self.yabai_detector = None
        self.workspace_analyzer = None
        self.space_response_generator = None
        if yabai_system_available:
            try:
                self.yabai_detector = YabaiSpaceDetector()
                self.workspace_analyzer = WorkspaceAnalyzer()
                self.space_response_generator = SpaceResponseGenerator(use_sir_prefix=True)
                logger.info("[VISION] ✅ Yabai multi-space intelligence initialized")
            except Exception as e:
                logger.warning(f"[VISION] Could not initialize Yabai system: {e}")

        # Initialize Intelligent Query Classification System
        self.classifier = None
        self.smart_router = None
        self.context_manager = None
        self.learning_system = None
        self.performance_monitor = None

        if intelligent_system_available:
            try:
                # Get singleton instances
                self.context_manager = get_context_manager()
                self.learning_system = get_learning_system()
                self.performance_monitor = get_performance_monitor(report_interval_minutes=60)

                logger.info("[VISION] ✅ Intelligent query classification system initialized")
            except Exception as e:
                logger.warning(f"[VISION] Could not initialize intelligent system: {e}")

        # Initialize Proactive Suggestions System
        self.proactive_system = None
        if proactive_suggestions_available:
            try:
                self.proactive_system = get_proactive_system()
                logger.info("[VISION] ✅ Proactive suggestions system initialized")
            except Exception as e:
                logger.warning(f"[VISION] Could not initialize proactive system: {e}")

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

            # Update enhanced system with intelligence if available
            if self.enhanced_system:
                self.enhanced_system.vision_intelligence = self.intelligence
                logger.info("[ENHANCED] Updated enhanced system with vision intelligence")

            # Initialize intelligent classification system with Claude client
            if intelligent_system_available and claude_client:
                try:
                    self.classifier = get_query_classifier(claude_client)

                    # Initialize smart router with handlers
                    self.smart_router = get_smart_router(
                        yabai_handler=self._handle_yabai_query,
                        vision_handler=self._handle_vision_query,
                        multi_space_handler=self._handle_multi_space_query,
                        claude_client=claude_client
                    )

                    logger.info("[INTELLIGENT] Classifier and router initialized with Claude client")
                except Exception as e:
                    logger.warning(f"[INTELLIGENT] Could not initialize classifier/router: {e}")

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

        # ==============================================================================
        # INTELLIGENT CLASSIFICATION SYSTEM
        # Use smart router to classify and route query to optimal pipeline
        # ==============================================================================
        if intelligent_system_available and self.smart_router and self.context_manager:
            try:
                logger.info("[INTELLIGENT] Using smart router for query classification")

                # Get context for classification
                context = self.context_manager.get_context_for_query(command_text)

                # Add current Yabai state to context if available
                if self.yabai_detector:
                    try:
                        active_space = await self.yabai_detector.get_focused_space()
                        all_spaces = await self.yabai_detector.get_all_spaces()
                        context['active_space'] = active_space.get('index') if active_space else None
                        context['total_spaces'] = len(all_spaces) if all_spaces else 0
                    except Exception:
                        pass  # Continue without Yabai context

                # Route the query through intelligent system
                routing_result = await self.smart_router.route_query(
                    query=command_text,
                    context=context
                )

                # Record query in context manager
                self.context_manager.record_query(
                    query=command_text,
                    intent=routing_result.intent.value,
                    active_space=context.get('active_space'),
                    total_spaces=context.get('total_spaces', 0),
                    response_latency_ms=routing_result.latency_ms
                )

                # Collect performance metrics periodically
                if self.performance_monitor and self.performance_monitor.should_generate_report():
                    await self.performance_monitor.collect_metrics()
                    self.performance_monitor.mark_report_generated()
                    logger.info("[INTELLIGENT] Performance metrics collected")

                # Return routed response
                return {
                    "handled": True,
                    "response": routing_result.response,
                    "intelligent_routing": True,
                    "intent": routing_result.intent.value,
                    "latency_ms": routing_result.latency_ms,
                    "metadata": routing_result.metadata,
                    "monitoring_active": self.monitoring_active,
                }

            except Exception as e:
                logger.error(f"[INTELLIGENT] Smart routing failed, falling back to legacy: {e}", exc_info=True)
                # Fall through to legacy handling
        # ==============================================================================
        # END INTELLIGENT CLASSIFICATION SYSTEM
        # ==============================================================================

        # Try enhanced system first if available
        if self.enhanced_system:
            try:
                enhanced_result = await self.enhanced_system.handle_vision_command(command_text)
                if enhanced_result.get('handled') != False:
                    # Enhanced system handled the command
                    response = enhanced_result.get('response', '')
                    logger.info(f"[ENHANCED] Successfully handled query with enhanced system")
                    return {
                        'handled': True,
                        'response': response,
                        'metadata': enhanced_result
                    }
            except Exception as e:
                logger.warning(f"[ENHANCED] Enhanced system error, falling back: {e}")

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
                # Enhanced vision query - use multi-space intelligence for comprehensive analysis
                logger.info(
                    f"[ENHANCED VISION] Processing vision query with multi-space intelligence: {command_text}"
                )

                # Check if this needs multi-space analysis
                needs_multi_space = False
                if self.intelligence and hasattr(
                    self.intelligence, "_should_use_multi_space"
                ):
                    needs_multi_space = self.intelligence._should_use_multi_space(
                        command_text
                    )
                    logger.info(
                        f"[ENHANCED VISION] Multi-space analysis needed: {needs_multi_space}"
                    )

                # Debug logging for screenshot type
                logger.info(f"[ENHANCED VISION] Screenshot type: {type(screenshot)}, needs_multi_space: {needs_multi_space}")
                if isinstance(screenshot, dict):
                    logger.info(f"[ENHANCED VISION] Screenshot is dict with {len(screenshot)} keys: {list(screenshot.keys())}")

                if needs_multi_space and isinstance(screenshot, dict):
                    # Multi-space query with multiple screenshots - use enhanced analysis
                    logger.info(
                        f"[ENHANCED VISION] Using enhanced multi-space analysis for {len(screenshot)} spaces"
                    )

                    # Get comprehensive workspace data
                    window_data = await self.intelligence._gather_multi_space_data()

                    # Debug log the window data
                    logger.info(f"[WORKSPACE DEBUG] Window data keys: {window_data.keys() if window_data else 'None'}")
                    if window_data and 'spaces' in window_data:
                        logger.info(f"[WORKSPACE DEBUG] Number of spaces: {len(window_data['spaces'])}")
                        for space_id, space_info in window_data['spaces'].items():
                            if isinstance(space_info, dict):
                                apps = space_info.get('applications', [])
                                primary = space_info.get('primary_app', 'None')
                                space_name = space_info.get('space_name', f'Desktop {space_id}')
                                logger.info(f"[WORKSPACE DEBUG] Space {space_id}: name='{space_name}', primary='{primary}', apps={apps[:2] if apps else []}")

                    # Use enhanced system if available
                    if hasattr(self.intelligence, "multi_space_extension") and hasattr(
                        self.intelligence.multi_space_extension,
                        "generate_enhanced_workspace_response",
                    ):

                        enhanced_response = self.intelligence.multi_space_extension.generate_enhanced_workspace_response(
                            command_text, window_data, screenshot
                        )

                        # Check if enhanced response returned None (signals to use Claude API)
                        if enhanced_response is None:
                            logger.info(
                                "[ENHANCED VISION] Enhanced system returned None, falling back to Claude API for intelligent analysis"
                            )
                            # Use Claude API for intelligent analysis
                            response = await self.intelligence.understand_and_respond(
                                screenshot, command_text
                            )

                            logger.info(f"[WORKSPACE DEBUG] Original response has 'Desktop': {'Desktop' in response if response else 'N/A'}")
                            if response and 'Desktop' in response:
                                logger.info(f"[WORKSPACE DEBUG] Sample of original: {response[:300]}")

                            # Process response to replace generic desktop names with actual workspace names
                            if workspace_detector_available:
                                logger.info("[WORKSPACE DEBUG] Using workspace_detector to process response")
                                processed_response = process_response_with_workspace_names(response, window_data)
                                logger.info(f"[WORKSPACE DEBUG] After detector - has 'Desktop': {'Desktop' in processed_response if processed_response else 'N/A'}")
                                response = processed_response
                            elif workspace_processor_available and window_data:
                                logger.info("[WORKSPACE DEBUG] Using workspace_processor to process response")
                                processed_response = process_jarvis_response(response, window_data)
                                logger.info(f"[WORKSPACE DEBUG] After processor - has 'Desktop': {'Desktop' in processed_response if processed_response else 'N/A'}")
                                response = processed_response

                            logger.info(f"[ENHANCED VISION] Response processed with workspace names")

                            return {
                                "handled": True,
                                "response": response,
                                "claude_api": True,
                                "multi_space": True,
                                "spaces_analyzed": len(screenshot),
                                "monitoring_active": self.monitoring_active,
                                "context": self.intelligence.context.get_temporal_context(),
                            }
                        else:
                            logger.info(
                                f"[ENHANCED VISION] Generated enhanced response: {len(enhanced_response)} chars"
                            )
                            return {
                                "handled": True,
                                "response": enhanced_response,
                                "enhanced_analysis": True,
                                "multi_space": True,
                                "spaces_analyzed": len(screenshot),
                                "monitoring_active": self.monitoring_active,
                                "context": self.intelligence.context.get_temporal_context(),
                            }
                    else:
                        # Fallback to Claude analysis with multi-space prompt
                        logger.info(
                            "[ENHANCED VISION] Enhanced system not available, using Claude multi-space analysis"
                        )
                        response = await self.intelligence.understand_and_respond(
                            screenshot, command_text
                        )

                        # Process response for multi-space queries
                        if workspace_detector_available:
                            response = process_response_with_workspace_names(response, window_data)
                        elif workspace_processor_available and window_data:
                            response = process_jarvis_response(response, window_data)
                else:
                    # Single space or basic query - use standard Claude analysis
                    logger.info("[ENHANCED VISION] Single space analysis")
                    response = await self.intelligence.understand_and_respond(
                        screenshot, command_text
                    )

                    # Even single space responses might contain Desktop references
                    if workspace_detector_available:
                        # Always try to process with workspace detector for any Desktop references
                        try:
                            if hasattr(self.intelligence, "_gather_multi_space_data"):
                                window_data = await self.intelligence._gather_multi_space_data()
                                response = process_response_with_workspace_names(response, window_data)
                            else:
                                # Use detector without window data
                                response = process_response_with_workspace_names(response, None)
                        except:
                            pass  # Fallback silently if can't process
                    elif workspace_processor_available and hasattr(self.intelligence, "_gather_multi_space_data"):
                        try:
                            window_data = await self.intelligence._gather_multi_space_data()
                            if window_data:
                                response = process_jarvis_response(response, window_data)
                        except:
                            pass  # Fallback silently if can't get window data

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
                    # Enhanced vision query - check for multi-space needs
                    logger.info(
                        f"[ENHANCED VISION FALLBACK] Processing vision query: {command_text}"
                    )

                    # Check if this needs multi-space analysis
                    needs_multi_space = False
                    if self.intelligence and hasattr(
                        self.intelligence, "_should_use_multi_space"
                    ):
                        needs_multi_space = self.intelligence._should_use_multi_space(
                            command_text
                        )

                    if needs_multi_space and isinstance(screenshot, dict):
                        # Multi-space query - use enhanced analysis
                        logger.info(
                            f"[ENHANCED VISION FALLBACK] Using enhanced multi-space analysis for {len(screenshot)} spaces"
                        )

                        window_data = await self.intelligence._gather_multi_space_data()

                        if hasattr(
                            self.intelligence, "multi_space_extension"
                        ) and hasattr(
                            self.intelligence.multi_space_extension,
                            "generate_enhanced_workspace_response",
                        ):

                            enhanced_response = self.intelligence.multi_space_extension.generate_enhanced_workspace_response(
                                command_text, window_data, screenshot
                            )
                            logger.info(
                                f"[ENHANCED VISION FALLBACK] Generated enhanced response: {len(enhanced_response)} chars"
                            )

                            return {
                                "handled": True,
                                "response": enhanced_response,
                                "enhanced_analysis": True,
                                "multi_space": True,
                                "spaces_analyzed": len(screenshot),
                                "monitoring_active": self.monitoring_active,
                                "context": self.intelligence.context.get_temporal_context(),
                            }

                    # Pure vision query - use standard Claude analysis
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


        # Fallback: If we reach here, something went wrong
        logger.warning(f"[VISION] No handler processed the command: {command_text}")
        return {
            "handled": True,
            "response": "Let me analyze your desktop spaces for you, Sir.",
            "fallback": True
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

            # Try Yabai multi-space intelligence first
            if needs_multi_space and self.yabai_detector and self.workspace_analyzer and self.space_response_generator:
                try:
                    logger.info("[YABAI] Using Yabai-based multi-space intelligence")

                    # Check Yabai availability
                    status = await self.yabai_detector.check_availability()

                    if status == YabaiStatus.AVAILABLE:
                        # Get workspace data from Yabai
                        workspace_data = await self.yabai_detector.get_workspace_data()
                        spaces = workspace_data['spaces']
                        windows = workspace_data['windows']

                        # Analyze workspace activity
                        analysis = self.workspace_analyzer.analyze(spaces, windows)

                        # Generate natural language response
                        response = self.space_response_generator.generate_overview_response(
                            analysis, include_details=True
                        )

                        # Get performance stats
                        perf_stats = self.yabai_detector.get_performance_stats()
                        logger.info(f"[YABAI] Performance: {perf_stats}")

                        return {
                            "handled": True,
                            "response": response,
                            "pure_intelligence": True,
                            "yabai_powered": True,
                            "monitoring_active": self.monitoring_active,
                            "context": self.intelligence.context.get_temporal_context() if self.intelligence else {},
                            "analysis_metadata": {
                                "total_spaces": analysis.total_spaces,
                                "active_spaces": analysis.active_spaces,
                                "unique_applications": analysis.unique_applications,
                                "detected_project": analysis.detected_project,
                                "yabai_status": status.value,
                                "performance": perf_stats
                            }
                        }
                    else:
                        # Yabai not available - provide installation guidance
                        logger.warning(f"[YABAI] Not available (status: {status.value})")
                        response = self.space_response_generator.generate_yabai_installation_response(status)

                        return {
                            "handled": True,
                            "response": response,
                            "yabai_status": status.value,
                            "monitoring_active": self.monitoring_active,
                        }

                except Exception as e:
                    logger.error(f"[YABAI] Error using Yabai system: {e}", exc_info=True)
                    # Fall through to Claude-based analysis

            # Use enhanced multi-space intelligence if available (fallback)
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

    async def _handle_yabai_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Handle metadata-only query using Yabai (no screenshots)
        Fast path for workspace overview queries
        """
        try:
            if not self.yabai_detector or not self.workspace_analyzer:
                raise ValueError("Yabai system not available")

            logger.info("[INTELLIGENT] Handling metadata-only query with Yabai")

            # Get workspace data from Yabai
            workspace_data = await self.yabai_detector.get_all_spaces()
            windows_data = await self.yabai_detector.get_all_windows()

            # Analyze workspace
            if self.workspace_analyzer:
                analysis = self.workspace_analyzer.analyze(workspace_data, windows_data)

                # Generate natural response
                if self.space_response_generator:
                    response = self.space_response_generator.generate_overview_response(
                        analysis,
                        include_details=True
                    )
                else:
                    # Fallback to basic response
                    response = f"You have {analysis.total_spaces} desktop spaces with {analysis.active_spaces} active workspaces."
            else:
                # Basic fallback response
                response = f"You have {len(workspace_data)} desktop spaces visible."

            return response

        except Exception as e:
            logger.error(f"[INTELLIGENT] Yabai query handler error: {e}")
            raise

    async def _handle_vision_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        multi_space: bool = False
    ) -> str:
        """
        Handle visual analysis query with current screen capture
        """
        try:
            logger.info(f"[INTELLIGENT] Handling vision query (multi_space: {multi_space})")

            # Capture screen
            screenshot = await self._capture_screen(multi_space=multi_space)

            if not screenshot:
                return "I couldn't capture your screen, Sir. Please check screen recording permissions."

            # Use Claude vision intelligence to analyze
            if self.intelligence:
                response = await self.intelligence.understand_and_respond(
                    screenshot, query
                )
                return response
            else:
                return "Vision intelligence not initialized yet, Sir."

        except Exception as e:
            logger.error(f"[INTELLIGENT] Vision query handler error: {e}")
            raise

    async def _handle_multi_space_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Handle deep analysis query with multi-space capture + Yabai metadata
        """
        try:
            logger.info("[INTELLIGENT] Handling multi-space deep analysis query")

            # Capture all desktop spaces
            screenshots = await self._capture_screen(multi_space=True)

            if not screenshots:
                return "I couldn't capture your desktop spaces, Sir."

            # Get Yabai metadata for context
            window_data = None
            if self.intelligence and hasattr(self.intelligence, "_gather_multi_space_data"):
                window_data = await self.intelligence._gather_multi_space_data()

            # Use enhanced multi-space analysis if available
            if (
                hasattr(self.intelligence, "multi_space_extension")
                and hasattr(
                    self.intelligence.multi_space_extension,
                    "generate_enhanced_workspace_response"
                )
            ):
                response = self.intelligence.multi_space_extension.generate_enhanced_workspace_response(
                    query, window_data, screenshots
                )

                # If enhanced system returns None, use Claude API
                if response is None and self.intelligence:
                    response = await self.intelligence.understand_and_respond(
                        screenshots, query
                    )

                    # Process with workspace names
                    if workspace_detector_available and window_data:
                        response = process_response_with_workspace_names(response, window_data)

                return response or "I analyzed your workspaces, Sir."

            # Fallback to standard Claude analysis
            elif self.intelligence:
                response = await self.intelligence.understand_and_respond(
                    screenshots, query
                )

                # Process with workspace names
                if workspace_detector_available and window_data:
                    response = process_response_with_workspace_names(response, window_data)

                return response

            return "Multi-space analysis not available, Sir."

        except Exception as e:
            logger.error(f"[INTELLIGENT] Multi-space query handler error: {e}")
            raise

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
        stats = {
            "conversation_length": 0,
            "monitoring_active": self.monitoring_active,
            "workflow_state": "unknown",
            "emotional_state": "neutral",
        }

        # Legacy intelligence stats
        if self.intelligence and self.intelligence.context:
            stats.update({
                "conversation_length": len(self.intelligence.context.history),
                "workflow_state": self.intelligence.context.workflow_state,
                "emotional_state": (
                    self.intelligence.context.emotional_context.value
                    if self.intelligence.context.emotional_context
                    else "neutral"
                ),
            })

        # Add intelligent system stats if available
        if intelligent_system_available and self.context_manager:
            try:
                intelligent_stats = self.context_manager.get_session_stats()
                stats['intelligent_system'] = intelligent_stats
            except Exception as e:
                logger.warning(f"Could not get intelligent system stats: {e}")

        return stats

    async def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report from intelligent system

        Returns:
            Performance report with metrics and insights
        """
        if not intelligent_system_available or not self.performance_monitor:
            return {
                "available": False,
                "message": "Intelligent system not available"
            }

        try:
            # Collect latest metrics
            await self.performance_monitor.collect_metrics()

            # Generate report
            report = self.performance_monitor.generate_report()

            # Add insights
            report['insights'] = self.performance_monitor.get_performance_insights()

            # Add real-time stats
            report['real_time'] = self.performance_monitor.get_real_time_stats()

            return {
                "available": True,
                "report": report
            }

        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {
                "available": False,
                "error": str(e)
            }

    async def get_classification_stats(self) -> Dict[str, Any]:
        """
        Get classification statistics

        Returns:
            Classification accuracy and routing stats
        """
        if not intelligent_system_available:
            return {"available": False}

        try:
            stats = {}

            # Classifier stats
            if self.classifier:
                stats['classifier'] = self.classifier.get_performance_stats()

            # Router stats
            if self.smart_router:
                stats['router'] = self.smart_router.get_routing_stats()

            # Learning stats
            if self.learning_system:
                stats['learning'] = self.learning_system.get_accuracy_report()

            # Context stats
            if self.context_manager:
                stats['context'] = self.context_manager.get_session_stats()
                stats['user_preferences'] = self.context_manager.get_user_preferences()

            # Proactive suggestions stats
            if self.proactive_system:
                stats['proactive_suggestions'] = self.proactive_system.get_statistics()

            # A/B testing stats
            if self.smart_router and hasattr(self.smart_router, 'ab_test') and self.smart_router.ab_test:
                stats['ab_testing'] = self.smart_router.get_ab_test_report()

            return {
                "available": True,
                "stats": stats
            }

        except Exception as e:
            logger.error(f"Failed to get classification stats: {e}")
            return {
                "available": False,
                "error": str(e)
            }

    async def get_proactive_suggestions(self) -> Dict[str, Any]:
        """
        Get proactive suggestions based on current state

        Returns:
            Dict with suggestions if available
        """
        if not proactive_suggestions_available or not self.proactive_system:
            return {"available": False}

        try:
            # Get current context
            context = {}
            if self.context_manager:
                context = self.context_manager.get_context_for_query("")

            # Get Yabai data if available
            yabai_data = None
            if self.yabai_detector:
                try:
                    yabai_data = {
                        'spaces': await self.yabai_detector.get_all_spaces() if hasattr(self.yabai_detector, 'get_all_spaces') else {},
                        'active_space': context.get('active_space')
                    }
                except Exception:
                    pass  # Continue without Yabai data

            # Analyze and get suggestion
            suggestion = await self.proactive_system.analyze_and_suggest(context, yabai_data)

            if suggestion:
                return {
                    "available": True,
                    "has_suggestion": True,
                    "suggestion": {
                        "id": suggestion.suggestion_id,
                        "type": suggestion.type.value,
                        "priority": suggestion.priority.value,
                        "message": suggestion.message,
                        "action": suggestion.action
                    }
                }
            else:
                return {
                    "available": True,
                    "has_suggestion": False
                }

        except Exception as e:
            logger.error(f"Failed to get proactive suggestions: {e}")
            return {
                "available": False,
                "error": str(e)
            }

    async def respond_to_suggestion(
        self,
        suggestion_id: str,
        accepted: bool
    ) -> Dict[str, Any]:
        """
        Handle user's response to a proactive suggestion

        Args:
            suggestion_id: ID of the suggestion
            accepted: Whether user accepted or dismissed

        Returns:
            Dict with response or action result
        """
        if not proactive_suggestions_available or not self.proactive_system:
            return {"available": False}

        try:
            # Record user response
            await self.proactive_system.record_user_response(suggestion_id, accepted)

            if accepted:
                # Find the suggestion and execute its action
                suggestions = self.proactive_system.get_active_suggestions()
                suggestion = next(
                    (s for s in suggestions if s.suggestion_id == suggestion_id),
                    None
                )

                if suggestion:
                    # Execute the suggested action
                    action = suggestion.action

                    if action.startswith("analyze_space_"):
                        space_id = action.split("_")[-1]
                        # Analyze the specific space
                        return {
                            "accepted": True,
                            "action": "analyze_space",
                            "space_id": space_id,
                            "message": f"Analyzing Space {space_id}..."
                        }

                    elif action == "workspace_summary":
                        # Generate workspace summary
                        return {
                            "accepted": True,
                            "action": "workspace_summary",
                            "message": "Generating workspace summary..."
                        }

                    elif action == "workspace_overview":
                        # Generate workspace overview
                        return {
                            "accepted": True,
                            "action": "workspace_overview",
                            "message": "Here's your workspace overview..."
                        }

                    elif action == "analyze_workflow":
                        # Analyze workflow
                        return {
                            "accepted": True,
                            "action": "analyze_workflow",
                            "message": "Analyzing your workflow patterns..."
                        }

                    else:
                        return {
                            "accepted": True,
                            "action": action,
                            "message": "Processing your request..."
                        }

            return {
                "accepted": accepted,
                "message": "Dismissed" if not accepted else "Accepted"
            }

        except Exception as e:
            logger.error(f"Failed to respond to suggestion: {e}")
            return {
                "error": str(e)
            }


# Singleton instance
vision_command_handler = VisionCommandHandler()
