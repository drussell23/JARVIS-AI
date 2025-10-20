"""
Unified Command Processor - Dynamic command interpretation with zero hardcoding
Learns from the system and adapts to any environment
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime
import json
from pathlib import Path
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

# Import manual unlock handler
try:
    from api.manual_unlock_handler import handle_manual_unlock
except ImportError:
    handle_manual_unlock = None
    logger.warning("Manual unlock handler not available")


class DynamicPatternLearner:
    """Learns command patterns from usage and system analysis"""

    def __init__(self):
        self.learned_patterns = defaultdict(list)
        self.app_verbs = set()
        self.system_verbs = set()
        self.query_indicators = set()
        self.learned_apps = set()
        self.pattern_confidence = defaultdict(float)
        self._initialize_base_patterns()
        self._learn_from_system()

    def _initialize_base_patterns(self):
        """Initialize with minimal base patterns that will be expanded"""
        # These are just seeds - the system will learn more
        self.app_verbs = {"open", "close", "launch", "quit", "start", "kill"}
        self.system_verbs = {"set", "adjust", "toggle", "take", "enable", "disable"}
        self.query_indicators = {
            "what",
            "who",
            "where",
            "when",
            "why",
            "how",
            "is",
            "are",
            "can",
        }

    def _learn_from_system(self):
        """Learn available applications and commands from the system"""
        try:
            # Dynamically discover installed applications
            from system_control.dynamic_app_controller import get_dynamic_app_controller

            controller = get_dynamic_app_controller()

            # Learn all installed apps
            if hasattr(controller, "installed_apps_cache"):
                for app_key, app_info in controller.installed_apps_cache.items():
                    self.learned_apps.add(app_info["name"].lower())
                    # Also learn variations
                    self.learned_apps.add(app_key.lower())

            logger.info(f"Learned {len(self.learned_apps)} applications from system")

        except Exception as e:
            logger.debug(f"Could not learn from system controller: {e}")

    def learn_pattern(self, command: str, command_type: str, success: bool):
        """Learn from command execution results"""
        words = command.lower().split()
        if success and len(words) > 0:
            # Learn verb patterns
            first_word = words[0]
            if command_type == "system" and first_word not in self.system_verbs:
                self.system_verbs.add(first_word)
                self.pattern_confidence[f"verb_{first_word}"] += 0.1

            # Learn app names from successful commands
            if command_type == "system" and any(
                verb in words for verb in self.app_verbs
            ):
                # Extract potential app names
                for i, word in enumerate(words):
                    if word in self.app_verbs and i + 1 < len(words):
                        potential_app = words[i + 1]
                        if (
                            potential_app not in self.app_verbs
                            and potential_app not in self.system_verbs
                        ):
                            self.learned_apps.add(potential_app)

    def is_learned_app(self, word: str) -> bool:
        """Check if a word is a learned app name"""
        return word.lower() in self.learned_apps

    def get_command_patterns(self, command_type: str) -> List[str]:
        """Get learned patterns for a command type"""
        return self.learned_patterns.get(command_type, [])


class CommandType(Enum):
    """Types of commands JARVIS can handle"""

    VISION = "vision"
    SYSTEM = "system"
    WEATHER = "weather"
    COMMUNICATION = "communication"
    AUTONOMY = "autonomy"
    QUERY = "query"
    COMPOUND = "compound"
    META = "meta"
    VOICE_UNLOCK = "voice_unlock"
    DOCUMENT = "document"
    DISPLAY = "display"
    UNKNOWN = "unknown"


@dataclass
class UnifiedContext:
    """Single context shared across all command processing"""

    conversation_history: List[Dict[str, Any]]
    current_visual: Optional[Dict[str, Any]] = None
    last_entity: Optional[Dict[str, Any]] = None  # For "it/that" resolution
    active_monitoring: bool = False
    user_preferences: Optional[Dict[str, Any]] = None
    system_state: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.system_state is None:
            self.system_state = {}

    def resolve_reference(self, text: str) -> Tuple[Optional[str], float]:
        """Resolve 'it', 'that', 'this' to actual entities"""
        reference_words = ["it", "that", "this", "them"]

        for word in reference_words:
            if word in text.lower():
                if self.last_entity:
                    # Check how recent the entity is
                    if "timestamp" in self.last_entity:
                        age = (datetime.now() - self.last_entity["timestamp"]).seconds
                        confidence = 0.9 if age < 30 else 0.7 if age < 60 else 0.5
                    else:
                        confidence = 0.8
                    return self.last_entity.get("value", ""), confidence

                # Check visual context
                if self.current_visual:
                    return self.current_visual.get("focused_element", ""), 0.7

        return None, 0.0

    def update_from_command(self, command_type: CommandType, result: Dict[str, Any]):
        """Update context based on command execution"""
        self.conversation_history.append(
            {"type": command_type.value, "result": result, "timestamp": datetime.now()}
        )

        # Extract entities for future reference
        if command_type == CommandType.VISION and "elements" in result:
            if result["elements"]:
                self.last_entity = {
                    "value": result["elements"][0],
                    "timestamp": datetime.now(),
                    "type": "visual_element",
                }

        # Update visual context
        if command_type == CommandType.VISION:
            self.current_visual = result.get("visual_context", {})


class UnifiedCommandProcessor:
    """Dynamic command processor that learns and adapts"""

    def __init__(self, claude_api_key: Optional[str] = None, app=None):
        self.context = UnifiedContext(conversation_history=[])
        self.handlers = {}
        self.pattern_learner = DynamicPatternLearner()
        self.command_stats = defaultdict(int)
        self.success_patterns = defaultdict(list)
        self._initialize_handlers()
        self.claude_api_key = claude_api_key
        self._app = app  # Store app reference for accessing app.state
        self._load_learned_data()

        # Initialize multi-space context graph for advanced context tracking
        self.context_graph = None

        # Initialize resolver systems
        self.contextual_resolver = None   # Space/monitor resolution
        self.implicit_resolver = None     # Entity/intent resolution
        self.multi_space_handler = None   # Multi-space query handler
        self.temporal_handler = None      # Temporal query handler (change detection, error tracking, timeline)
        self.query_complexity_manager = None  # Query complexity classification and routing
        self.medium_complexity_handler = None  # Medium complexity (Level 2) query execution
        self.display_reference_handler = None  # Display voice command resolution
        self._initialize_resolvers()

    def _initialize_resolvers(self):
        """Initialize both resolver systems for comprehensive query understanding"""

        # Step 1: Initialize MultiSpaceContextGraph (required for implicit resolver)
        try:
            from core.context.multi_space_context_graph import MultiSpaceContextGraph
            self.context_graph = MultiSpaceContextGraph()
            logger.info("[UNIFIED] ✅ MultiSpaceContextGraph initialized")
        except ImportError as e:
            logger.warning(f"[UNIFIED] MultiSpaceContextGraph not available: {e}")
            self.context_graph = None
        except Exception as e:
            logger.error(f"[UNIFIED] Failed to initialize context graph: {e}")
            self.context_graph = None

        # Step 2: Initialize ImplicitReferenceResolver (entity/intent resolution)
        if self.context_graph:
            try:
                from core.nlp.implicit_reference_resolver import initialize_implicit_resolver
                self.implicit_resolver = initialize_implicit_resolver(self.context_graph)
                logger.info("[UNIFIED] ✅ ImplicitReferenceResolver initialized")
            except ImportError as e:
                logger.warning(f"[UNIFIED] ImplicitReferenceResolver not available: {e}")
                self.implicit_resolver = None
            except Exception as e:
                logger.error(f"[UNIFIED] Failed to initialize implicit resolver: {e}")
                self.implicit_resolver = None
        else:
            logger.info("[UNIFIED] Skipping implicit resolver (no context graph)")
            self.implicit_resolver = None

        # Step 3: Initialize ContextualQueryResolver (space/monitor resolution)
        try:
            from context_intelligence.resolvers import get_contextual_resolver
            self.contextual_resolver = get_contextual_resolver()
            logger.info("[UNIFIED] ✅ ContextualQueryResolver initialized")
        except ImportError as e:
            logger.warning(f"[UNIFIED] ContextualQueryResolver not available: {e}")
            self.contextual_resolver = None
        except Exception as e:
            logger.error(f"[UNIFIED] Failed to initialize contextual resolver: {e}")
            self.contextual_resolver = None

        # Step 4: Initialize MultiSpaceQueryHandler (parallel space analysis)
        if self.context_graph:
            try:
                from context_intelligence.handlers import initialize_multi_space_handler
                self.multi_space_handler = initialize_multi_space_handler(
                    context_graph=self.context_graph,
                    implicit_resolver=self.implicit_resolver,
                    contextual_resolver=self.contextual_resolver
                )
                logger.info("[UNIFIED] ✅ MultiSpaceQueryHandler initialized")
            except ImportError as e:
                logger.warning(f"[UNIFIED] MultiSpaceQueryHandler not available: {e}")
                self.multi_space_handler = None
            except Exception as e:
                logger.error(f"[UNIFIED] Failed to initialize multi-space handler: {e}")
                self.multi_space_handler = None
        else:
            logger.info("[UNIFIED] Skipping multi-space handler (no context graph)")
            self.multi_space_handler = None

        # Step 5: Initialize TemporalQueryHandler (PLACEHOLDER - will be initialized after dependencies in Step 6.13)
        # This is a placeholder to maintain backward compatibility with old code
        # The actual initialization happens in Step 6.13 after ProactiveMonitoringManager and ChangeDetectionManager are ready
        self.temporal_handler = None
        logger.info("[UNIFIED] TemporalQueryHandler initialization deferred to Step 6.13")

        # Step 6: Initialize QueryComplexityManager (query classification and routing)
        try:
            from context_intelligence.handlers import initialize_query_complexity_manager

            self.query_complexity_manager = initialize_query_complexity_manager(
                implicit_resolver=self.implicit_resolver
            )
            logger.info("[UNIFIED] ✅ QueryComplexityManager initialized")
        except ImportError as e:
            logger.warning(f"[UNIFIED] QueryComplexityManager not available: {e}")
            self.query_complexity_manager = None
        except Exception as e:
            logger.error(f"[UNIFIED] Failed to initialize query complexity manager: {e}")
            self.query_complexity_manager = None

        # Step 6.5: Initialize ResponseStrategyManager (response quality enhancement)
        try:
            from context_intelligence.managers import (
                initialize_response_strategy_manager,
                ResponseQuality
            )

            self.response_strategy_manager = initialize_response_strategy_manager(
                vision_client=None,  # Will be set if vision client available
                min_quality=ResponseQuality.SPECIFIC
            )
            logger.info("[UNIFIED] ✅ ResponseStrategyManager initialized")
        except ImportError as e:
            logger.warning(f"[UNIFIED] ResponseStrategyManager not available: {e}")
            self.response_strategy_manager = None
        except Exception as e:
            logger.error(f"[UNIFIED] Failed to initialize response strategy manager: {e}")
            self.response_strategy_manager = None

        # Step 6.6: Initialize ContextAwareResponseManager (conversation context tracking)
        try:
            from context_intelligence.managers import initialize_context_aware_response_manager

            self.context_aware_manager = initialize_context_aware_response_manager(
                implicit_resolver=self.implicit_resolver,
                max_history=10,
                context_ttl=300.0  # 5 minutes
            )
            logger.info("[UNIFIED] ✅ ContextAwareResponseManager initialized")
        except ImportError as e:
            logger.warning(f"[UNIFIED] ContextAwareResponseManager not available: {e}")
            self.context_aware_manager = None
        except Exception as e:
            logger.error(f"[UNIFIED] Failed to initialize context-aware manager: {e}")
            self.context_aware_manager = None

        # Step 6.7: Initialize ProactiveSuggestionManager (next-step suggestions)
        try:
            from context_intelligence.managers import initialize_proactive_suggestion_manager

            # Get conversation tracker from context-aware manager
            conversation_tracker = None
            if self.context_aware_manager:
                conversation_tracker = self.context_aware_manager.conversation_tracker

            self.proactive_suggestion_manager = initialize_proactive_suggestion_manager(
                conversation_tracker=conversation_tracker,
                implicit_resolver=self.implicit_resolver,
                max_suggestions=2,
                confidence_threshold=0.5
            )
            logger.info("[UNIFIED] ✅ ProactiveSuggestionManager initialized")
        except ImportError as e:
            logger.warning(f"[UNIFIED] ProactiveSuggestionManager not available: {e}")
            self.proactive_suggestion_manager = None
        except Exception as e:
            logger.error(f"[UNIFIED] Failed to initialize proactive suggestion manager: {e}")
            self.proactive_suggestion_manager = None

        # Step 6.8: Initialize ConfidenceManager (confidence-based response formatting)
        try:
            from context_intelligence.managers import initialize_confidence_manager

            self.confidence_manager = initialize_confidence_manager(
                include_visual_indicators=True,  # Include ✅ ⚠️ ❓
                include_reasoning=True,  # Include reasoning for non-high confidence
                min_confidence_for_high=0.8,
                min_confidence_for_medium=0.5
            )
            logger.info("[UNIFIED] ✅ ConfidenceManager initialized")
        except ImportError as e:
            logger.warning(f"[UNIFIED] ConfidenceManager not available: {e}")
            self.confidence_manager = None
        except Exception as e:
            logger.error(f"[UNIFIED] Failed to initialize confidence manager: {e}")
            self.confidence_manager = None

        # Step 6.9: Initialize MultiMonitorManager (multi-monitor support with spatial awareness)
        try:
            from context_intelligence.managers import initialize_multi_monitor_manager

            # Get conversation tracker from context-aware manager
            conversation_tracker = None
            if self.context_aware_manager:
                conversation_tracker = self.context_aware_manager.conversation_tracker

            self.multi_monitor_manager = initialize_multi_monitor_manager(
                implicit_resolver=self.implicit_resolver,
                conversation_tracker=conversation_tracker,
                auto_refresh_interval=30.0  # Refresh monitor layout every 30 seconds
            )
            logger.info("[UNIFIED] ✅ MultiMonitorManager initialized (will be fully initialized on first use)")
        except ImportError as e:
            logger.warning(f"[UNIFIED] MultiMonitorManager not available: {e}")
            self.multi_monitor_manager = None
        except Exception as e:
            logger.error(f"[UNIFIED] Failed to initialize multi-monitor manager: {e}")
            self.multi_monitor_manager = None

        # Step 6.10: Initialize MultiMonitorQueryHandler (multi-monitor query processing)
        try:
            from context_intelligence.handlers import initialize_multi_monitor_query_handler
            from context_intelligence.managers import (
                get_capture_strategy_manager,
                get_ocr_strategy_manager
            )

            self.multi_monitor_query_handler = initialize_multi_monitor_query_handler(
                multi_monitor_manager=self.multi_monitor_manager,
                capture_manager=get_capture_strategy_manager(),
                ocr_manager=get_ocr_strategy_manager(),
                implicit_resolver=self.implicit_resolver
            )
            logger.info("[UNIFIED] ✅ MultiMonitorQueryHandler initialized")
        except ImportError as e:
            logger.warning(f"[UNIFIED] MultiMonitorQueryHandler not available: {e}")
            self.multi_monitor_query_handler = None
        except Exception as e:
            logger.error(f"[UNIFIED] Failed to initialize multi-monitor query handler: {e}")
            self.multi_monitor_query_handler = None

        # Step 6.11: Initialize ChangeDetectionManager (temporal & state-based change detection)
        try:
            from context_intelligence.managers import initialize_change_detection_manager
            from pathlib import Path

            # Get conversation tracker from context-aware manager
            conversation_tracker = None
            if self.context_aware_manager:
                conversation_tracker = self.context_aware_manager.conversation_tracker

            self.change_detection_manager = initialize_change_detection_manager(
                cache_dir=Path.home() / ".jarvis" / "change_cache",
                cache_ttl=3600.0,  # 1 hour
                max_cache_size=100,
                implicit_resolver=self.implicit_resolver,
                conversation_tracker=conversation_tracker
            )
            logger.info("[UNIFIED] ✅ ChangeDetectionManager initialized")
        except ImportError as e:
            logger.warning(f"[UNIFIED] ChangeDetectionManager not available: {e}")
            self.change_detection_manager = None
        except Exception as e:
            logger.error(f"[UNIFIED] Failed to initialize change detection manager: {e}")
            self.change_detection_manager = None

        # Step 6.12: Initialize ProactiveMonitoringManager (autonomous monitoring & alerts)
        try:
            from context_intelligence.managers import initialize_proactive_monitoring_manager, get_capture_strategy_manager, get_ocr_strategy_manager

            # Get conversation tracker from context-aware manager
            conversation_tracker = None
            if self.context_aware_manager:
                conversation_tracker = self.context_aware_manager.conversation_tracker

            # Alert callback to display alerts to user
            def alert_callback(alert):
                logger.info(f"[ALERT] {alert.message}")
                # Could be extended to notify user via TTS, notifications, etc.

            self.proactive_monitoring_manager = initialize_proactive_monitoring_manager(
                change_detection_manager=self.change_detection_manager,
                capture_manager=get_capture_strategy_manager(),
                ocr_manager=get_ocr_strategy_manager(),
                implicit_resolver=self.implicit_resolver,
                conversation_tracker=conversation_tracker,
                default_interval=10.0,  # Check every 10 seconds
                alert_callback=alert_callback
            )
            logger.info("[UNIFIED] ✅ ProactiveMonitoringManager initialized (not started - use start_monitoring())")
        except ImportError as e:
            logger.warning(f"[UNIFIED] ProactiveMonitoringManager not available: {e}")
            self.proactive_monitoring_manager = None
        except Exception as e:
            logger.error(f"[UNIFIED] Failed to initialize proactive monitoring manager: {e}")
            self.proactive_monitoring_manager = None

        # Step 6.13: Initialize Enhanced TemporalQueryHandler (v2.0 with ProactiveMonitoring integration)
        try:
            from context_intelligence.handlers import initialize_temporal_query_handler

            # Get conversation tracker from context-aware manager
            conversation_tracker = None
            if self.context_aware_manager:
                conversation_tracker = self.context_aware_manager.conversation_tracker

            self.temporal_handler = initialize_temporal_query_handler(
                proactive_monitoring_manager=self.proactive_monitoring_manager,
                change_detection_manager=self.change_detection_manager,
                implicit_resolver=self.implicit_resolver,
                conversation_tracker=conversation_tracker
            )

            # Register alert callback for ProactiveMonitoringManager to feed alerts to TemporalHandler
            if self.proactive_monitoring_manager and self.temporal_handler:
                # Update alert callback to also register alerts with TemporalHandler
                original_callback = alert_callback

                def enhanced_alert_callback(alert):
                    original_callback(alert)  # Log to console
                    self.temporal_handler.register_monitoring_alert({
                        'space_id': alert.space_id,
                        'event_type': alert.event_type.value,
                        'message': alert.message,
                        'priority': alert.priority.value,
                        'timestamp': alert.timestamp,
                        'metadata': alert.metadata
                    })

                # Re-initialize ProactiveMonitoringManager with enhanced callback
                self.proactive_monitoring_manager = initialize_proactive_monitoring_manager(
                    change_detection_manager=self.change_detection_manager,
                    capture_manager=get_capture_strategy_manager(),
                    ocr_manager=get_ocr_strategy_manager(),
                    implicit_resolver=self.implicit_resolver,
                    conversation_tracker=conversation_tracker,
                    default_interval=10.0,
                    alert_callback=enhanced_alert_callback
                )

            logger.info("[UNIFIED] ✅ Enhanced TemporalQueryHandler v2.0 initialized with ProactiveMonitoring integration")
        except ImportError as e:
            logger.warning(f"[UNIFIED] Enhanced TemporalQueryHandler not available: {e}")
            self.temporal_handler = None
        except Exception as e:
            logger.error(f"[UNIFIED] Failed to initialize enhanced temporal handler: {e}")
            self.temporal_handler = None

        # Step 7: Initialize MediumComplexityHandler (Level 2 query execution)
        try:
            from context_intelligence.handlers import initialize_medium_complexity_handler
            from context_intelligence.managers import (
                get_capture_strategy_manager,
                get_ocr_strategy_manager
            )

            self.medium_complexity_handler = initialize_medium_complexity_handler(
                capture_manager=get_capture_strategy_manager(),
                ocr_manager=get_ocr_strategy_manager(),
                response_manager=self.response_strategy_manager,
                context_aware_manager=self.context_aware_manager,
                proactive_suggestion_manager=self.proactive_suggestion_manager,
                confidence_manager=self.confidence_manager,
                multi_monitor_manager=self.multi_monitor_manager,
                multi_monitor_query_handler=self.multi_monitor_query_handler,
                implicit_resolver=self.implicit_resolver
            )
            logger.info("[UNIFIED] ✅ MediumComplexityHandler initialized")
        except ImportError as e:
            logger.warning(f"[UNIFIED] MediumComplexityHandler not available: {e}")
            self.medium_complexity_handler = None
        except Exception as e:
            logger.error(f"[UNIFIED] Failed to initialize medium complexity handler: {e}")
            self.medium_complexity_handler = None

        # Step 8: Initialize DisplayReferenceHandler (voice command → display connection)
        try:
            from context_intelligence.handlers.display_reference_handler import initialize_display_reference_handler

            self.display_reference_handler = initialize_display_reference_handler(
                implicit_resolver=self.implicit_resolver,
                display_monitor=None  # Will be integrated with advanced_display_monitor later
            )
            logger.info("[UNIFIED] ✅ DisplayReferenceHandler initialized")
        except ImportError as e:
            logger.warning(f"[UNIFIED] DisplayReferenceHandler not available: {e}")
            self.display_reference_handler = None
        except Exception as e:
            logger.error(f"[UNIFIED] Failed to initialize display reference handler: {e}")
            self.display_reference_handler = None

        # Step 8: Initialize ComplexComplexityHandler (Level 3 query execution)
        try:
            from context_intelligence.handlers import (
                initialize_complex_complexity_handler,
                get_predictive_handler,
                get_action_query_handler
            )
            from context_intelligence.managers import (
                get_capture_strategy_manager,
                get_ocr_strategy_manager
            )

            self.complex_complexity_handler = initialize_complex_complexity_handler(
                temporal_handler=self.temporal_handler,
                multi_space_handler=self.multi_space_handler,
                predictive_handler=get_predictive_handler(),
                capture_manager=get_capture_strategy_manager(),
                ocr_manager=get_ocr_strategy_manager(),
                multi_monitor_manager=self.multi_monitor_manager,
                implicit_resolver=self.implicit_resolver,
                cache_ttl=60.0,
                max_concurrent_captures=5
            )
            logger.info("[UNIFIED] ✅ ComplexComplexityHandler initialized")
        except ImportError as e:
            logger.warning(f"[UNIFIED] ComplexComplexityHandler not available: {e}")
            self.complex_complexity_handler = None
        except Exception as e:
            logger.error(f"[UNIFIED] Failed to initialize complex complexity handler: {e}")
            self.complex_complexity_handler = None

        # Log integration status
        resolvers_active = []
        if self.context_graph:
            resolvers_active.append("ContextGraph")
        if self.implicit_resolver:
            resolvers_active.append("ImplicitResolver")
        if self.contextual_resolver:
            resolvers_active.append("ContextualResolver")
        if self.multi_space_handler:
            resolvers_active.append("MultiSpaceHandler")
        if self.temporal_handler:
            resolvers_active.append("TemporalHandler")
        if self.query_complexity_manager:
            resolvers_active.append("QueryComplexityManager")
        if self.response_strategy_manager:
            resolvers_active.append("ResponseStrategyManager")
        if self.context_aware_manager:
            resolvers_active.append("ContextAwareManager")
        if self.proactive_suggestion_manager:
            resolvers_active.append("ProactiveSuggestionManager")
        if self.confidence_manager:
            resolvers_active.append("ConfidenceManager")
        if self.multi_monitor_manager:
            resolvers_active.append("MultiMonitorManager")
        if self.multi_monitor_query_handler:
            resolvers_active.append("MultiMonitorQueryHandler")
        if self.change_detection_manager:
            resolvers_active.append("ChangeDetectionManager")
        if self.proactive_monitoring_manager:
            resolvers_active.append("ProactiveMonitoringManager")
        if self.medium_complexity_handler:
            resolvers_active.append("MediumComplexityHandler")
        if self.complex_complexity_handler:
            resolvers_active.append("ComplexComplexityHandler")

        if resolvers_active:
            logger.info(f"[UNIFIED] 🎯 Active resolvers: {', '.join(resolvers_active)}")
        else:
            logger.warning("[UNIFIED] ⚠️  No resolvers available - queries will use basic processing")

    def _load_learned_data(self):
        """Load previously learned patterns and statistics"""
        try:
            data_dir = Path.home() / ".jarvis" / "learning"
            data_dir.mkdir(parents=True, exist_ok=True)

            stats_file = data_dir / "command_stats.json"
            if stats_file.exists():
                with open(stats_file, "r") as f:
                    self.command_stats = defaultdict(int, json.load(f))

            patterns_file = data_dir / "success_patterns.json"
            if patterns_file.exists():
                with open(patterns_file, "r") as f:
                    self.success_patterns = defaultdict(list, json.load(f))

        except Exception as e:
            logger.debug(f"Could not load learned data: {e}")

    def _save_learned_data(self):
        """Save learned patterns and statistics"""
        try:
            data_dir = Path.home() / ".jarvis" / "learning"
            data_dir.mkdir(parents=True, exist_ok=True)

            with open(data_dir / "command_stats.json", "w") as f:
                json.dump(dict(self.command_stats), f)

            with open(data_dir / "success_patterns.json", "w") as f:
                json.dump(dict(self.success_patterns), f)

        except Exception as e:
            logger.debug(f"Could not save learned data: {e}")

    def _initialize_handlers(self):
        """Initialize command handlers lazily"""
        # We'll import handlers only when needed to avoid circular imports
        self.handler_modules = {
            CommandType.VISION: "api.vision_command_handler",
            CommandType.SYSTEM: "system_control.macos_controller",
            CommandType.WEATHER: "system_control.weather_system_config",
            CommandType.AUTONOMY: "api.autonomy_handler",
            CommandType.VOICE_UNLOCK: "api.voice_unlock_handler",
            CommandType.QUERY: "api.query_handler",  # Add basic query handler
        }

    async def process_command(
        self, command_text: str, websocket=None
    ) -> Dict[str, Any]:
        """Process any command through unified pipeline with FULL context awareness"""
        logger.info(f"[UNIFIED] Processing with context awareness: '{command_text}'")

        # Track command frequency
        self.command_stats[command_text.lower()] += 1

        # NEW: Get context-aware handler for ALL commands
        from context_intelligence.handlers.context_aware_handler import (
            get_context_aware_handler,
        )

        context_handler = get_context_aware_handler()

        # Step 1: Classify query complexity (if available)
        classified_query = None
        if self.query_complexity_manager:
            try:
                context = {"recent_commands": list(self.context.conversation_history)[-5:]}
                classified_query = await self.query_complexity_manager.process_query(
                    command_text, context=context
                )
                logger.info(
                    f"[UNIFIED] Query complexity: {classified_query.complexity.level.name} "
                    f"(type={classified_query.query_type}, intent={classified_query.intent}, "
                    f"latency={classified_query.complexity.estimated_latency[0]:.1f}-"
                    f"{classified_query.complexity.estimated_latency[1]:.1f}s, "
                    f"api_calls={classified_query.complexity.estimated_api_calls[0]}-"
                    f"{classified_query.complexity.estimated_api_calls[1]})"
                )
            except Exception as e:
                logger.warning(f"[UNIFIED] Query complexity classification failed: {e}")

        # Step 2: Classify command intent
        command_type, confidence = await self._classify_command(command_text)
        logger.info(
            f"[UNIFIED] Classified as {command_type.value} (confidence: {confidence})"
        )

        # Step 3: Check system context FIRST (screen lock, active apps, etc.)
        system_context = await self._get_full_system_context()
        logger.info(
            f"[UNIFIED] System context: screen_locked={system_context.get('screen_locked')}, active_apps={len(system_context.get('active_apps', []))}"
        )

        # Step 4: Route to Medium Complexity Handler if appropriate
        if (classified_query and
            classified_query.complexity.level.name == "MODERATE" and
            self.medium_complexity_handler and
            classified_query.entities.get("spaces")):

            try:
                from context_intelligence.handlers import MediumQueryType

                # Determine medium query type
                if classified_query.query_type == "comparison":
                    medium_type = MediumQueryType.COMPARISON
                elif classified_query.intent == "find":
                    medium_type = MediumQueryType.CROSS_SPACE_SEARCH
                else:
                    medium_type = MediumQueryType.MULTI_SPACE

                logger.info(f"[UNIFIED] Routing to MediumComplexityHandler ({medium_type.value})")

                # Execute medium complexity query
                result = await self.medium_complexity_handler.process_query(
                    query=command_text,
                    space_ids=classified_query.entities["spaces"],
                    query_type=medium_type,
                    context={"system_context": system_context}
                )

                # Return formatted result
                return {
                    "success": result.success,
                    "response": result.synthesis,
                    "command_type": "medium_complexity_query",
                    "query_complexity": {
                        "level": "MODERATE",
                        "query_type": classified_query.query_type,
                        "spaces_processed": result.spaces_processed,
                        "execution_time": result.execution_time,
                        "api_calls": result.total_api_calls
                    },
                    "captures": [
                        {
                            "space_id": c.space_id,
                            "success": c.success,
                            "text_length": len(c.ocr_text) if c.ocr_text else 0,
                            "confidence": c.ocr_confidence,
                            "method": f"{c.capture_method} + {c.ocr_method}"
                        }
                        for c in result.captures
                    ]
                }

            except Exception as e:
                logger.error(f"[UNIFIED] Medium complexity handler failed: {e}")
                # Fall through to regular processing

        # Step 4.5: Route to Complex Complexity Handler if appropriate
        if (classified_query and
            classified_query.complexity.level.name == "COMPLEX" and
            self.complex_complexity_handler):

            try:
                from context_intelligence.handlers import ComplexQueryType

                # Determine complex query type
                if any(word in command_text.lower() for word in ["changed", "change", "different", "history"]):
                    complex_type = ComplexQueryType.TEMPORAL
                elif any(word in command_text.lower() for word in ["find", "search", "all", "across", "every"]):
                    complex_type = ComplexQueryType.CROSS_SPACE
                elif any(word in command_text.lower() for word in ["progress", "predict", "will", "going"]):
                    complex_type = ComplexQueryType.PREDICTIVE
                else:
                    complex_type = ComplexQueryType.ANALYTICAL

                logger.info(f"[UNIFIED] Routing to ComplexComplexityHandler ({complex_type.value})")

                # Determine space IDs (all spaces if not specified)
                space_ids = classified_query.entities.get("spaces") if classified_query.entities else None

                # Determine time range for temporal queries
                time_range = None
                if complex_type == ComplexQueryType.TEMPORAL:
                    # Extract time range from query
                    import re
                    minutes_match = re.search(r'(\d+)\s*minute', command_text.lower())
                    hours_match = re.search(r'(\d+)\s*hour', command_text.lower())
                    if minutes_match:
                        time_range = {"minutes": int(minutes_match.group(1))}
                    elif hours_match:
                        time_range = {"hours": int(hours_match.group(1))}
                    else:
                        time_range = {"minutes": 5}  # Default to 5 minutes

                # Execute complex query
                result = await self.complex_complexity_handler.process_query(
                    query=command_text,
                    query_type=complex_type,
                    space_ids=space_ids,
                    time_range=time_range,
                    context={"system_context": system_context}
                )

                # Return formatted result
                response_parts = [result.synthesis]

                # Add temporal analysis if available
                if result.temporal_analysis:
                    ta = result.temporal_analysis
                    response_parts.append(f"\n\n**Temporal Analysis:**")
                    response_parts.append(f"- Changes detected: {ta.changes_detected}")
                    response_parts.append(f"- Changed spaces: {', '.join(map(str, ta.changed_spaces)) if ta.changed_spaces else 'none'}")

                # Add cross-space analysis if available
                if result.cross_space_analysis:
                    csa = result.cross_space_analysis
                    response_parts.append(f"\n\n**Cross-Space Analysis:**")
                    response_parts.append(f"- Spaces scanned: {csa.total_spaces_scanned}")
                    response_parts.append(f"- Matches found: {csa.matches_found}")

                # Add predictive analysis if available
                if result.predictive_analysis:
                    pa = result.predictive_analysis
                    response_parts.append(f"\n\n**Confidence:** {pa.confidence:.1%}")

                return {
                    "success": result.success,
                    "response": "\n".join(response_parts),
                    "command_type": "complex_complexity_query",
                    "query_complexity": {
                        "level": "COMPLEX",
                        "query_type": complex_type.value,
                        "spaces_processed": result.spaces_processed,
                        "execution_time": result.execution_time,
                        "api_calls": result.api_calls
                    },
                    "snapshots": [
                        {
                            "space_id": s.space_id,
                            "success": s.ocr_text is not None and not s.error,
                            "text_length": len(s.ocr_text) if s.ocr_text else 0,
                            "confidence": s.ocr_confidence,
                            "error": s.error
                        }
                        for s in result.snapshots
                    ]
                }

            except Exception as e:
                logger.error(f"[UNIFIED] Complex complexity handler failed: {e}")
                # Fall through to regular processing

        # Step 5: Resolve references with context (use resolved query if available)
        resolved_text = classified_query.resolved_query if classified_query else command_text
        reference, ref_confidence = self.context.resolve_reference(resolved_text)
        if reference and ref_confidence > 0.5:
            # Replace reference with resolved entity
            for word in ["it", "that", "this"]:
                if word in resolved_text.lower():
                    resolved_text = resolved_text.lower().replace(word, reference)
                    logger.info(f"[UNIFIED] Resolved '{word}' to '{reference}'")
                    break

        # Step 6: Define command execution callback
        async def execute_with_context(cmd: str, context: Dict[str, Any] = None):
            """Execute command with full context awareness"""
            if command_type == CommandType.COMPOUND:
                return await self._handle_compound_command(cmd, context=context)
            else:
                return await self._execute_command(
                    command_type, cmd, websocket, context=context
                )

        # Step 7: Process through context-aware handler
        logger.info(f"[UNIFIED] Processing through context-aware handler...")
        result = await context_handler.handle_command_with_context(
            resolved_text, execute_callback=execute_with_context
        )

        # Step 8: Extract actual result from context handler response
        if result.get("result"):
            # Use the nested result from context handler
            actual_result = result["result"]
        else:
            # Fallback to the full result
            actual_result = result

        # Step 9: Learn from the result
        if actual_result.get("success", False):
            self.pattern_learner.learn_pattern(command_text, command_type.value, True)
            self.success_patterns[command_type.value].append(
                {
                    "command": command_text,
                    "timestamp": datetime.now().isoformat(),
                    "context": system_context,  # Store context for learning
                }
            )
            # Keep only recent patterns
            if len(self.success_patterns[command_type.value]) > 100:
                self.success_patterns[command_type.value] = self.success_patterns[
                    command_type.value
                ][-100:]

        # Step 10: Update context with result
        self.context.update_from_command(command_type, actual_result)
        self.context.system_state = system_context  # Update system state

        # Save learned data periodically (every 10 commands)
        if sum(self.command_stats.values()) % 10 == 0:
            self._save_learned_data()

        # Return the formatted result with complexity information
        result_dict = {
            "success": actual_result.get("success", False),
            "response": result.get("summary", actual_result.get("response", "")),
            "command_type": command_type.value,
            "context_aware": True,
            "system_context": system_context,
            **actual_result,
        }

        # Add query complexity information if available
        if classified_query:
            result_dict["query_complexity"] = {
                "level": classified_query.complexity.level.name,
                "query_type": classified_query.query_type,
                "intent": classified_query.intent,
                "estimated_latency": classified_query.complexity.estimated_latency,
                "estimated_api_calls": classified_query.complexity.estimated_api_calls,
                "spaces_involved": classified_query.complexity.spaces_involved,
                "confidence": classified_query.complexity.confidence,
                "resolved_query": classified_query.resolved_query,
            }

        return result_dict

    async def _classify_command(self, command_text: str) -> Tuple[CommandType, float]:
        """Dynamically classify command using learned patterns"""
        command_lower = command_text.lower().strip()
        words = command_lower.split()

        # Manual screen lock/unlock detection (HIGHEST PRIORITY - check first!)
        # Check for exact matches first
        if command_lower in [
            "unlock my screen",
            "unlock screen",
            "unlock the screen",
            "lock my screen",
            "lock screen",
            "lock the screen",
        ]:
            logger.info(
                f"[CLASSIFY] Manual screen lock/unlock command detected: '{command_lower}'"
            )
            return (
                CommandType.VOICE_UNLOCK,
                0.99,
            )  # Route to voice unlock handler with high confidence

        # Voice unlock detection FIRST (highest priority to catch "enable voice unlock")
        voice_patterns = self._detect_voice_unlock_patterns(command_lower)
        if voice_patterns > 0:
            logger.info(
                f"Voice unlock command detected: '{command_lower}' with patterns={voice_patterns}"
            )
            return CommandType.VOICE_UNLOCK, 0.85 + (voice_patterns * 0.05)

        # Display/Screen mirroring detection (HIGH PRIORITY - before vision to avoid confusion)
        display_score = self._calculate_display_score(words, command_lower)
        if display_score > 0.7:
            logger.info(
                f"Display command detected: '{command_lower}' with score={display_score}"
            )
            return CommandType.DISPLAY, display_score

        # Check for implicit compound commands (app + action without connector)
        # Example: "open safari search for cats"
        if len(words) >= 3:
            # Look for patterns like "verb app verb"
            potential_app_indices = []
            for i, word in enumerate(words):
                if self.pattern_learner.is_learned_app(word):
                    potential_app_indices.append(i)

            # Check if there are verbs before and after an app name
            for app_idx in potential_app_indices:
                if app_idx > 0 and app_idx < len(words) - 1:
                    # Check if word before app is a verb
                    before_verb = words[app_idx - 1] in self.pattern_learner.app_verbs
                    # Check if there's a verb or action after the app
                    after_has_action = any(
                        word
                        in self.pattern_learner.app_verbs
                        | {"search", "navigate", "go", "type"}
                        for word in words[app_idx + 1 :]
                    )

                    if before_verb and after_has_action:
                        # This is an implicit compound command
                        logger.info(
                            f"[CLASSIFY] Detected implicit compound: verb-app-action pattern"
                        )
                        return CommandType.COMPOUND, 0.9

        # Dynamic compound detection - learn from connectors
        compound_indicators = {" and ", " then ", ", and ", ", then ", " && ", " ; "}
        for indicator in compound_indicators:
            if indicator in command_lower:
                # Check if this is truly compound by analyzing both sides
                parts = command_lower.split(indicator)
                if len(parts) >= 2 and all(len(p.strip()) > 0 for p in parts):
                    # Both sides have content - likely compound
                    # But exclude if it's part of a single concept
                    if not self._is_single_concept(command_lower, indicator):
                        return CommandType.COMPOUND, 0.95

        # Dynamic system command detection using learned patterns
        first_word = words[0] if words else ""

        # Check if first word is a learned verb
        if (
            first_word in self.pattern_learner.app_verbs
            or first_word in self.pattern_learner.system_verbs
        ):
            # Check if any word is a learned app
            for word in words[1:]:
                if self.pattern_learner.is_learned_app(word):
                    return CommandType.SYSTEM, 0.9

            # Check for system settings patterns
            system_indicators = self._detect_system_indicators(words)
            if system_indicators > 0:
                return CommandType.SYSTEM, 0.85 + (system_indicators * 0.05)

        # Dynamic app detection - if any learned app is mentioned
        mentioned_apps = [
            word for word in words if self.pattern_learner.is_learned_app(word)
        ]
        if mentioned_apps:
            # Check for action context
            has_action = any(word in self.pattern_learner.app_verbs for word in words)
            if has_action:
                return CommandType.SYSTEM, 0.9

        # Document creation detection (high priority - before vision)
        # Use root words that will match variations (write/writing/writes, create/creating, etc.)
        document_keywords = [
            "writ",
            "creat",
            "draft",
            "compos",
            "generat",
        ]  # Root forms
        document_types = [
            "essay",
            "report",
            "paper",
            "article",
            "document",
            "blog",
            "letter",
            "story",
        ]

        # Check if any word starts with a document keyword (handles write/writing/writes, etc.)
        has_document_keyword = any(
            any(word.startswith(kw) for word in words) for kw in document_keywords
        )
        has_document_type = any(dtype in words for dtype in document_types)

        if has_document_keyword and has_document_type:
            logger.info(
                f"[CLASSIFY] Document creation command detected: '{command_text}'"
            )
            return CommandType.DOCUMENT, 0.95

        # Vision detection through semantic analysis (CHECK BEFORE QUERY!)
        # This must come before query detection to catch vision questions
        vision_score = self._calculate_vision_score(words, command_lower)
        if vision_score > 0.7:
            return CommandType.VISION, vision_score

        # Weather detection - simple but effective
        if "weather" in words:
            return CommandType.WEATHER, 0.95

        # Autonomy detection
        autonomy_score = self._calculate_autonomy_score(words)
        if autonomy_score > 0.7:
            return CommandType.AUTONOMY, autonomy_score

        # Meta command detection
        meta_indicators = {"cancel", "stop", "undo", "never", "not", "wait", "hold"}
        meta_count = sum(1 for word in words if word in meta_indicators)
        if meta_count > 0 and len(words) < 5:
            return CommandType.META, 0.8 + (meta_count * 0.05)

        # Wake word detection
        if command_lower.strip() in {
            "hey",
            "hi",
            "hello",
            "jarvis",
            "hey jarvis",
            "activate",
            "wake",
            "wake up",
        }:
            return CommandType.META, 0.9

        # Query detection through linguistic analysis
        # NOTE: This comes AFTER vision detection to allow vision questions
        is_question = self._is_question_pattern(words)
        if is_question:
            # Check again if this might be a vision question
            if vision_score > 0.5:  # Lower threshold for questions
                return CommandType.VISION, vision_score
            return CommandType.QUERY, 0.8

        # URL/Web detection
        if self._contains_url_pattern(command_lower):
            return CommandType.SYSTEM, 0.85

        # Navigation patterns
        nav_patterns = {"go", "navigate", "browse", "visit", "open", "search"}
        if any(word in words for word in nav_patterns) and len(words) > 1:
            return CommandType.SYSTEM, 0.75

        # Short commands default to system if they contain verbs
        if len(words) <= 3 and any(
            word in self.pattern_learner.app_verbs | self.pattern_learner.system_verbs
            for word in words
        ):
            return CommandType.SYSTEM, 0.7

        # Default to query with lower confidence
        return CommandType.QUERY, 0.5

    def _is_single_concept(self, text: str, connector: str) -> bool:
        """Check if connector is part of a single concept rather than joining commands"""
        # Common phrases that shouldn't be split
        single_concepts = {
            "and press enter",
            "and enter",
            "and return",
            "black and white",
            "up and down",
            "back and forth",
            "pros and cons",
            "dos and don'ts",
        }

        for concept in single_concepts:
            if concept in text:
                return True

        # Check if it's part of a search query or typed text
        before_connector = text.split(connector)[0]
        if any(
            pattern in before_connector
            for pattern in ["search for", "type", "write", "enter"]
        ):
            return True

        return False

    def _detect_system_indicators(self, words: List[str]) -> int:
        """Count system-related indicators in words"""
        indicators = 0

        # System settings
        settings_words = {
            "volume",
            "brightness",
            "wifi",
            "bluetooth",
            "display",
            "sound",
            "network",
        }
        indicators += sum(1 for word in words if word in settings_words)

        # System actions
        action_words = {"screenshot", "restart", "shutdown", "sleep", "lock", "unlock"}
        indicators += sum(1 for word in words if word in action_words)

        # File operations
        file_words = {
            "file",
            "folder",
            "directory",
            "document",
            "save",
            "open",
            "create",
        }
        indicators += sum(1 for word in words if word in file_words)

        return indicators

    def _calculate_display_score(self, words: List[str], command_lower: str) -> float:
        """
        Calculate likelihood of display/screen mirroring command

        Detects commands like:
        - "screen mirror my Mac to the Living Room TV"
        - "connect to Living Room TV"
        - "extend display to Sony TV"
        - "airplay to Living Room TV"
        - "stop living room tv"
        - "disconnect from living room tv"
        - "stop screen mirroring"
        """
        score = 0.0

        # Clean words by removing punctuation
        import re
        clean_words = [re.sub(r'[^\w\s]', '', word) for word in words]

        # Primary display/mirroring keywords (STRONG indicators)
        primary_keywords = {
            "mirror": 0.8,
            "airplay": 0.9,
            "extend": 0.7,
        }

        for keyword, weight in primary_keywords.items():
            if keyword in clean_words:
                score += weight

        # Secondary display keywords (combined with display action)
        secondary_keywords = {"display", "screen", "tv", "television"}
        has_secondary = any(kw in clean_words for kw in secondary_keywords)

        # Display action verbs (both connection and disconnection)
        action_verbs = {"connect", "cast", "project", "stream", "share", "stop", "disconnect", "turn", "disable"}
        has_action = any(verb in clean_words for verb in action_verbs)

        # Disconnection indicators (boost score for disconnect commands)
        disconnect_indicators = {"stop", "disconnect", "turn", "disable", "off"}
        has_disconnect = any(indicator in clean_words for indicator in disconnect_indicators)
        if has_disconnect and has_secondary:
            score += 0.7

        # Boost if we have action verb + display keyword
        if has_action and has_secondary:
            score += 0.6

        # Boost for prepositions indicating target ("to", "on")
        if ("to" in clean_words or "on" in clean_words) and has_secondary:
            score += 0.2

        # Check for TV/display names (Living Room, Sony, etc.)
        # If "room" or "tv" or brand names are mentioned with action
        tv_indicators = {"room", "tv", "television", "sony", "lg", "samsung"}
        has_tv_indicator = any(indicator in clean_words for indicator in tv_indicators)

        if has_tv_indicator and (has_action or score > 0):
            score += 0.3

        # Specific display name patterns (HIGH confidence even without action verb)
        # These patterns strongly indicate user wants to connect to a display
        display_name_patterns = [
            r"living\s*room\s*tv",      # "living room tv"
            r"bedroom\s*tv",             # "bedroom tv"
            r"kitchen\s*tv",             # "kitchen tv"
            r"office\s*tv",              # "office tv"
            r"\w+\s*room\s*tv",         # "any room tv"
            r"(sony|lg|samsung)\s*tv",   # "sony tv", "lg tv", etc.
        ]

        for pattern in display_name_patterns:
            if re.search(pattern, command_lower):
                # Known display name mentioned - very likely a connection request
                score = max(score, 0.85)
                break

        # Specific phrase matching (highest confidence)
        if "screen mirror" in command_lower or "screen mirroring" in command_lower:
            score = max(score, 0.95)

        if "airplay" in command_lower and "to" in command_lower:
            score = max(score, 0.95)

        # Disconnection phrases (high confidence)
        disconnect_phrases = [
            "stop screen mirror",
            "stop mirroring",
            "disconnect display",
            "turn off screen mirror",
            "stop airplay"
        ]
        for phrase in disconnect_phrases:
            if phrase in command_lower:
                score = max(score, 0.95)
                break

        # Mode change phrases (high confidence)
        mode_change_phrases = [
            "change to extended",
            "change to entire",
            "change to window",
            "switch to extended",
            "switch to entire",
            "switch to window",
            "set to extended",
            "set to entire",
            "extended display",
            "entire screen",
            "window or app"
        ]
        for phrase in mode_change_phrases:
            if phrase in command_lower:
                score = max(score, 0.95)
                break

        return min(score, 1.0)  # Cap at 1.0

    def _calculate_vision_score(self, words: List[str], command_lower: str) -> float:
        """Calculate likelihood of vision command"""
        score = 0.0

        # Clean words by removing punctuation for better matching
        import re
        clean_words = [re.sub(r'[^\w\s]', '', word) for word in words]

        # EXCLUDE lock/unlock commands - they're system commands, not vision
        if "lock" in clean_words or "unlock" in clean_words:
            return 0.0

        # Vision verbs
        vision_verbs = {
            "see",
            "look",
            "watch",
            "monitor",
            "analyze",
            "describe",
            "show",
            "read",
            "check",
            "examine",
        }
        verb_count = sum(1 for word in clean_words if word in vision_verbs)
        score += verb_count * 0.2
        
        # "monitor" or "analyze" with "screen" is definitely vision
        if ("monitor" in clean_words or "analyze" in clean_words) and "screen" in clean_words:
            score += 0.5  # Extra boost for monitor/analyze screen

        # Vision nouns (but be careful with 'screen' - it could be system related)
        vision_nouns = {
            "display",
            "window",
            "image",
            "visual",
            "picture",
            "desktop",
            "space",
            "workspace",
            "screen",
        }
        score += sum(0.15 for word in clean_words if word in vision_nouns)

        # Multi-space indicators (very strong vision signal)
        multi_space_indicators = {
            "desktop",
            "space",
            "spaces",  # Added plural
            "workspace",
            "workspaces",  # Added plural
            "across",
            "multiple",
            "different",
            "other",
            "all",
        }
        multi_space_count = sum(1 for word in clean_words if word in multi_space_indicators)
        if multi_space_count > 0:
            score += 0.4 * multi_space_count  # Strong boost for multi-space queries
            
        # Extra boost for "desktop spaces" or "workspace" combinations
        if ("desktop" in clean_words and ("space" in clean_words or "spaces" in clean_words)) or \
           ("workspace" in clean_words or "workspaces" in clean_words):
            score += 0.3  # Extra boost for these specific combinations

        # 'screen' only counts as vision if paired with vision verbs or multi-space indicators
        if "screen" in clean_words:
            if any(word in vision_verbs for word in clean_words) or multi_space_count > 0:
                score += 0.15
            # Questions about screen are very likely vision
            elif clean_words[0] in {"what", "whats", "show", "display"}:
                score += 0.6  # Strong boost for screen questions

        # Questioning about visual or workspace
        if clean_words and clean_words[0] in {"what", "whats"}:
            visual_indicators = {
                "screen",
                "see",
                "display",
                "desktop",
                "space",
                "workspace",
                "happening",
                "going",
                "doing",
            }
            if any(word in clean_words for word in visual_indicators):
                score += 0.3

        # Phrases that strongly indicate workspace/multi-space vision queries
        workspace_phrases = [
            "desktop space",
            "across my desktop",
            "multiple desktop",
            "different space",
            "what am i working",
            "what is happening",
            "what's happening",
            "what is going on",
            "happening across",
            "across my desktop spaces",
        ]
        for phrase in workspace_phrases:
            if phrase in command_lower:
                score += 0.5

        return min(score, 0.95)

    def _detect_voice_unlock_patterns(self, text: str) -> int:
        """Detect voice unlock related patterns"""
        patterns = 0

        voice_words = {"voice", "vocal", "speech", "voiceprint"}
        unlock_words = {
            "unlock",
            "lock",
            "authenticate",
            "verify",
            "enroll",
            "enrollment",
        }

        # Check for voice + unlock combinations
        has_voice = any(word in text for word in voice_words)
        has_unlock = any(word in text for word in unlock_words)

        if has_voice and has_unlock:
            patterns += 2
        elif has_voice or has_unlock:
            patterns += 1

        # Log for debugging
        if has_voice or has_unlock:
            logger.debug(
                f"Voice unlock pattern detection: text='{text}', has_voice={has_voice}, has_unlock={has_unlock}, patterns={patterns}"
            )

        # Direct phrases - these are definitely voice unlock commands
        voice_unlock_phrases = [
            "voice unlock",
            "unlock with voice",
            "enable voice unlock",
            "disable voice unlock",
            "enroll my voice",
            "enroll voice",
            "voice enrollment",
        ]

        if any(phrase in text for phrase in voice_unlock_phrases):
            patterns += 3  # Strong match

        return patterns

    def _calculate_autonomy_score(self, words: List[str]) -> float:
        """Calculate autonomy command likelihood"""
        score = 0.0

        autonomy_words = {"autonomy", "autonomous", "auto", "automatic", "self"}
        control_words = {"control", "take", "activate", "enable", "mode"}

        score += sum(0.3 for word in words if word in autonomy_words)
        score += sum(0.2 for word in words if word in control_words)

        # Boost for specific phrases
        text = " ".join(words)
        if "take over" in text or "full control" in text:
            score += 0.4

        return min(score, 0.95)

    def _is_question_pattern(self, words: List[str]) -> bool:
        """Detect if command is a question"""
        if not words:
            return False

        # Question starters
        question_starts = {
            "what",
            "who",
            "where",
            "when",
            "why",
            "how",
            "is",
            "are",
            "can",
            "could",
            "would",
            "should",
            "will",
            "do",
            "does",
        }

        # Check first word
        if words[0] in question_starts:
            return True

        # Check for question marks (though unlikely in voice)
        if any("?" in word for word in words):
            return True

        return False

    def _contains_url_pattern(self, text: str) -> bool:
        """Check if text contains URL patterns"""
        # URL indicators
        url_patterns = [
            r"https?://",
            r"www\.",
            r"\.(com|org|net|edu|gov|io|co|uk)",
            r"://",
        ]

        for pattern in url_patterns:
            if re.search(pattern, text):
                return True

        # Common websites without full URLs
        websites = {
            "google",
            "facebook",
            "twitter",
            "youtube",
            "github",
            "amazon",
            "reddit",
        }
        words = text.split()

        # Check if website is mentioned with navigation verb
        nav_verbs = {"go", "visit", "open", "navigate", "browse"}
        for i, word in enumerate(words):
            if word in websites and i > 0 and words[i - 1] in nav_verbs:
                return True

        return False

    async def _resolve_vision_query(self, query: str) -> Dict[str, Any]:
        """
        Two-stage resolution for comprehensive query understanding

        Stage 1 (Implicit Resolver): Entity & Intent Resolution
        - "What does it say?" -> "it" = error in Terminal
        - Intent: DESCRIBE
        - Entity type: error
        - May include space_id from visual attention

        Stage 2 (Contextual Resolver): Space & Monitor Resolution
        - If Stage 1 didn't find space, resolve it now
        - "What's happening?" -> Space 2 (active space)
        - "Compare them" -> Spaces [3, 5] (last queried)

        Returns:
            Dict with comprehensive resolution including:
            - intent: QueryIntent (from implicit resolver)
            - entity: Resolved entity (error, file, etc.)
            - spaces: List[int] (resolved space IDs)
            - confidence: Combined confidence score
        """
        resolution = {
            "original_query": query,
            "resolved": False,
            "query": query,
            "intent": None,
            "entity_resolution": None,
            "space_resolution": None,
            "spaces": None,
            "confidence": 0.0
        }

        # ============================================================
        # STAGE 1: Implicit Reference Resolution (Entity & Intent)
        # ============================================================
        if self.implicit_resolver:
            try:
                logger.debug(f"[UNIFIED] Stage 1: Implicit resolution for '{query}'")
                implicit_result = await self.implicit_resolver.resolve_query(query)

                # Extract intent
                resolution["intent"] = implicit_result.get("intent")

                # Extract entity referent
                referent = implicit_result.get("referent", {})
                if referent and referent.get("source") != "none":
                    resolution["entity_resolution"] = {
                        "source": referent.get("source"),
                        "type": referent.get("type"),
                        "entity": referent.get("entity"),
                        "confidence": referent.get("relevance", 0.0)
                    }

                    logger.info(
                        f"[UNIFIED] Stage 1 ✅: Intent={resolution['intent']}, "
                        f"Entity={referent.get('type')} from {referent.get('source')}"
                    )

                    # If implicit resolver found a specific space, use it (high confidence!)
                    if referent.get("space_id"):
                        resolution["spaces"] = [referent["space_id"]]
                        resolution["space_resolution"] = {
                            "strategy": "implicit_reference",
                            "confidence": 1.0,
                            "source": "visual_attention"
                        }
                        resolution["resolved"] = True
                        resolution["confidence"] = implicit_result.get("confidence", 0.9)

                        # Enhance query with entity and space info
                        entity_desc = referent.get("entity", "")[:50]
                        resolution["query"] = f"{query} [entity: {entity_desc}, space: {referent['space_id']}]"

                        logger.info(
                            f"[UNIFIED] Stage 1 complete: Space {referent['space_id']} from implicit resolver"
                        )
                        return resolution

            except Exception as e:
                logger.warning(f"[UNIFIED] Stage 1 error: {e}", exc_info=True)

        # ============================================================
        # STAGE 2: Contextual Space Resolution (if needed)
        # ============================================================
        if self.contextual_resolver:
            try:
                logger.debug(f"[UNIFIED] Stage 2: Contextual space resolution for '{query}'")
                space_result = await self.contextual_resolver.resolve_query(query)

                if space_result.requires_clarification:
                    # Query is too ambiguous
                    resolution["clarification_needed"] = True
                    resolution["clarification_message"] = space_result.clarification_message
                    logger.info(f"[UNIFIED] Stage 2: Clarification needed")
                    return resolution

                if space_result.success and space_result.resolved_spaces:
                    # Successfully resolved spaces
                    spaces = space_result.resolved_spaces
                    strategy = space_result.strategy_used.value

                    resolution["spaces"] = spaces
                    resolution["space_resolution"] = {
                        "strategy": strategy,
                        "confidence": space_result.confidence,
                        "monitors": space_result.resolved_monitors
                    }
                    resolution["resolved"] = True

                    # Calculate combined confidence
                    if resolution["entity_resolution"]:
                        # Both stages succeeded
                        entity_conf = resolution["entity_resolution"]["confidence"]
                        space_conf = space_result.confidence
                        resolution["confidence"] = (entity_conf + space_conf) / 2
                    else:
                        # Only space resolution
                        resolution["confidence"] = space_result.confidence

                    # Enhance query with space info (and entity if available)
                    enhanced_query = query
                    if resolution["entity_resolution"]:
                        entity = resolution["entity_resolution"]["entity"][:50]
                        enhanced_query = f"{query} [entity: {entity}]"

                    if len(spaces) == 1:
                        enhanced_query = f"{enhanced_query} [space {spaces[0]}]"
                    elif len(spaces) > 1:
                        enhanced_query = f"{enhanced_query} [spaces {', '.join(map(str, spaces))}]"

                    resolution["query"] = enhanced_query

                    logger.info(
                        f"[UNIFIED] Stage 2 ✅: Resolved to spaces {spaces} "
                        f"using {strategy} (confidence: {space_result.confidence})"
                    )

            except Exception as e:
                logger.warning(f"[UNIFIED] Stage 2 error: {e}", exc_info=True)

        # ============================================================
        # FALLBACK: No resolution
        # ============================================================
        if not resolution["resolved"]:
            logger.debug(f"[UNIFIED] No resolution available for '{query}' - using original query")
            resolution["query"] = query
            resolution["confidence"] = 0.0

        return resolution

    def record_visual_attention(self, space_id: int, app_name: str, ocr_text: str,
                               content_type: str = "unknown", significance: str = "normal"):
        """
        Record visual attention for implicit reference resolution

        This creates a feedback loop where vision analysis feeds into the
        implicit resolver's visual attention tracker.

        Args:
            space_id: The space where content was seen
            app_name: The application displaying the content
            ocr_text: OCR text from the screen
            content_type: Type of content (error, code, documentation, terminal_output)
            significance: Importance level (critical, high, normal, low)
        """
        if not self.implicit_resolver:
            return

        try:
            self.implicit_resolver.record_visual_attention(
                space_id=space_id,
                app_name=app_name,
                ocr_text=ocr_text,
                content_type=content_type,
                significance=significance
            )
            logger.debug(
                f"[UNIFIED] Recorded visual attention: {content_type} in {app_name} "
                f"(Space {space_id}, significance={significance})"
            )
        except Exception as e:
            logger.warning(f"[UNIFIED] Failed to record visual attention: {e}")
    
    def _is_multi_space_query(self, query: str) -> bool: # check if the query is about multiple spaces
        """
        Detect if a query is asking about multiple spaces.

        Examples:
        - "Compare space 3 and space 5"
        - "Which space has the error?"
        - "Find the terminal across all spaces"
        - "What's different between space 1 and space 2?"
        """
        query_lower = query.lower() # convert the query to lowercase    

        # Keywords that indicate multi-space queries
        multi_space_keywords = [
            "compare",
            "difference",
            "different",
            "find",
            "which space",
            "across",
            "all spaces",
            "search",
            "locate"
        ]

        # Check for keywords
        if any(keyword in query_lower for keyword in multi_space_keywords): # if any of the keywords are in the query, it's a multi-space query
            return True # return True if it's a multi-space query

        # Check for multiple space mentions
        import re
        space_matches = re.findall(r'space\s+\d+', query_lower) # find all space mentions in the query
        if len(space_matches) >= 2: # if there are at least two space mentions, it's a multi-space query
            return True # return True if it's a multi-space query

        return False # return False if it's not a multi-space query

    # Function to handle multi-space queries
    async def _handle_multi_space_query(self, query: str) -> Dict[str, Any]:
        """
        Handle multi-space queries using the MultiSpaceQueryHandler.

        Args:
            query: User's multi-space query

        Returns:
            Dict with comprehensive multi-space analysis
        """
        if not self.multi_space_handler: # if the multi-space handler is not available
            # Fallback: treat as regular vision query
            logger.warning("[UNIFIED] Multi-space query detected but handler not available")
            return {
                "success": False, # indicate failure
                "response": "Multi-space analysis not available. Please specify a single space.", # add error message
                "multi_space": False # indicate that it's not a multi-space query
            }

        try:
            logger.info(f"[UNIFIED] Handling multi-space query: '{query}'")

            # Use the multi-space handler
            result = await self.multi_space_handler.handle_query(query) # handle the multi-space query

            # Build response with the results of the multi-space query
            response = {
                "success": True, # indicate success
                "response": result.synthesis, # add the synthesis to the response
                "multi_space": True, # indicate that it's a multi-space query
                "query_type": result.query_type.value, # add the query type to the response
                "spaces_analyzed": result.spaces_analyzed, # add the spaces analyzed to the response
                "results": [
                    {
                        "space_id": r.space_id, # add the space id to the response
                        "success": r.success, # add the success to the response
                        "app": r.app_name, # add the app name to the response
                        "content_type": r.content_type, # add the content type to the response
                        "summary": r.content_summary, # add the content summary to the response
                        "errors": r.errors, # add the errors to the response
                        "significance": r.significance # add the significance to the response
                    }
                    for r in result.results # loop through the results
                ],
                "confidence": result.confidence, # add the confidence to the response
                "analysis_time": result.total_time # add the analysis time to the response
            }

            # Add comparison if available
            if result.comparison: # if there is a comparison, add it to the response
                response["comparison"] = result.comparison # add the comparison to the response

            # Add differences if available
            if result.differences: # if there is a difference, add it to the response
                response["differences"] = result.differences # add the difference to the response

            # Add search matches if available
            if result.search_matches: # if there is a search match, add it to the response
                response["search_matches"] = result.search_matches # add the search match to the response

            logger.info(
                f"[UNIFIED] Multi-space query completed: "
                f"{len(result.spaces_analyzed)} spaces analyzed in {result.total_time:.2f}s"
            )

            return response

        except Exception as e:
            logger.error(f"[UNIFIED] Multi-space query failed: {e}", exc_info=True)
            return {
                "success": False,
                "response": f"Multi-space analysis failed: {str(e)}",
                "multi_space": True,
                "error": str(e)
            }

    def _is_temporal_query(self, query: str) -> bool:
        """
        Detect if a query is temporal (time-based, change detection, error tracking).

        Examples:
        - "What changed in space 3?"
        - "Has the error been fixed?"
        - "What's new in the last 5 minutes?"
        - "When did this error first appear?"
        """
        query_lower = query.lower()

        # Keywords that indicate temporal queries
        temporal_keywords = [
            "changed", "change", "different",
            "fixed", "error", "bug", "issue",
            "new", "recently", "last",
            "when", "history", "timeline",
            "appeared", "first", "started",
            "ago", "since", "before", "after",
            "latest", "recent", "past"
        ]

        # Check for keywords
        if any(keyword in query_lower for keyword in temporal_keywords):
            return True

        # Check for time expressions
        import re
        time_patterns = [
            r'\d+\s+(minute|hour|day|second)s?\s+ago',
            r'last\s+\d+\s+(minute|hour|day|second)s?',
            r'in\s+the\s+last',
            r'(today|yesterday|recently|just now)'
        ]

        for pattern in time_patterns:
            if re.search(pattern, query_lower):
                return True

        return False

    async def _handle_temporal_query(self, query: str) -> Dict[str, Any]:
        """
        Handle temporal queries using the TemporalQueryHandler.

        Args:
            query: User's temporal query

        Returns:
            Dict with temporal analysis results
        """
        if not self.temporal_handler:
            # Fallback: treat as regular query
            logger.warning("[UNIFIED] Temporal query detected but handler not available")
            return {
                "success": False,
                "response": "Temporal analysis not available. Cannot track changes over time.",
                "temporal": False
            }

        try:
            logger.info(f"[UNIFIED] Handling temporal query: '{query}'")

            # Get current space (or from query)
            space_id = None
            import re
            space_match = re.search(r'space\s+(\d+)', query.lower())
            if space_match:
                space_id = int(space_match.group(1))

            # Use the temporal handler
            result = await self.temporal_handler.handle_query(query, space_id)

            # Build response
            response = {
                "success": True,
                "response": result.summary,
                "temporal": True,
                "query_type": result.query_type.name,
                "time_range": {
                    "start": result.time_range.start.isoformat(),
                    "end": result.time_range.end.isoformat(),
                    "duration_seconds": result.time_range.duration_seconds
                },
                "changes": [
                    {
                        "type": change.change_type.value,
                        "description": change.description,
                        "confidence": change.confidence,
                        "timestamp": change.timestamp.isoformat(),
                        "space_id": change.space_id
                    }
                    for change in result.changes
                ],
                "timeline": result.timeline,
                "screenshot_count": len(result.screenshots)
            }

            # Add metadata if available
            if result.metadata:
                response["metadata"] = result.metadata

            logger.info(
                f"[UNIFIED] Temporal query completed: "
                f"{len(result.changes)} changes detected over {result.time_range.duration_seconds:.0f}s"
            )

            return response

        except Exception as e:
            logger.error(f"[UNIFIED] Temporal query failed: {e}", exc_info=True)
            return {
                "success": False,
                "response": f"Temporal analysis failed: {str(e)}",
                "temporal": True,
                "error": str(e)
            }

    async def _get_full_system_context(self) -> Dict[str, Any]:
        """Get comprehensive system context for intelligent command processing"""
        try:
            from context_intelligence.detectors.screen_lock_detector import (
                get_screen_lock_detector,
            )

            screen_detector = get_screen_lock_detector()
            is_locked = await screen_detector.is_screen_locked()

            # Get active applications (you can expand this)
            active_apps = []
            try:
                import subprocess

                result = subprocess.run(
                    [
                        "osascript",
                        "-e",
                        'tell application "System Events" to get name of (processes where background only is false)',
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0:
                    active_apps = result.stdout.strip().split(", ")
            except:
                pass

            return {
                "screen_locked": is_locked,
                "active_apps": active_apps,
                "network_connected": True,  # You can expand this check
                "timestamp": datetime.now().isoformat(),
                "user_preferences": self.context.user_preferences,
                "conversation_history": len(self.context.conversation_history),
            }
        except Exception as e:
            logger.warning(f"Could not get full system context: {e}")
            return {
                "screen_locked": False,
                "active_apps": [],
                "network_connected": True,
                "timestamp": datetime.now().isoformat(),
            }

    async def _execute_command(
        self,
        command_type: CommandType,
        command_text: str,
        websocket=None,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Execute command using appropriate handler"""

        # Get or initialize handler
        if command_type not in self.handlers:
            handler = await self._get_handler(command_type)
            if handler:
                self.handlers[command_type] = handler

        handler = self.handlers.get(command_type)

        if not handler and command_type not in [
            CommandType.SYSTEM,
            CommandType.META,
            CommandType.DOCUMENT,
            CommandType.DISPLAY,  # Display has dedicated handler below
        ]:
            return {
                "success": False,
                "response": f"I don't have a handler for {command_type.value} commands yet.",
                "command_type": command_type.value,
            }

        # Execute with unified context
        try:
            # Different handlers have different interfaces, normalize them
            if command_type == CommandType.VISION:
                # For vision commands, check if it's a monitoring command or analysis
                if any(
                    word in command_text.lower()
                    for word in ["start", "stop", "monitor"]
                ):
                    result = await handler.handle_command(command_text)
                else:
                    # Check if this is a temporal query first (change detection, error tracking, timeline)
                    if self._is_temporal_query(command_text):
                        logger.info(f"[UNIFIED] Detected temporal query: '{command_text}'")
                        return await self._handle_temporal_query(command_text)

                    # Check if this is a multi-space query
                    if self._is_multi_space_query(command_text):
                        logger.info(f"[UNIFIED] Detected multi-space query: '{command_text}'")
                        return await self._handle_multi_space_query(command_text)

                    # It's a single-space vision query - use two-stage resolution
                    resolved_query = await self._resolve_vision_query(command_text)

                    # Check if clarification is needed
                    if resolved_query.get("clarification_needed"):
                        return {
                            "success": False,
                            "response": resolved_query.get("clarification_message"),
                            "command_type": command_type.value,
                            "clarification_needed": True
                        }

                    # Analyze the screen with the enhanced query
                    result = await handler.analyze_screen(resolved_query.get("query", command_text))

                    # Add comprehensive resolution context to result
                    if resolved_query.get("resolved"):
                        result["query_resolution"] = {
                            "original_query": command_text,
                            "intent": resolved_query.get("intent"),
                            "entity_resolution": resolved_query.get("entity_resolution"),
                            "space_resolution": resolved_query.get("space_resolution"),
                            "resolved_spaces": resolved_query.get("spaces"),
                            "confidence": resolved_query.get("confidence"),
                            "two_stage": True  # Indicates both resolvers were used
                        }

                        # Log the comprehensive resolution
                        logger.info(
                            f"[UNIFIED] Vision query resolved - "
                            f"Intent: {resolved_query.get('intent')}, "
                            f"Spaces: {resolved_query.get('spaces')}, "
                            f"Confidence: {resolved_query.get('confidence')}"
                        )

                return {
                    "success": result.get("handled", False),
                    "response": result.get("response", ""),
                    "command_type": command_type.value,
                    **result,
                }
            elif command_type == CommandType.WEATHER:
                result = await handler.get_weather(command_text)
                return {
                    "success": result.get("success", False),
                    "response": result.get(
                        "formatted_response", result.get("message", "")
                    ),
                    "command_type": command_type.value,
                    **result,
                }
            elif command_type == CommandType.SYSTEM:
                # Handle system commands (app control, system settings, etc.)
                result = await self._execute_system_command(command_text)
                return {
                    "success": result.get("success", False),
                    "response": result.get("response", ""),
                    "command_type": command_type.value,
                    **result,
                }
            elif command_type == CommandType.META:
                # Handle meta commands (wake words, cancellations)
                if command_text.lower().strip() in [
                    "activate",
                    "wake",
                    "wake up",
                    "hello",
                    "hey",
                ]:
                    # Silent acknowledgment for wake words
                    return {
                        "success": True,
                        "response": "",
                        "command_type": "meta",
                        "silent": True,
                    }
                else:
                    return {
                        "success": True,
                        "response": "Understood",
                        "command_type": "meta",
                    }
            elif command_type == CommandType.DISPLAY:
                # Handle display/screen mirroring commands
                result = await self._execute_display_command(command_text)
                return {
                    "success": result.get("success", False),
                    "response": result.get("response", ""),
                    "command_type": command_type.value,
                    **result,
                }
            elif command_type == CommandType.DOCUMENT:
                # Handle document creation commands WITH CONTEXT AWARENESS
                logger.info(
                    f"[DOCUMENT] Routing to context-aware document handler: '{command_text}'"
                )
                try:
                    from context_intelligence.handlers.context_aware_handler import (
                        get_context_aware_handler,
                    )
                    from context_intelligence.executors import (
                        get_document_writer,
                        parse_document_request,
                    )

                    # Get the context-aware handler
                    context_handler = get_context_aware_handler()

                    # Define the document creation callback
                    async def create_document_callback(
                        command: str, context: Dict[str, Any] = None
                    ):
                        logger.info(
                            f"[DOCUMENT] Creating document within context-aware flow"
                        )

                        # Parse the document request
                        doc_request = parse_document_request(command, {})

                        # Get document writer
                        writer = get_document_writer()

                        # Start document creation as a background task (non-blocking)
                        # This allows us to return immediately with feedback
                        logger.info(
                            f"[DOCUMENT] Starting background document creation task"
                        )
                        asyncio.create_task(
                            writer.create_document(
                                request=doc_request, websocket=websocket
                            )
                        )

                        # Return immediate feedback to user
                        return {
                            "success": True,
                            "task_started": True,
                            "topic": doc_request.topic,
                            "message": f"I'm creating an essay about {doc_request.topic} for you, Sir.",
                        }

                    # Use context-aware handler to check screen lock FIRST
                    logger.info(
                        f"[DOCUMENT] Checking context (including screen lock) before document creation..."
                    )
                    result = await context_handler.handle_command_with_context(
                        command_text, execute_callback=create_document_callback
                    )

                    # The context handler will handle all messaging including screen lock notifications
                    if result.get("success"):
                        return {
                            "success": True,
                            "response": result.get(
                                "summary",
                                result.get("messages", ["Document created"])[0],
                            ),
                            "command_type": command_type.value,
                            "speak": False,  # Context handler already spoke if needed
                            **result,
                        }
                    else:
                        return {
                            "success": False,
                            "response": result.get(
                                "summary",
                                result.get("messages", ["Failed to create document"])[
                                    0
                                ],
                            ),
                            "command_type": command_type.value,
                            **result,
                        }

                except Exception as e:
                    logger.error(
                        f"[DOCUMENT] Error in context-aware document creation: {e}",
                        exc_info=True,
                    )
                    return {
                        "success": False,
                        "response": f"I encountered an error creating the document: {str(e)}",
                        "command_type": command_type.value,
                        "error": str(e),
                    }
            elif command_type == CommandType.VOICE_UNLOCK:
                # Handle voice unlock commands with quick response
                command_lower = command_text.lower()

                # Check for initial enrollment request
                if (
                    "enroll" in command_lower
                    and "voice" in command_lower
                    and "start" not in command_lower
                ):
                    # Quick response for enrollment instructions
                    return {
                        "success": True,
                        "response": 'To enroll your voice, Sir, I need you to speak clearly for about 10 seconds. Say "Start voice enrollment now" when you are ready in a quiet environment.',
                        "command_type": command_type.value,
                        "type": "voice_unlock",
                        "action": "enrollment_instructions",
                        "next_command": "start voice enrollment now",
                    }
                # For actual enrollment start, let the handler process it
                elif any(
                    phrase in command_lower
                    for phrase in [
                        "start voice enrollment now",
                        "begin voice enrollment",
                        "start enrollment now",
                    ]
                ):
                    # Let the actual handler process enrollment
                    result = await handler.handle_command(command_text, websocket)
                    return {
                        "success": result.get(
                            "success", result.get("type") == "voice_unlock"
                        ),
                        "response": result.get("message", result.get("response", "")),
                        "command_type": command_type.value,
                        **result,
                    }
                else:
                    # Other voice unlock commands - use the handler
                    result = await handler.handle_command(command_text, websocket)
                    return {
                        "success": result.get(
                            "success", result.get("type") == "voice_unlock"
                        ),
                        "response": result.get("message", result.get("response", "")),
                        "command_type": command_type.value,
                        **result,
                    }
            else:
                # Generic handler interface
                return {
                    "success": True,
                    "response": f"Executing {command_type.value} command",
                    "command_type": command_type.value,
                }

        except Exception as e:
            logger.error(
                f"Error executing {command_type.value} command: {e}", exc_info=True
            )
            return {
                "success": False,
                "response": f"I encountered an error with that {command_type.value} command.",
                "command_type": command_type.value,
                "error": str(e),
            }

    async def _get_handler(self, command_type: CommandType):
        """Dynamically import and get handler for command type"""
        # System commands are handled directly in _execute_command
        if command_type == CommandType.SYSTEM:
            return True  # Return True to indicate system handler is available

        module_name = self.handler_modules.get(command_type)
        if not module_name:
            return None

        try:
            if command_type == CommandType.VISION:
                from api.vision_command_handler import vision_command_handler

                return vision_command_handler
            elif command_type == CommandType.WEATHER:
                from system_control.weather_system_config import get_weather_system

                return get_weather_system()
            elif command_type == CommandType.AUTONOMY:
                from api.autonomy_handler import get_autonomy_handler

                return get_autonomy_handler()
            elif command_type == CommandType.VOICE_UNLOCK:
                from api.voice_unlock_handler import get_voice_unlock_handler

                return get_voice_unlock_handler()
            # Add other handlers as needed

        except ImportError as e:
            logger.error(f"Failed to import handler for {command_type.value}: {e}")
            return None

    async def _handle_compound_command(
        self, command_text: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Handle commands with multiple parts and maintain context between them"""
        logger.info(
            f"[COMPOUND] Handling compound command with context: {context is not None}"
        )

        # Parse compound commands more intelligently
        parts = self._parse_compound_parts(command_text)

        results = []
        all_success = True
        responses = []

        # Track context for dependent commands
        active_app = None
        previous_result = None

        # Use provided context if available
        if context:
            logger.info(f"[COMPOUND] Using provided system context")

        # Check if all parts are similar operations that can be parallelized
        can_parallelize = self._can_parallelize_commands(parts)

        if can_parallelize:
            # Process similar operations in parallel (e.g., closing multiple apps)
            logger.info(
                f"[COMPOUND] Processing {len(parts)} similar commands in parallel"
            )

            # Create tasks for parallel execution
            tasks = []
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                # Process each part as an independent command
                async def process_part(p):
                    command_type, _ = await self._classify_command(p)
                    if command_type == CommandType.COMPOUND:
                        command_type = CommandType.SYSTEM
                    return await self._execute_command(command_type, p)

                tasks.append(process_part(part))

            # Execute all tasks in parallel
            results = await asyncio.gather(*tasks)

            # Collect responses
            for result in results:
                if result.get("success", False):
                    responses.append(result.get("response", ""))
                else:
                    all_success = False
                    responses.append(
                        f"Failed: {result.get('response', 'Unknown error')}"
                    )
        else:
            # Sequential processing for dependent commands
            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue

                # Provide user feedback for multi-step commands
                if len(parts) > 1:
                    logger.info(f"[COMPOUND] Step {i+1}/{len(parts)}: {part}")

                # Check if this is a dependent command that needs context
                enhanced_command = self._enhance_with_context(
                    part, active_app, previous_result
                )
                logger.info(
                    f"[COMPOUND] Enhanced command: '{part}' -> '{enhanced_command}' (active_app: {active_app})"
                )

                # Process individual part (not as compound to avoid recursion)
                command_type, _ = await self._classify_command(enhanced_command)
                # Force non-compound to avoid recursion
                if command_type == CommandType.COMPOUND:
                    command_type = CommandType.SYSTEM

                result = await self._execute_command(command_type, enhanced_command)
                results.append(result)

                # Update context for next command
                if result.get("success", False):
                    # Track opened apps for subsequent commands
                    if any(
                        word in part.lower() for word in ["open", "launch", "start"]
                    ):
                        # Find which app was opened dynamically
                        words = enhanced_command.lower().split()
                        for word in words:
                            if self.pattern_learner.is_learned_app(word):
                                active_app = word
                                # Even if app is already open, we want to track it for context
                                logger.info(
                                    f"[COMPOUND] Tracking active app: {active_app}"
                                )
                                break

                    # Skip "already open" messages in compound commands
                    response = result.get("response", "")
                    if "already open" not in response.lower() or len(parts) == 1:
                        responses.append(response)
                else:
                    all_success = False
                    responses.append(
                        f"Failed: {result.get('response', 'Unknown error')}"
                    )
                    # Don't continue if a step fails
                    break

                previous_result = result

                # Add small delay between commands for reliability
                if i < len(parts) - 1:
                    await asyncio.sleep(0.5)

        # Create conversational response
        if len(responses) > 1:
            # Clean up individual responses first
            cleaned_responses = []
            for i, resp in enumerate(responses):
                # Remove trailing "Sir" from all but the last response
                if resp.endswith(", Sir") and i < len(responses) - 1:
                    resp = resp[:-5]
                cleaned_responses.append(resp)

            # Combine into natural response
            if len(cleaned_responses) == 2:
                # For 2 steps: "Opening Safari and searching for dogs"
                response = f"{cleaned_responses[0]} and {cleaned_responses[1]}"
            else:
                # For 3+ steps: "Opening Safari, navigating to Google, and taking a screenshot"
                response = (
                    ", ".join(cleaned_responses[:-1]) + f" and {cleaned_responses[-1]}"
                )

            # Add "Sir" at the end if it's not already there
            if not response.endswith(", Sir"):
                response += ", Sir"
        else:
            response = responses[0] if responses else "I'll help you with that"

        return {
            "success": all_success,
            "response": response,
            "command_type": CommandType.COMPOUND.value,
            "sub_results": results,
            "steps_completed": len([r for r in results if r.get("success", False)]),
            "total_steps": len(parts),
            "context": context or {},
            "steps_taken": [f"Step {i+1}: {part}" for i, part in enumerate(parts)],
        }

    def _parse_compound_parts(self, command_text: str) -> List[str]:
        """Dynamically parse compound command into logical parts"""
        # Check for multi-operation patterns
        multi_op_patterns = [
            "separate tabs",
            "different tabs",
            "multiple tabs",
            "each tab",
            "respectively",
        ]
        if any(pattern in command_text.lower() for pattern in multi_op_patterns):
            # This is a single complex command with multiple targets
            return [command_text]

        # First check for implicit compound (no connector)
        words = command_text.lower().split()
        if len(words) >= 3:
            # Look for app names to find split points
            for i, word in enumerate(words):
                if (
                    self.pattern_learner.is_learned_app(word)
                    and i > 0
                    and i < len(words) - 1
                ):
                    # Check if there's a verb before and action after
                    if words[i - 1] in self.pattern_learner.app_verbs:
                        # Check if there's an action after the app
                        remaining_words = words[i + 1 :]
                        if any(
                            w
                            in self.pattern_learner.app_verbs
                            | {"search", "navigate", "go", "type"}
                            for w in remaining_words
                        ):
                            # Split at the app name
                            part1 = " ".join(words[: i + 1])
                            part2 = " ".join(words[i + 1 :])
                            logger.info(
                                f"[PARSE] Split implicit compound: '{part1}' | '{part2}'"
                            )
                            return [part1, part2]

        # Dynamic connector detection
        connectors = [
            " and ",
            " then ",
            ", and ",
            ", then ",
            " && ",
            " ; ",
            " plus ",
            " also ",
        ]

        # Smart parsing - analyze command structure
        parts = []
        remaining = command_text

        # Find all connector positions
        connector_positions = []
        for connector in connectors:
            pos = 0
            while connector in remaining[pos:]:
                index = remaining.find(connector, pos)
                if index != -1:
                    connector_positions.append((index, connector))
                    pos = index + 1
                else:
                    break

        # Sort by position
        connector_positions.sort(key=lambda x: x[0])

        if not connector_positions:
            return [command_text]

        # Analyze each potential split point
        last_pos = 0
        for pos, connector in connector_positions:
            before = remaining[last_pos:pos].strip()
            after = remaining[pos + len(connector) :].strip()

            # Use intelligent splitting logic
            should_split = self._should_split_at_connector(before, after, connector)

            if should_split:
                if before:
                    parts.append(before)
                last_pos = pos + len(connector)

        # Add remaining part
        final_part = remaining[last_pos:].strip()
        if final_part:
            parts.append(final_part)

        # If no valid splits, return original
        return parts if parts else [command_text]

    def _should_split_at_connector(
        self, before: str, after: str, connector: str
    ) -> bool:
        """Determine if we should split at this connector"""
        if not before or not after:
            return False

        # Get word analysis
        before_words = before.lower().split()
        after_words = after.lower().split()

        # Check if both sides have verbs (indicating separate commands)
        before_has_verb = any(
            word in self.pattern_learner.app_verbs | self.pattern_learner.system_verbs
            for word in before_words
        )
        after_has_verb = any(
            word in self.pattern_learner.app_verbs | self.pattern_learner.system_verbs
            for word in after_words[:3]
        )  # Check first 3 words of after

        if before_has_verb and after_has_verb:
            return True

        # Check if both sides mention apps
        before_has_app = any(
            self.pattern_learner.is_learned_app(word) for word in before_words
        )
        after_has_app = any(
            self.pattern_learner.is_learned_app(word) for word in after_words
        )

        if before_has_app and after_has_app and before_has_verb:
            return True

        # Don't split if it's part of a single concept
        single_concepts = {
            connector + "press enter",
            connector + "enter",
            connector + "return",
            "type",
            "write",
            "and then",
        }

        # Check if the connector is truly part of a single phrase
        full_text = (before + connector + after).lower()
        for concept in single_concepts:
            if concept in full_text:
                return False

        # Special handling for search/navigation commands
        # "search for X and Y" should not split, but "open X and search for Y" should
        if connector == " and " and "search for" in after.lower():
            # If before has an app operation, this should split
            if any(
                verb in before.lower() for verb in ["open", "launch", "start", "close"]
            ):
                return True

        # Don't split URLs or domains
        if self._contains_url_pattern(before + connector + after):
            url_start = before.rfind("http")
            if url_start == -1:
                url_start = before.rfind("www.")
            if url_start != -1:
                return False

        return True

    def _can_parallelize_commands(self, parts: List[str]) -> bool:
        """Dynamically determine if commands can be run in parallel"""
        if len(parts) < 2:
            return False

        # Analyze command dependencies
        command_analyses = []

        for i, part in enumerate(parts):
            words = part.lower().split()

            analysis = {
                "has_verb": any(
                    word
                    in self.pattern_learner.app_verbs
                    | self.pattern_learner.system_verbs
                    for word in words
                ),
                "has_app": any(
                    self.pattern_learner.is_learned_app(word) for word in words
                ),
                "has_dependency": False,
                "operation_type": None,
                "affects_state": False,
            }

            # Detect operation type
            for word in words:
                if word in self.pattern_learner.app_verbs:
                    if word in {"open", "launch", "start"}:
                        analysis["operation_type"] = "open"
                    elif word in {"close", "quit", "kill"}:
                        analysis["operation_type"] = "close"
                    break

            # Check for dependencies on previous commands
            dependency_indicators = {"then", "after", "next", "followed", "using"}
            if any(indicator in words for indicator in dependency_indicators):
                analysis["has_dependency"] = True

            # Check if command affects state that next command might depend on
            state_affecting = {"search", "type", "navigate", "click", "select", "focus"}
            if any(action in words for action in state_affecting):
                analysis["affects_state"] = True

            # Check for explicit references to previous results
            if i > 0:
                reference_words = {"it", "that", "there", "result"}
                if any(ref in words for ref in reference_words):
                    analysis["has_dependency"] = True

            command_analyses.append(analysis)

        # Determine if parallelizable
        # Commands can be parallel if:
        # 1. No command has dependencies
        # 2. No command affects state that others might use
        # 3. All are similar operation types

        has_dependencies = any(a["has_dependency"] for a in command_analyses)
        if has_dependencies:
            return False

        affects_state = any(a["affects_state"] for a in command_analyses)
        if affects_state:
            return False

        # Check operation types
        operation_types = [
            a["operation_type"] for a in command_analyses if a["operation_type"]
        ]
        if operation_types:
            # All same type = parallelizable
            return len(set(operation_types)) == 1

        # Default: if simple and independent, allow parallel
        all_simple = all(len(part.split()) <= 4 for part in parts)
        all_have_verbs = all(a["has_verb"] for a in command_analyses)

        return all_simple and all_have_verbs

    def _enhance_with_context(
        self, command: str, active_app: Optional[str], previous_result: Optional[Dict]
    ) -> str:
        """Enhance command with context from previous commands"""
        command_lower = command.lower()
        words = command_lower.split()

        # Dynamic pattern detection for navigation and search
        nav_indicators = {"go", "navigate", "browse", "visit", "open"}
        search_indicators = {"search", "find", "look", "google", "query"}

        # Check if this needs browser context
        has_nav = any(word in words for word in nav_indicators)
        has_search = any(word in words for word in search_indicators)

        if (has_nav or has_search) and active_app:
            # Check if active app is likely a browser (learned from system)
            browser_indicators = {"browser", "web", "internet"}
            is_browser = active_app.lower() in {"safari", "chrome", "firefox"} or any(
                indicator in active_app.lower() for indicator in browser_indicators
            )

            if is_browser:
                # Check if browser not already specified
                if not self.pattern_learner.is_learned_app(active_app):
                    # Learn this as a browser app
                    self.pattern_learner.learned_apps.add(active_app.lower())

                # Enhance command if browser not mentioned
                app_mentioned = any(
                    self.pattern_learner.is_learned_app(word) for word in words
                )
                if not app_mentioned:
                    # Add browser context
                    if has_search:
                        # Clean up the command - remove "and", "search", "search for" to get just the query
                        cleaned = command
                        # Remove leading "and" if present
                        if cleaned.lower().startswith("and "):
                            cleaned = cleaned[4:]
                        # Remove search-related words
                        cleaned = (
                            cleaned.replace("search for", "")
                            .replace("search", "")
                            .strip()
                        )
                        command = f"search in {active_app} for {cleaned}"
                    elif "go to" in command_lower:
                        command = command.replace(
                            "go to", f"tell {active_app} to go to"
                        )
                    else:
                        command = f"in {active_app} {command}"

        # Use previous result context if available
        if previous_result and previous_result.get("success"):
            # Could enhance with information from previous successful command
            pass

        return command

    def _parse_system_command(
        self, command_text: str
    ) -> Tuple[str, Optional[str], Dict[str, Any]]:
        """Dynamically parse system command to extract type, target, and parameters"""
        words = command_text.lower().split()
        command_type = None
        target = None
        params = {}

        # Detect command type based on verb patterns
        first_word = words[0] if words else ""

        # Tab/browser operations
        tab_indicators = {"tab", "tabs"}
        if any(indicator in words for indicator in tab_indicators):
            command_type = "tab_control"
            # Find browser if mentioned
            for word in words:
                if self.pattern_learner.is_learned_app(word):
                    # Check if it's likely a browser
                    if any(
                        browser_hint in word
                        for browser_hint in ["safari", "chrome", "firefox", "browser"]
                    ):
                        target = word
                        break
            # Extract URL if present
            url_patterns = ["go to", "navigate to", "and open", "open"]
            for pattern in url_patterns:
                if pattern in command_text.lower():
                    idx = command_text.lower().find(pattern)
                    url_part = command_text[idx + len(pattern) :].strip()
                    if url_part:
                        params["url"] = self._normalize_url(url_part)
                    break

        # App control operations
        elif first_word in self.pattern_learner.app_verbs:
            if first_word in {"open", "launch", "start"}:
                command_type = "app_open"
            elif first_word in {"close", "quit", "kill"}:
                command_type = "app_close"

            # Find target app
            for i, word in enumerate(words[1:], 1):
                if self.pattern_learner.is_learned_app(word):
                    target = word
                    break
                # Also check multi-word apps
                if i < len(words) - 1:
                    two_word = f"{word} {words[i+1]}"
                    if self.pattern_learner.is_learned_app(two_word):
                        target = two_word
                        break

        # System settings operations
        elif any(
            setting in words
            for setting in {"volume", "brightness", "wifi", "bluetooth", "screenshot"}
        ):
            command_type = "system_setting"
            # Determine which setting
            if "volume" in words:
                target = "volume"
                if "mute" in words:
                    params["action"] = "mute"
                elif "unmute" in words:
                    params["action"] = "unmute"
                else:
                    # Extract level
                    for word in words:
                        if word.isdigit():
                            params["level"] = int(word)
                            break
            elif "brightness" in words:
                target = "brightness"
                for word in words:
                    if word.isdigit():
                        params["level"] = int(word)
                        break
            elif "wifi" in words or "wi-fi" in words:
                target = "wifi"
                params["enable"] = "on" in words or "enable" in words
            elif "screenshot" in words:
                target = "screenshot"

        # Web operations
        elif any(
            web_verb in words
            for web_verb in {"search", "google", "browse", "navigate", "visit"}
        ):
            command_type = "web_action"
            # Determine specific action
            if "search" in words or "google" in words:
                params["action"] = "search"
                # Extract search query
                # Handle "search in X for Y" pattern first
                if (
                    "search in" in command_text.lower()
                    and " for " in command_text.lower()
                ):
                    # Extract query after "for"
                    for_idx = command_text.lower().find(" for ")
                    if for_idx != -1:
                        query = command_text[for_idx + 5 :].strip()
                        if query:
                            params["query"] = query
                            logger.info(
                                f"[PARSE] Extracted query from 'search in X for Y' pattern: '{query}'"
                            )
                else:
                    # Standard search patterns
                    search_patterns = ["search for", "google", "look up", "find"]
                    for pattern in search_patterns:
                        if pattern in command_text.lower():
                            idx = command_text.lower().find(pattern)
                            query = command_text[idx + len(pattern) :].strip()
                            if query:
                                params["query"] = query
                                logger.info(
                                    f"[PARSE] Extracted query from '{pattern}' pattern: '{query}'"
                                )
                                break
            else:
                params["action"] = "navigate"
                # Extract URL
                nav_patterns = ["go to", "navigate to", "visit", "browse to"]
                for pattern in nav_patterns:
                    if pattern in command_text.lower():
                        idx = command_text.lower().find(pattern)
                        url = command_text[idx + len(pattern) :].strip()
                        if url:
                            params["url"] = self._normalize_url(url)
                            break

            # Find browser if specified
            # Check both original words and full command text (in case of "search in X for Y")
            if "in " in command_text.lower():
                # Extract browser from "in [browser]" pattern
                in_match = re.search(r"\bin\s+(\w+)\s+", command_text.lower())
                if in_match:
                    potential_browser = in_match.group(1)
                    if self.pattern_learner.is_learned_app(potential_browser):
                        target = potential_browser

            # If not found, check individual words
            if not target:
                for word in words:
                    if self.pattern_learner.is_learned_app(word):
                        if any(
                            hint in word
                            for hint in ["safari", "chrome", "firefox", "browser"]
                        ):
                            target = word
                            break

        # Multi-tab searches
        if (
            "separate tabs" in command_text.lower()
            or "different tabs" in command_text.lower()
        ):
            command_type = "multi_tab_search"
            params["multi_tab"] = True

        # Typing operations
        elif "type" in words:
            command_type = "type_text"
            # Extract text to type
            idx = command_text.lower().find("type")
            text = command_text[idx + 4 :].strip()
            # Remove trailing instructions
            text = text.replace(" and press enter", "").replace(" and enter", "")
            params["text"] = text
            params["press_enter"] = "enter" in command_text.lower()

        return command_type or "unknown", target, params

    def _normalize_url(self, url: str) -> str:
        """Normalize URL input"""
        url = url.strip()

        # Handle common shortcuts
        if url.lower() in {"google", "google.com"}:
            return "https://google.com"

        # Add protocol if missing
        if not url.startswith(("http://", "https://")):
            if "." in url:
                return f"https://{url}"
            else:
                # Assume .com for single words
                return f"https://{url}.com"

        return url

    def _detect_default_browser(self) -> str:
        """Detect the default browser dynamically"""
        # Try to get default browser from system
        try:
            import subprocess

            result = subprocess.run(
                [
                    "defaults",
                    "read",
                    "com.apple.LaunchServices/com.apple.launchservices.secure",
                    "LSHandlers",
                ],
                capture_output=True,
                text=True,
            )
            if "safari" in result.stdout.lower():
                return "safari"
            elif "chrome" in result.stdout.lower():
                return "chrome"
            elif "firefox" in result.stdout.lower():
                return "firefox"
        except:
            pass

        # Default fallback
        return "safari"

    async def _execute_display_action(self, display_ref, original_command: str) -> Dict[str, Any]:
        """
        Execute display action based on resolved DisplayReference

        This is the new direct routing system that uses display_ref.action
        instead of pattern matching.

        Args:
            display_ref: DisplayReference from display_reference_handler
            original_command: Original user command (for context)

        Returns:
            Dict with success status and response
        """
        from context_intelligence.handlers.display_reference_handler import ActionType, ModeType

        logger.info(
            f"[DISPLAY-ACTION] Executing: action={display_ref.action.value}, "
            f"display={display_ref.display_name}, mode={display_ref.mode.value if display_ref.mode else 'auto'}"
        )

        try:
            # Get display monitor instance
            monitor = None
            if hasattr(self, '_app') and self._app:
                if hasattr(self._app.state, 'display_monitor'):
                    monitor = self._app.state.display_monitor

            if monitor is None:
                from display import get_display_monitor
                monitor = get_display_monitor()

            # Route based on action type
            if display_ref.action == ActionType.CONNECT:
                return await self._action_connect_display(monitor, display_ref, original_command)

            elif display_ref.action == ActionType.DISCONNECT:
                return await self._action_disconnect_display(monitor, display_ref, original_command)

            elif display_ref.action == ActionType.CHANGE_MODE:
                return await self._action_change_mode(monitor, display_ref, original_command)

            elif display_ref.action == ActionType.QUERY_STATUS:
                return await self._action_query_status(monitor, display_ref, original_command)

            elif display_ref.action == ActionType.LIST_DISPLAYS:
                return await self._action_list_displays(monitor, display_ref, original_command)

            else:
                logger.warning(f"[DISPLAY-ACTION] Unknown action: {display_ref.action.value}")
                return {
                    "success": False,
                    "response": f"I don't know how to perform action: {display_ref.action.value}",
                }

        except Exception as e:
            logger.error(f"[DISPLAY-ACTION] Error: {e}", exc_info=True)
            return {
                "success": False,
                "response": f"Error executing display action: {str(e)}",
            }

    async def _action_connect_display(self, monitor, display_ref, original_command: str) -> Dict[str, Any]:
        """Execute CONNECT action"""
        from context_intelligence.handlers.display_reference_handler import ModeType

        display_name = display_ref.display_name
        display_id = display_ref.display_id or display_name.lower().replace(" ", "-")
        mode = display_ref.mode

        logger.info(f"[DISPLAY-ACTION] Connecting to '{display_name}' (id={display_id})")

        # Determine mode string for monitor.connect_display
        mode_str = "extended" if mode == ModeType.EXTENDED else \
                   "entire" if mode == ModeType.ENTIRE_SCREEN else \
                   "window" if mode == ModeType.WINDOW else \
                   "mirror"  # Default

        try:
            # Connect using display monitor
            result = await monitor.connect_display(display_id)

            if result.get("success"):
                # Generate time-aware response
                from datetime import datetime
                hour = datetime.now().hour
                greeting = "Good morning" if 5 <= hour < 12 else \
                          "Good afternoon" if 12 <= hour < 17 else \
                          "Good evening" if 17 <= hour < 21 else \
                          "Good night"

                response = f"{greeting}! Connected to {display_name}, sir."

                # Add mode info if specified
                if mode:
                    response += f" Display mode: {mode.value}."

                return {
                    "success": True,
                    "response": response,
                    "display_name": display_name,
                    "display_id": display_id,
                    "mode": mode_str,
                    "action": "connect",
                    "resolution_strategy": display_ref.resolution_strategy.value,
                    "confidence": display_ref.confidence
                }
            else:
                return {
                    "success": False,
                    "response": result.get("message", f"Unable to connect to {display_name}."),
                    "display_name": display_name,
                }

        except Exception as e:
            logger.error(f"[DISPLAY-ACTION] Connect error: {e}", exc_info=True)
            return {
                "success": False,
                "response": f"Error connecting to {display_name}: {str(e)}",
            }

    async def _action_disconnect_display(self, monitor, display_ref, original_command: str) -> Dict[str, Any]:
        """Execute DISCONNECT action"""
        display_name = display_ref.display_name
        display_id = display_ref.display_id or display_name.lower().replace(" ", "-")

        logger.info(f"[DISPLAY-ACTION] Disconnecting from '{display_name}' (id={display_id})")

        try:
            result = await monitor.disconnect_display(display_id)

            if result.get("success"):
                return {
                    "success": True,
                    "response": f"Disconnected from {display_name}, sir.",
                    "display_name": display_name,
                    "display_id": display_id,
                    "action": "disconnect"
                }
            else:
                return {
                    "success": False,
                    "response": result.get("message", f"Unable to disconnect from {display_name}."),
                    "display_name": display_name,
                }

        except Exception as e:
            logger.error(f"[DISPLAY-ACTION] Disconnect error: {e}", exc_info=True)
            return {
                "success": False,
                "response": f"Error disconnecting from {display_name}: {str(e)}",
            }

    async def _action_change_mode(self, monitor, display_ref, original_command: str) -> Dict[str, Any]:
        """Execute CHANGE_MODE action"""
        from context_intelligence.handlers.display_reference_handler import ModeType

        display_name = display_ref.display_name
        display_id = display_ref.display_id or display_name.lower().replace(" ", "-")
        mode = display_ref.mode

        if not mode:
            return {
                "success": False,
                "response": "Please specify which mode you'd like: entire screen, window, or extended display.",
            }

        logger.info(f"[DISPLAY-ACTION] Changing '{display_name}' to {mode.value} mode")

        # Map ModeType to mode string
        mode_str = "entire" if mode == ModeType.ENTIRE_SCREEN else \
                   "window" if mode == ModeType.WINDOW else \
                   "extended" if mode == ModeType.EXTENDED else \
                   "mirror"

        try:
            result = await monitor.change_display_mode(display_id, mode_str)

            if result.get("success"):
                return {
                    "success": True,
                    "response": f"Changed {display_name} to {mode.value} mode, sir.",
                    "display_name": display_name,
                    "mode": mode_str,
                    "action": "change_mode"
                }
            else:
                return {
                    "success": False,
                    "response": result.get("message", f"Unable to change {display_name} to {mode.value} mode."),
                }

        except Exception as e:
            logger.error(f"[DISPLAY-ACTION] Change mode error: {e}", exc_info=True)
            return {
                "success": False,
                "response": f"Error changing mode: {str(e)}",
            }

    async def _action_query_status(self, monitor, display_ref, original_command: str) -> Dict[str, Any]:
        """Execute QUERY_STATUS action"""
        logger.info(f"[DISPLAY-ACTION] Querying display status")

        try:
            status = monitor.get_status()
            connected = status.get('connected_displays', [])
            available = monitor.get_available_display_details()

            if connected:
                display_names = [d.get('display_name', d) for d in connected]
                response = f"You have {len(connected)} display(s) connected: {', '.join(display_names)}."
            else:
                response = "No displays are currently connected."

            if available:
                avail_names = [d['display_name'] for d in available]
                response += f" Available displays: {', '.join(avail_names)}."

            return {
                "success": True,
                "response": response,
                "connected_displays": connected,
                "available_displays": available,
                "action": "query_status"
            }

        except Exception as e:
            logger.error(f"[DISPLAY-ACTION] Query status error: {e}", exc_info=True)
            return {
                "success": False,
                "response": f"Error querying display status: {str(e)}",
            }

    async def _action_list_displays(self, monitor, display_ref, original_command: str) -> Dict[str, Any]:
        """Execute LIST_DISPLAYS action"""
        logger.info(f"[DISPLAY-ACTION] Listing available displays")

        try:
            available = monitor.get_available_display_details()

            if available:
                names = [d['display_name'] for d in available]
                response = f"Available displays: {', '.join(names)}."
            else:
                response = "No displays are currently available."

            return {
                "success": True,
                "response": response,
                "available_displays": available,
                "action": "list_displays"
            }

        except Exception as e:
            logger.error(f"[DISPLAY-ACTION] List displays error: {e}", exc_info=True)
            return {
                "success": False,
                "response": f"Error listing displays: {str(e)}",
            }

    async def _execute_display_command(self, command_text: str) -> Dict[str, Any]:
        """
        Execute display/screen mirroring commands

        Handles commands like:
        - "Living Room TV" (implicit: connect to Living Room TV)
        - "screen mirror my Mac to the Living Room TV"
        - "connect to Living Room TV"
        - "connect to the TV" (uses context to resolve "the TV")
        - "extend display to Sony TV"
        - "airplay to Living Room TV"
        - "stop living room tv"
        - "disconnect from living room tv"
        - "disconnect from that display" (uses context)
        - "stop screen mirroring"
        - "change to extended display"
        - "switch to entire screen"
        - "set to window mode"

        Flow:
        1. TV is in standby mode (AirPlay chip active, broadcasts availability)
        2. macOS Control Center sees "Living Room TV"
        3. JARVIS detects "Living Room TV" via DNS-SD
        4. User command triggers AirPlay connection request
        5. Sony TV receives wake signal → Powers ON automatically
        6. Mac screen appears on Sony TV
        """
        command_lower = command_text.lower()
        logger.info(f"[DISPLAY] Processing display command: '{command_text}'")

        # NEW: Try display reference handler first for intelligent voice command resolution
        display_ref = None
        if self.display_reference_handler:
            try:
                display_ref = await self.display_reference_handler.handle_voice_command(command_text)

                if display_ref:
                    logger.info(
                        f"[DISPLAY] Display reference resolved: {display_ref.display_name} "
                        f"(action={display_ref.action.value}, mode={display_ref.mode.value if display_ref.mode else None}, "
                        f"confidence={display_ref.confidence:.2f}, strategy={display_ref.resolution_strategy.value})"
                    )

                    # NEW: Direct action routing based on display_ref.action
                    # This bypasses the old pattern matching logic and goes straight to execution
                    try:
                        result = await self._execute_display_action(display_ref, command_text)

                        # Learn from success/failure
                        if result.get("success"):
                            self.display_reference_handler.learn_from_success(command_text, display_ref)
                            logger.info(f"[DISPLAY] ✅ Action completed successfully - learned from: '{command_text}'")
                        else:
                            self.display_reference_handler.learn_from_failure(command_text, display_ref)
                            logger.warning(f"[DISPLAY] ❌ Action failed - learned from: '{command_text}'")

                        return result

                    except Exception as e:
                        logger.error(f"[DISPLAY] Error executing display action: {e}", exc_info=True)
                        # Learn from failure
                        self.display_reference_handler.learn_from_failure(command_text, display_ref)
                        # Fall through to legacy logic as fallback
                        logger.warning("[DISPLAY] Falling back to legacy display command logic")

            except Exception as e:
                logger.warning(f"[DISPLAY] Display reference handler error (continuing with fallback): {e}")
                # Continue with existing logic if handler fails

        try:
            # Try to get the running display monitor instance
            monitor = None

            # Check if we have app reference
            if hasattr(self, '_app') and self._app:
                if hasattr(self._app.state, 'display_monitor'):
                    monitor = self._app.state.display_monitor
                    logger.info("[DISPLAY] Using running display monitor from app.state")

            # Fallback: get singleton instance
            if monitor is None:
                from display import get_display_monitor
                monitor = get_display_monitor()
                logger.info("[DISPLAY] Using display monitor singleton")

            # Check if this is a mode change command
            mode_keywords = ["change", "switch", "set"]
            mode_types = {
                "entire": ["entire", "entire screen", "full screen"],
                "window": ["window", "window or app", "app"],
                "extended": ["extended", "extend", "extended display"]
            }

            is_mode_change = any(keyword in command_lower for keyword in mode_keywords)
            detected_mode = None

            if is_mode_change:
                # Detect which mode the user wants
                for mode_key, mode_phrases in mode_types.items():
                    if any(phrase in command_lower for phrase in mode_phrases):
                        detected_mode = mode_key
                        break

            if is_mode_change and detected_mode:
                # Handle mode change
                logger.info(f"[DISPLAY] Detected mode change command to '{detected_mode}'")

                # Check config for connected displays
                status = monitor.get_status()
                connected_displays = list(status.get('connected_displays', []))

                logger.debug(f"[DISPLAY] Connected displays: {connected_displays}")

                # If only one display is connected, change its mode
                if len(connected_displays) == 1:
                    display_id = connected_displays[0]
                    logger.info(f"[DISPLAY] Changing '{display_id}' to {detected_mode} mode...")

                    result = await monitor.change_display_mode(display_id, detected_mode)

                    if result.get("success"):
                        mode_name = result.get("mode", detected_mode)
                        return {
                            "success": True,
                            "response": f"Changed to {mode_name} mode, sir.",
                            "mode": mode_name,
                        }
                    else:
                        return {
                            "success": False,
                            "response": result.get("message", f"Unable to change to {detected_mode} mode."),
                        }
                elif len(connected_displays) > 1:
                    # Multiple displays connected, need to specify which one
                    return {
                        "success": False,
                        "response": f"Multiple displays are connected. Please specify which display to change: {', '.join(connected_displays)}",
                        "connected_displays": connected_displays,
                    }
                else:
                    return {
                        "success": False,
                        "response": "No displays are currently connected.",
                    }

            # Check if this is a disconnection command
            disconnect_keywords = ["stop", "disconnect", "turn off", "disable"]
            is_disconnect = any(keyword in command_lower for keyword in disconnect_keywords)

            # Make sure it's not a mode change command being misdetected
            if is_disconnect and not is_mode_change:
                # Handle disconnection
                logger.info(f"[DISPLAY] Detected disconnection command")

                # Check config for monitored displays
                status = monitor.get_status()
                connected_displays = list(status.get('connected_displays', []))

                logger.debug(f"[DISPLAY] Connected displays: {connected_displays}")

                # If only one display is connected, disconnect it
                if len(connected_displays) == 1:
                    display_id = connected_displays[0]
                    logger.info(f"[DISPLAY] Disconnecting from '{display_id}'...")

                    result = await monitor.disconnect_display(display_id)

                    if result.get("success"):
                        return {
                            "success": True,
                            "response": "Display disconnected, sir.",
                        }
                    else:
                        return {
                            "success": False,
                            "response": result.get("message", "Unable to disconnect display."),
                        }
                elif len(connected_displays) > 1:
                    # Multiple displays connected, need to specify which one
                    return {
                        "success": False,
                        "response": f"Multiple displays are connected. Please specify which one to disconnect: {', '.join(connected_displays)}",
                        "connected_displays": connected_displays,
                    }
                else:
                    return {
                        "success": False,
                        "response": "No displays are currently connected.",
                    }

            # Extract display name from command (for connection)
            # Look for TV names, room names, or brand names
            display_name = None

            # Check config for monitored displays
            status = monitor.get_status()
            available_displays = monitor.get_available_display_details()

            logger.debug(f"[DISPLAY] Available displays: {[d['display_name'] for d in available_displays]}")

            # Match display name in command text
            display_id = None
            for display_info in available_displays:
                name = display_info["display_name"]
                # Check if display name appears in command (case insensitive)
                if name.lower() in command_lower:
                    display_name = name
                    display_id = display_info["display_id"]
                    break

            if not display_name:
                # Try to extract room name or TV reference
                import re
                # Match patterns like "living room", "bedroom", "sony", "lg", etc.
                patterns = [
                    r"(living\s*room|bedroom|kitchen|office)\s*tv",
                    r"(sony|lg|samsung)\s*tv",
                    r"to\s+([a-z\s]+tv)",
                ]
                for pattern in patterns:
                    match = re.search(pattern, command_lower)
                    if match:
                        extracted = match.group(0).replace("to ", "").strip()
                        # Try to match with available displays
                        for display_info in available_displays:
                            if extracted.lower() in display_info["display_name"].lower():
                                display_name = display_info["display_name"]
                                display_id = display_info["display_id"]
                                break
                        if display_name:
                            break

            # Determine mode (mirror vs extend)
            mode = "mirror" if "mirror" in command_lower else "extend"

            if not display_id:
                # No specific display found, show available options
                if available_displays:
                    names = [d["display_name"] for d in available_displays]
                    return {
                        "success": False,
                        "response": f"I couldn't determine which display to connect to. Available displays: {', '.join(names)}. Please specify one.",
                        "available_displays": names,
                    }
                else:
                    return {
                        "success": False,
                        "response": "No displays are currently available. Please ensure your TV or display is powered on and connected to the network.",
                    }

            logger.info(f"[DISPLAY] Connecting to '{display_name}' (id: {display_id}) in {mode} mode...")

            # Connect to display using display_id
            result = await monitor.connect_display(display_id)

            if result.get("success"):
                return {
                    "success": True,
                    "response": f"Connected to {display_name}, sir. Your screen is now being {mode}ed.",
                    "display_name": display_name,
                    "mode": mode,
                }
            else:
                return {
                    "success": False,
                    "response": result.get("message", f"Unable to connect to {display_name}."),
                    "display_name": display_name,
                }

        except Exception as e:
            logger.error(f"[DISPLAY] Error executing display command: {e}", exc_info=True)
            return {
                "success": False,
                "response": f"I encountered an error while trying to connect to the display: {str(e)}",
            }

    async def _execute_system_command(self, command_text: str) -> Dict[str, Any]:
        """Dynamically execute system commands without hardcoding"""

        # Check if this is actually a voice unlock command misclassified as system
        command_lower = command_text.lower()
        if ("voice" in command_lower and "unlock" in command_lower) or (
            "enable" in command_lower and "voice unlock" in command_lower
        ):
            # Redirect to voice unlock handler
            handler = await self._get_handler(CommandType.VOICE_UNLOCK)
            if handler:
                result = await handler.handle_command(command_text)
                return {
                    "success": result.get(
                        "success", result.get("type") == "voice_unlock"
                    ),
                    "response": result.get("message", result.get("response", "")),
                    "command_type": "voice_unlock",
                    **result,
                }

        try:
            from system_control.macos_controller import MacOSController
            from system_control.dynamic_app_controller import get_dynamic_app_controller

            macos_controller = MacOSController()
            dynamic_controller = get_dynamic_app_controller()

            # Check for lock/unlock screen commands first
            # Use the existing voice unlock integration for proper daemon support
            if (
                "lock" in command_lower or "unlock" in command_lower
            ) and "screen" in command_lower:
                logger.info(
                    f"[SYSTEM] Screen lock/unlock command detected, using voice unlock handler"
                )
                try:
                    from api.simple_unlock_handler import handle_unlock_command

                    # Pass the command to the existing unlock handler which integrates with the daemon
                    result = await handle_unlock_command(command_text)

                    # Ensure we return a properly formatted result
                    if isinstance(result, dict):
                        # Add command_type if not present
                        if "command_type" not in result:
                            result["command_type"] = (
                                "screen_lock"
                                if "lock" in command_lower
                                else "screen_unlock"
                            )
                        return result
                    else:
                        # Fallback to macos_controller if the unlock handler returns unexpected format
                        logger.warning(
                            f"[SYSTEM] Unexpected result from unlock handler, falling back"
                        )
                        result = await macos_controller.handle_command(command_text)
                        return result

                except ImportError:
                    logger.warning(
                        f"[SYSTEM] Simple unlock handler not available, using macos_controller"
                    )
                    result = await macos_controller.handle_command(command_text)
                    return result
                except Exception as e:
                    logger.error(
                        f"[SYSTEM] Error with unlock handler: {e}, falling back"
                    )
                    result = await macos_controller.handle_command(command_text)
                    return result

            # Parse command dynamically
            command_type, target, params = self._parse_system_command(command_text)

            logger.info(f"[SYSTEM] Parsing '{command_text}'")
            logger.info(
                f"[SYSTEM] Parsed: type={command_type}, target={target}, params={params}"
            )

            # Execute based on parsed command type
            if command_type == "tab_control":
                # Handle new tab operations
                browser = target or self._detect_default_browser()
                url = params.get("url")
                success, message = macos_controller.open_new_tab(browser, url)
                return {"success": success, "response": message}

            elif command_type == "app_open":
                # Open application
                if target:
                    success, message = await dynamic_controller.open_app_intelligently(
                        target
                    )
                    return {"success": success, "response": message}
                else:
                    return {
                        "success": False,
                        "response": "Please specify which app to open",
                    }

            elif command_type == "app_close":
                # Close application
                if target:
                    success, message = await dynamic_controller.close_app_intelligently(
                        target
                    )
                    return {"success": success, "response": message}
                else:
                    return {
                        "success": False,
                        "response": "Please specify which app to close",
                    }

            elif command_type == "system_setting":
                # Handle system settings
                if target == "volume":
                    action = params.get("action")
                    if action == "mute":
                        success, message = macos_controller.mute_volume(True)
                    elif action == "unmute":
                        success, message = macos_controller.mute_volume(False)
                    elif "level" in params:
                        success, message = macos_controller.set_volume(params["level"])
                    else:
                        return {
                            "success": False,
                            "response": "Please specify volume level or mute/unmute",
                        }
                    return {"success": success, "response": message}

                elif target == "brightness":
                    if "level" in params:
                        level = params["level"] / 100.0  # Convert to 0.0-1.0
                        success, message = macos_controller.adjust_brightness(level)
                    else:
                        return {
                            "success": False,
                            "response": "Please specify brightness level (0-100)",
                        }
                    return {"success": success, "response": message}

                elif target == "wifi":
                    enable = params.get("enable", True)
                    success, message = macos_controller.toggle_wifi(enable)
                    return {"success": success, "response": message}

                elif target == "screenshot":
                    success, message = macos_controller.take_screenshot()
                    return {"success": success, "response": message}

                else:
                    return {
                        "success": False,
                        "response": f"Unknown system setting: {target}",
                    }

            elif command_type == "web_action":
                # Handle web navigation and searches
                action = params.get("action")
                browser = target

                if action == "search" and "query" in params:
                    success, message = macos_controller.web_search(
                        params["query"], browser=browser
                    )
                    return {"success": success, "response": message}

                elif action == "navigate" and "url" in params:
                    success, message = macos_controller.open_url(params["url"], browser)
                    return {"success": success, "response": message}

                else:
                    return {
                        "success": False,
                        "response": "Please specify what to search for or where to navigate",
                    }

            elif command_type == "multi_tab_search":
                # Handle multi-tab searches dynamically
                return await self._handle_multi_tab_search(command_text, target)

            elif command_type == "type_text":
                # Handle typing
                text = params.get("text", "")
                press_enter = params.get("press_enter", False)
                browser = target

                if text:
                    success, message = macos_controller.type_in_browser(
                        text, browser, press_enter
                    )
                    return {"success": success, "response": message}
                else:
                    return {"success": False, "response": "Please specify what to type"}

            else:
                # Unknown command type - try to be helpful
                return {
                    "success": False,
                    "response": f"I'm not sure how to handle that command. I parsed it as '{command_type}' but couldn't execute it. Try rephrasing or being more specific.",
                }

        except Exception as e:
            logger.error(f"Error executing system command: {e}", exc_info=True)
            return {
                "success": False,
                "response": f"Failed to execute system command: {str(e)}",
            }

    async def _handle_multi_tab_search(
        self, command_text: str, browser: Optional[str]
    ) -> Dict[str, Any]:
        """Handle searches across multiple tabs dynamically"""
        try:
            from system_control.macos_controller import MacOSController
            from system_control.dynamic_app_controller import get_dynamic_app_controller

            macos_controller = MacOSController()
            dynamic_controller = get_dynamic_app_controller()

            # Extract search terms dynamically
            search_terms = self._extract_multi_search_terms(command_text)

            if not search_terms:
                return {
                    "success": False,
                    "response": "Couldn't identify what to search for",
                }

            # Detect browser if not specified
            if not browser:
                browser = self._detect_browser_from_context(command_text)

            # Open browser if needed
            if "open" in command_text.lower():
                success, _ = await dynamic_controller.open_app_intelligently(browser)
                if success:
                    await asyncio.sleep(1.5)

            # Open tabs for each search term
            results = []
            for i, term in enumerate(search_terms):
                if i == 0:
                    # First search in current tab
                    success, msg = macos_controller.web_search(term, browser=browser)
                else:
                    # Subsequent searches in new tabs
                    await asyncio.sleep(0.5)
                    search_url = f"https://google.com/search?q={term.replace(' ', '+')}"
                    success, msg = macos_controller.open_new_tab(browser, search_url)
                results.append(success)

            if all(results):
                terms_str = " and ".join(f"'{term}'" for term in search_terms)
                return {
                    "success": True,
                    "response": f"Searching for {terms_str} in separate tabs, Sir",
                }
            else:
                return {"success": False, "response": "Had trouble opening some tabs"}

        except Exception as e:
            logger.error(f"Error in multi-tab search: {e}", exc_info=True)
            return {
                "success": False,
                "response": f"Failed to perform multi-tab search: {str(e)}",
            }

    def _extract_multi_search_terms(self, command_text: str) -> List[str]:
        """Extract multiple search terms from command"""
        # Look for pattern like "search for X and Y and Z"
        patterns = ["search for", "google", "look up", "find"]

        for pattern in patterns:
            if pattern in command_text.lower():
                idx = command_text.lower().find(pattern)
                after_pattern = command_text[idx + len(pattern) :].strip()

                # Remove trailing instructions
                for instruction in [
                    "on separate tabs",
                    "in different tabs",
                    "on multiple tabs",
                ]:
                    if instruction in after_pattern.lower():
                        after_pattern = after_pattern[
                            : after_pattern.lower().find(instruction)
                        ].strip()

                # Split by 'and' to get individual terms
                terms = []
                parts = after_pattern.split(" and ")
                for part in parts:
                    part = part.strip()
                    if part and not any(
                        skip in part.lower() for skip in ["open", "close", "quit"]
                    ):
                        terms.append(part)

                return terms

        return []

    def _detect_browser_from_context(self, command_text: str) -> str:
        """Detect which browser is mentioned in command"""
        words = command_text.lower().split()

        # Check for any learned app that might be a browser
        for word in words:
            if self.pattern_learner.is_learned_app(word):
                # Check if it's likely a browser
                browser_hints = ["safari", "chrome", "firefox", "browser", "web"]
                if any(hint in word for hint in browser_hints):
                    return word

        # Default to detected default browser
        return self._detect_default_browser()


# Singleton instance
_unified_processor = None


def get_unified_processor(api_key: Optional[str] = None, app=None) -> UnifiedCommandProcessor:
    """Get or create the unified command processor"""
    global _unified_processor
    if _unified_processor is None:
        _unified_processor = UnifiedCommandProcessor(api_key, app=app)
    elif app is not None and not hasattr(_unified_processor, '_app'):
        # Update existing processor with app reference
        _unified_processor._app = app
    return _unified_processor
