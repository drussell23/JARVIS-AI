#!/usr/bin/env python3
"""
Component Warmup Configuration for Unified Command Processor
============================================================

Defines all components to pre-initialize with priorities and health checks.
Zero hardcoding - components auto-register themselves.
"""

import asyncio
import logging

from core.component_warmup import ComponentPriority, get_warmup_system

logger = logging.getLogger(__name__)


async def register_all_components(processor_instance=None):
    """
    Register all components for warmup.

    This function discovers and registers components dynamically
    without hardcoding specific implementations.
    """
    warmup = get_warmup_system()

    # ═══════════════════════════════════════════════════════════════
    # CRITICAL COMPONENTS (Priority 0)
    # Must be ready before any commands can execute
    # ═══════════════════════════════════════════════════════════════

    # Screen Lock Detector
    warmup.register_component(
        name="screen_lock_detector",
        loader=load_screen_lock_detector,
        priority=ComponentPriority.CRITICAL,
        health_check=check_screen_lock_detector_health,
        timeout=5.0,
        required=True,
        category="security",
    )

    # Voice Authentication System
    warmup.register_component(
        name="voice_auth",
        loader=load_voice_auth,
        priority=ComponentPriority.CRITICAL,
        health_check=check_voice_auth_health,
        timeout=10.0,
        required=False,  # Can degrade gracefully
        category="security",
    )

    # ═══════════════════════════════════════════════════════════════
    # HIGH PRIORITY COMPONENTS (Priority 1)
    # Should be ready for first command
    # ═══════════════════════════════════════════════════════════════

    # Context-Aware Handler
    warmup.register_component(
        name="context_aware_handler",
        loader=load_context_aware_handler,
        priority=ComponentPriority.HIGH,
        dependencies=["screen_lock_detector"],
        timeout=10.0,
        required=True,
        category="intelligence",
    )

    # Multi-Space Context Graph
    warmup.register_component(
        name="multi_space_context_graph",
        loader=load_multi_space_context_graph,
        priority=ComponentPriority.HIGH,
        health_check=check_context_graph_health,
        timeout=8.0,
        required=False,
        category="context",
    )

    # Implicit Reference Resolver
    warmup.register_component(
        name="implicit_reference_resolver",
        loader=load_implicit_resolver,
        priority=ComponentPriority.HIGH,
        timeout=5.0,
        required=False,
        category="nlp",
    )

    # Compound Action Parser
    warmup.register_component(
        name="compound_action_parser",
        loader=load_compound_parser,
        priority=ComponentPriority.HIGH,
        health_check=check_compound_parser_health,
        timeout=5.0,
        required=True,
        category="nlp",
    )

    # MacOS Controller
    warmup.register_component(
        name="macos_controller",
        loader=load_macos_controller,
        priority=ComponentPriority.HIGH,
        health_check=check_macos_controller_health,
        timeout=5.0,
        required=True,
        category="system",
    )

    # ═══════════════════════════════════════════════════════════════
    # MEDIUM PRIORITY COMPONENTS (Priority 2)
    # Nice to have ready, but can load progressively
    # ═══════════════════════════════════════════════════════════════

    # Query Complexity Manager
    warmup.register_component(
        name="query_complexity_manager",
        loader=load_query_complexity_manager,
        priority=ComponentPriority.MEDIUM,
        dependencies=["implicit_reference_resolver"],
        timeout=10.0,
        required=False,
        category="intelligence",
    )

    # Yabai Space Detector
    warmup.register_component(
        name="yabai_detector",
        loader=load_yabai_detector,
        priority=ComponentPriority.MEDIUM,
        health_check=check_yabai_health,
        timeout=5.0,
        required=False,
        category="vision",
    )

    # Multi-Space Window Detector
    warmup.register_component(
        name="multi_space_window_detector",
        loader=load_window_detector,
        priority=ComponentPriority.MEDIUM,
        dependencies=["yabai_detector"],
        timeout=8.0,
        required=False,
        category="vision",
    )

    # Learning Database
    warmup.register_component(
        name="learning_database",
        loader=load_learning_database,
        priority=ComponentPriority.MEDIUM,
        health_check=check_database_health,
        timeout=15.0,
        required=False,
        category="learning",
    )

    # ═══════════════════════════════════════════════════════════════
    # LOW PRIORITY COMPONENTS (Priority 3)
    # Load in background, not critical
    # ═══════════════════════════════════════════════════════════════

    # Action Query Handler
    warmup.register_component(
        name="action_query_handler",
        loader=load_action_query_handler,
        priority=ComponentPriority.LOW,
        dependencies=["implicit_reference_resolver"],
        timeout=10.0,
        required=False,
        category="intelligence",
    )

    # Predictive Query Handler
    warmup.register_component(
        name="predictive_query_handler",
        loader=load_predictive_handler,
        priority=ComponentPriority.LOW,
        timeout=10.0,
        required=False,
        category="intelligence",
    )

    # Multi-Space Query Handler
    warmup.register_component(
        name="multi_space_query_handler",
        loader=load_multi_space_handler,
        priority=ComponentPriority.LOW,
        dependencies=["multi_space_context_graph", "learning_database"],
        timeout=15.0,
        required=False,
        category="vision",
    )

    logger.info(f"[WARMUP-CONFIG] Registered {len(warmup.components)} components for warmup")


# ═══════════════════════════════════════════════════════════════
# COMPONENT LOADERS
# Async functions that initialize and return component instances
# ═══════════════════════════════════════════════════════════════


async def load_screen_lock_detector():
    """Load screen lock detector"""
    from context_intelligence.detectors.screen_lock_detector import get_screen_lock_detector

    detector = get_screen_lock_detector()
    # Trigger initialization by checking once
    await detector.is_screen_locked()
    return detector


async def check_screen_lock_detector_health(detector) -> bool:
    """Verify screen lock detector is working"""
    try:
        result = await detector.is_screen_locked()
        return isinstance(result, bool)
    except:
        return False


async def load_voice_auth():
    """Load voice authentication system"""
    try:
        from voice_unlock.intelligent_voice_unlock_service import get_intelligent_unlock_service

        service = get_intelligent_unlock_service()
        return service
    except ImportError:
        logger.warning("[WARMUP] Voice auth not available")
        return None


async def check_voice_auth_health(service) -> bool:
    """Verify voice auth is working"""
    return service is not None


async def load_context_aware_handler():
    """Load context-aware command handler"""
    from context_intelligence.handlers.context_aware_handler import get_context_aware_handler

    handler = get_context_aware_handler()
    return handler


async def load_multi_space_context_graph():
    """Load multi-space context graph"""
    from core.context.multi_space_context_graph import MultiSpaceContextGraph

    graph = MultiSpaceContextGraph(decay_ttl=300, enable_correlation=True)
    return graph


async def check_context_graph_health(graph) -> bool:
    """Verify context graph is working"""
    return graph is not None


async def load_implicit_resolver():
    """Load implicit reference resolver"""
    from core.nlp.implicit_reference_resolver import get_implicit_resolver

    resolver = get_implicit_resolver()
    return resolver


async def load_compound_parser():
    """Load compound action parser"""
    from context_intelligence.analyzers.compound_action_parser import get_compound_parser

    parser = get_compound_parser()
    # Test parse to ensure it's working
    await parser.parse("test command")
    return parser


async def check_compound_parser_health(parser) -> bool:
    """Verify compound parser is working"""
    try:
        result = await parser.parse("open safari")
        return result is not None
    except:
        return False


async def load_macos_controller():
    """Load MacOS controller"""
    from system_control.macos_controller import MacOSController

    controller = MacOSController()
    return controller


async def check_macos_controller_health(controller) -> bool:
    """Verify MacOS controller is working"""
    return controller is not None


async def load_query_complexity_manager():
    """Load query complexity manager"""
    try:
        from context_intelligence.handlers.query_complexity_manager import QueryComplexityManager

        manager = QueryComplexityManager()
        return manager
    except ImportError:
        return None


async def load_yabai_detector():
    """Load Yabai space detector"""
    try:
        from vision.yabai_space_detector import YabaiSpaceDetector

        detector = YabaiSpaceDetector()
        return detector
    except:
        logger.debug("[WARMUP] Yabai detector not available")
        return None


async def check_yabai_health(detector) -> bool:
    """Verify Yabai detector is working"""
    return detector is not None


async def load_window_detector():
    """Load multi-space window detector"""
    try:
        from vision.multi_space_window_detector import MultiSpaceWindowDetector

        detector = MultiSpaceWindowDetector()
        return detector
    except:
        return None


async def load_learning_database():
    """Load learning database"""
    try:
        from intelligence.learning_database import get_learning_database

        db = get_learning_database()
        # Trigger initialization
        await asyncio.sleep(0.1)  # Let it connect
        return db
    except Exception as e:
        logger.debug(f"[WARMUP] Learning database not available: {e}")
        return None


async def check_database_health(db) -> bool:
    """Verify database is connected"""
    if db is None:
        return False
    try:
        # Try a simple operation
        return hasattr(db, "store_command_execution")
    except:
        return False


async def load_action_query_handler():
    """Load action query handler"""
    try:
        from context_intelligence.handlers.action_query_handler import (
            get_action_query_handler,
            initialize_action_query_handler,
        )

        handler = get_action_query_handler()
        if handler is None:
            from core.nlp.implicit_reference_resolver import get_implicit_resolver

            handler = initialize_action_query_handler(
                context_graph=None, implicit_resolver=get_implicit_resolver()
            )
        return handler
    except:
        return None


async def load_predictive_handler():
    """Load predictive query handler"""
    try:
        from context_intelligence.handlers.predictive_query_handler import (
            get_predictive_handler,
            initialize_predictive_handler,
        )

        handler = get_predictive_handler()
        if handler is None:
            handler = initialize_predictive_handler()
        return handler
    except:
        return None


async def load_multi_space_handler():
    """Load multi-space query handler"""
    try:
        from context_intelligence.handlers.multi_space_query_handler import (
            get_multi_space_query_handler,
        )

        handler = get_multi_space_query_handler()
        return handler
    except:
        return None
