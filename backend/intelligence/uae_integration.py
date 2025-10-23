#!/usr/bin/env python3
"""
UAE Integration Module
======================

Central integration point for Unified Awareness Engine (UAE) with JARVIS systems.

This module provides:
- UAE initialization and lifecycle management
- Integration with vision analyzer
- Integration with SAI engine
- Context Intelligence bootstrapping
- Metrics and monitoring
- Global UAE instance management

Usage in main.py:
    from intelligence.uae_integration import initialize_uae, get_uae

    # During startup
    uae = await initialize_uae(vision_analyzer=vision_analyzer)

    # In command handlers
    uae = get_uae()
    decision = await uae.get_element_position("control_center")

Author: Derek J. Russell
Date: October 2025
Version: 1.0.0
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from intelligence.unified_awareness_engine import (
    UnifiedAwarenessEngine,
    get_uae_engine
)
from vision.situational_awareness import (
    get_sai_engine,
    SituationalAwarenessEngine
)
from intelligence.learning_database import (
    get_learning_database,
    JARVISLearningDatabase
)

logger = logging.getLogger(__name__)

# Global UAE instance
_uae_instance: Optional[UnifiedAwarenessEngine] = None
_uae_initialized = False
_learning_db_instance: Optional[JARVISLearningDatabase] = None


async def initialize_uae(
    vision_analyzer=None,
    sai_monitoring_interval: float = 10.0,
    enable_auto_start: bool = True,
    knowledge_base_path: Optional[Path] = None,
    enable_learning_db: bool = True
) -> UnifiedAwarenessEngine:
    """
    Initialize UAE system with Learning Database integration

    Args:
        vision_analyzer: Claude Vision analyzer instance
        sai_monitoring_interval: SAI monitoring interval in seconds
        enable_auto_start: Whether to auto-start UAE
        knowledge_base_path: Path to knowledge base file
        enable_learning_db: Enable Learning Database integration

    Returns:
        Initialized UAE engine with persistent memory
    """
    global _uae_instance, _uae_initialized, _learning_db_instance

    if _uae_initialized and _uae_instance is not None:
        logger.info("[UAE-INIT] UAE already initialized")
        return _uae_instance

    logger.info("[UAE-INIT] Initializing Unified Awareness Engine with Learning Database...")

    try:
        # Step 1: Initialize Learning Database (if enabled)
        learning_db = None
        if enable_learning_db:
            logger.info("[UAE-INIT] Initializing Learning Database...")
            try:
                learning_db = await get_learning_database(config={
                    'cache_size': 2000,
                    'cache_ttl_seconds': 7200,
                    'enable_ml_features': True,
                    'auto_optimize': True,
                    'batch_insert_size': 100
                })
                _learning_db_instance = learning_db
                logger.info("[UAE-INIT] ✅ Learning Database initialized")
                logger.info(f"[UAE-INIT]    • Cache: 2000 entries, 2hr TTL")
                logger.info(f"[UAE-INIT]    • ML features: Enabled")
                logger.info(f"[UAE-INIT]    • Auto-optimize: Enabled")
            except Exception as e:
                logger.warning(f"[UAE-INIT] ⚠️  Learning Database failed to initialize: {e}")
                logger.info("[UAE-INIT]    • Continuing without persistent memory")

        # Step 2: Create SAI engine
        logger.info("[UAE-INIT] Creating Situational Awareness Engine...")
        sai_engine = get_sai_engine(
            vision_analyzer=vision_analyzer,
            monitoring_interval=sai_monitoring_interval,
            enable_auto_revalidation=True
        )
        logger.info("[UAE-INIT] ✅ SAI engine created")

        # Step 3: Create UAE engine with Learning DB
        logger.info("[UAE-INIT] Creating Unified Awareness Engine...")
        uae = get_uae_engine(
            sai_engine=sai_engine,
            vision_analyzer=vision_analyzer,
            learning_db=learning_db
        )

        # Set custom knowledge base path if provided
        if knowledge_base_path:
            uae.context_layer.knowledge_base_path = knowledge_base_path
            uae.context_layer._load_knowledge_base()

        # Initialize Learning DB connection in Context Layer
        if learning_db:
            await uae.context_layer.initialize_db(learning_db)
            logger.info("[UAE-INIT] ✅ Learning Database integrated with Context Intelligence")

        logger.info("[UAE-INIT] ✅ UAE engine created")

        # Step 4: Auto-start if enabled
        if enable_auto_start:
            logger.info("[UAE-INIT] Starting UAE...")
            await uae.start()
            logger.info("[UAE-INIT] ✅ UAE started and monitoring")

        # Store global instance
        _uae_instance = uae
        _uae_initialized = True

        logger.info("[UAE-INIT] ✅ UAE initialization complete with full intelligence stack")
        logger.info("[UAE-INIT]    • Context Intelligence: Active (with Learning DB)")
        logger.info("[UAE-INIT]    • Situational Awareness: Active (SAI)")
        logger.info("[UAE-INIT]    • Decision Fusion: Active")
        logger.info("[UAE-INIT]    • Persistent Memory: " + ("Enabled" if learning_db else "Disabled"))

        return uae

    except Exception as e:
        logger.error(f"[UAE-INIT] Failed to initialize UAE: {e}", exc_info=True)
        raise


def get_uae() -> Optional[UnifiedAwarenessEngine]:
    """
    Get global UAE instance

    Returns:
        UAE instance or None if not initialized
    """
    if not _uae_initialized or _uae_instance is None:
        logger.warning("[UAE] UAE not initialized - call initialize_uae() first")
        return None

    return _uae_instance


async def shutdown_uae():
    """Shutdown UAE system and Learning Database"""
    global _uae_instance, _uae_initialized, _learning_db_instance

    if not _uae_initialized or _uae_instance is None:
        return

    logger.info("[UAE-SHUTDOWN] Shutting down UAE...")

    try:
        # Stop UAE monitoring
        await _uae_instance.stop()
        logger.info("[UAE-SHUTDOWN] ✅ UAE stopped")

        # Close Learning Database
        if _learning_db_instance:
            logger.info("[UAE-SHUTDOWN] Closing Learning Database...")
            await _learning_db_instance.close()
            logger.info("[UAE-SHUTDOWN] ✅ Learning Database closed")
            _learning_db_instance = None

        _uae_initialized = False
        # Keep instance for potential restart

    except Exception as e:
        logger.error(f"[UAE-SHUTDOWN] Error during shutdown: {e}", exc_info=True)


def get_learning_db() -> Optional[JARVISLearningDatabase]:
    """
    Get global Learning Database instance

    Returns:
        Learning Database instance or None if not initialized
    """
    global _learning_db_instance
    return _learning_db_instance


def get_uae_metrics() -> Dict[str, Any]:
    """
    Get UAE metrics for monitoring

    Returns:
        Comprehensive metrics dict
    """
    if not _uae_initialized or _uae_instance is None:
        return {
            'initialized': False,
            'active': False,
            'error': 'UAE not initialized'
        }

    try:
        metrics = _uae_instance.get_comprehensive_metrics()
        metrics['initialized'] = True
        return metrics

    except Exception as e:
        logger.error(f"[UAE] Error getting metrics: {e}")
        return {
            'initialized': True,
            'active': False,
            'error': str(e)
        }


async def integrate_with_display_monitor(display_monitor):
    """
    Integrate UAE with display monitor service

    Args:
        display_monitor: Display monitor instance
    """
    uae = get_uae()
    if not uae:
        logger.warning("[UAE-INTEGRATION] UAE not available for display monitor integration")
        return

    try:
        # Register display changes with UAE
        def on_display_event(event_type, event_data):
            logger.info(f"[UAE-INTEGRATION] Display event: {event_type}")
            # UAE will automatically detect and adapt through SAI

        if hasattr(display_monitor, 'register_callback'):
            display_monitor.register_callback('all', on_display_event)
            logger.info("[UAE-INTEGRATION] ✅ Integrated with display monitor")

    except Exception as e:
        logger.error(f"[UAE-INTEGRATION] Display monitor integration failed: {e}")


async def integrate_with_vision_system(vision_analyzer):
    """
    Integrate UAE with vision system

    Args:
        vision_analyzer: Vision analyzer instance
    """
    uae = get_uae()
    if not uae:
        logger.warning("[UAE-INTEGRATION] UAE not available for vision integration")
        return

    try:
        # Update UAE with vision analyzer
        if uae.vision_analyzer is None:
            uae.vision_analyzer = vision_analyzer
            uae.situation_layer.sai_engine.vision_analyzer = vision_analyzer
            logger.info("[UAE-INTEGRATION] ✅ Integrated with vision system")

    except Exception as e:
        logger.error(f"[UAE-INTEGRATION] Vision system integration failed: {e}")


async def integrate_with_multi_space(multi_space_manager):
    """
    Integrate UAE with multi-space intelligence

    Args:
        multi_space_manager: Multi-space manager instance
    """
    uae = get_uae()
    if not uae:
        logger.warning("[UAE-INTEGRATION] UAE not available for multi-space integration")
        return

    try:
        # Register space change callback
        async def on_space_change(space_data):
            logger.info(f"[UAE-INTEGRATION] Space changed to: {space_data.get('space_id')}")
            # UAE SAI will detect and adapt automatically

        if hasattr(multi_space_manager, 'register_space_change_callback'):
            multi_space_manager.register_space_change_callback(on_space_change)
            logger.info("[UAE-INTEGRATION] ✅ Integrated with multi-space system")

    except Exception as e:
        logger.error(f"[UAE-INTEGRATION] Multi-space integration failed: {e}")


def register_uae_routes(app, prefix: str = "/api/uae"):
    """
    Register UAE API routes

    Args:
        app: FastAPI app instance
        prefix: API prefix
    """
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel

    router = APIRouter(prefix=prefix, tags=["UAE"])

    class ElementPositionRequest(BaseModel):
        element_id: str
        force_detect: bool = False

    @router.get("/status")
    async def get_status():
        """Get UAE status"""
        return {
            'initialized': _uae_initialized,
            'active': _uae_instance.is_active if _uae_instance else False
        }

    @router.get("/metrics")
    async def get_metrics():
        """Get comprehensive UAE metrics"""
        return get_uae_metrics()

    @router.post("/position")
    async def get_element_position(request: ElementPositionRequest):
        """Get element position using UAE"""
        uae = get_uae()
        if not uae:
            raise HTTPException(status_code=503, detail="UAE not initialized")

        try:
            decision = await uae.get_element_position(
                request.element_id,
                force_detect=request.force_detect
            )

            return {
                'success': True,
                'element_id': decision.element_id,
                'position': decision.chosen_position,
                'confidence': decision.confidence,
                'source': decision.decision_source.value,
                'reasoning': decision.reasoning
            }

        except Exception as e:
            logger.error(f"[UAE-API] Error getting position: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/start")
    async def start_uae():
        """Start UAE monitoring"""
        uae = get_uae()
        if not uae:
            raise HTTPException(status_code=503, detail="UAE not initialized")

        try:
            await uae.start()
            return {'success': True, 'message': 'UAE started'}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/stop")
    async def stop_uae():
        """Stop UAE monitoring"""
        uae = get_uae()
        if not uae:
            raise HTTPException(status_code=503, detail="UAE not initialized")

        try:
            await uae.stop()
            return {'success': True, 'message': 'UAE stopped'}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Register router with app
    app.include_router(router)
    logger.info(f"[UAE-API] Registered UAE routes at {prefix}")


# ============================================================================
# Helper Functions for Command Integration
# ============================================================================

async def uae_click(
    element_id: str,
    enable_communication: bool = True,
    communication_mode: str = 'normal',
    voice_callback=None,
    text_callback=None,
    **kwargs
) -> Dict[str, Any]:
    """
    Click using UAE intelligence with natural communication

    Args:
        element_id: Element to click
        enable_communication: Enable natural communication
        communication_mode: Communication verbosity (silent/minimal/normal/verbose/debug)
        voice_callback: Optional voice output callback
        text_callback: Optional text output callback
        **kwargs: Additional arguments

    Returns:
        Click result
    """
    uae = get_uae()
    if not uae:
        return {
            'success': False,
            'error': 'UAE not initialized'
        }

    try:
        # Import UAE-enhanced clicker and communication
        from display.uae_enhanced_control_center_clicker import get_uae_clicker
        from intelligence.uae_natural_communication import CommunicationMode

        # Convert string to CommunicationMode
        mode_map = {
            'silent': CommunicationMode.SILENT,
            'minimal': CommunicationMode.MINIMAL,
            'normal': CommunicationMode.NORMAL,
            'verbose': CommunicationMode.VERBOSE,
            'debug': CommunicationMode.DEBUG
        }
        comm_mode = mode_map.get(communication_mode.lower(), CommunicationMode.NORMAL)

        async with get_uae_clicker(
            enable_uae=True,
            enable_communication=enable_communication,
            communication_mode=comm_mode,
            voice_callback=voice_callback,
            text_callback=text_callback
        ) as clicker:
            result = await clicker.click(element_id, context=kwargs)

            return {
                'success': result.success,
                'coordinates': result.coordinates,
                'method': result.method_used,
                'confidence': result.confidence,
                'verification': result.verification_passed,
                'metadata': result.metadata
            }

    except Exception as e:
        logger.error(f"[UAE-CLICK] Error: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


async def uae_connect_device(
    device_name: str,
    enable_communication: bool = True,
    communication_mode: str = 'normal',
    voice_callback=None,
    text_callback=None
) -> Dict[str, Any]:
    """
    Connect to AirPlay device using UAE intelligence with natural communication

    Args:
        device_name: Device name
        enable_communication: Enable natural communication
        communication_mode: Communication verbosity (silent/minimal/normal/verbose/debug)
        voice_callback: Optional voice output callback
        text_callback: Optional text output callback

    Returns:
        Connection result
    """
    try:
        from display.uae_enhanced_control_center_clicker import get_uae_clicker
        from intelligence.uae_natural_communication import CommunicationMode

        # Convert string to CommunicationMode
        mode_map = {
            'silent': CommunicationMode.SILENT,
            'minimal': CommunicationMode.MINIMAL,
            'normal': CommunicationMode.NORMAL,
            'verbose': CommunicationMode.VERBOSE,
            'debug': CommunicationMode.DEBUG
        }
        comm_mode = mode_map.get(communication_mode.lower(), CommunicationMode.NORMAL)

        async with get_uae_clicker(
            enable_uae=True,
            enable_communication=enable_communication,
            communication_mode=comm_mode,
            voice_callback=voice_callback,
            text_callback=text_callback
        ) as clicker:
            result = await clicker.connect_to_device(device_name)
            return result

    except Exception as e:
        logger.error(f"[UAE-CONNECT] Error: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


# ============================================================================
# Import Function for main.py
# ============================================================================

def import_uae():
    """
    Import UAE components for main.py parallel loading

    Returns:
        Dict with UAE components
    """
    try:
        from intelligence.unified_awareness_engine import UnifiedAwarenessEngine
        from vision.situational_awareness import SituationalAwarenessEngine

        return {
            'UnifiedAwarenessEngine': UnifiedAwarenessEngine,
            'SituationalAwarenessEngine': SituationalAwarenessEngine,
            'initialize_uae': initialize_uae,
            'get_uae': get_uae,
            'available': True
        }

    except Exception as e:
        logger.error(f"[UAE-IMPORT] Failed to import UAE: {e}")
        return {
            'available': False,
            'error': str(e)
        }


# ============================================================================
# Example Usage
# ============================================================================

async def main():
    """Example UAE integration"""
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 80)
    print("UAE Integration Module - Demo")
    print("=" * 80)

    # Initialize UAE
    print("\n✅ Initializing UAE...")
    uae = await initialize_uae(enable_auto_start=True)

    # Get metrics
    print("\n📊 UAE Metrics:")
    metrics = get_uae_metrics()
    print(f"   Active: {metrics['engine']['active']}")
    print(f"   Total executions: {metrics['engine']['total_executions']}")

    # Test click
    print("\n🎯 Testing UAE click...")
    result = await uae_click("control_center")
    print(f"   Success: {result.get('success')}")

    # Shutdown
    print("\n⏹️  Shutting down...")
    await shutdown_uae()

    print("\n" + "=" * 80)
    print("✅ Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
