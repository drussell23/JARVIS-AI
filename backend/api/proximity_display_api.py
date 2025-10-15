"""
Proximity-Aware Display API
============================

REST API endpoints for proximity-aware display connection system.

Endpoints:
- GET /api/proximity-display/status - Get current proximity and display status
- GET /api/proximity-display/context - Get full proximity-display context
- POST /api/proximity-display/register - Register a display location
- POST /api/proximity-display/decision - Get connection decision
- GET /api/proximity-display/stats - Get service statistics
- POST /api/proximity-display/scan - Trigger proximity scan

Author: Derek Russell
Date: 2025-10-14
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

router = APIRouter(prefix="/api/proximity-display", tags=["proximity-display"])
logger = logging.getLogger(__name__)


# Request/Response models
class DisplayLocationRequest(BaseModel):
    """Request body for registering a display location"""
    display_id: int = Field(..., description="Display ID from CoreGraphics")
    location_name: str = Field(..., description="Human-readable name (e.g., 'Living Room TV')")
    zone: str = Field(..., description="Location zone (e.g., 'living_room', 'office')")
    min_distance: float = Field(..., description="Minimum expected distance in meters")
    max_distance: float = Field(..., description="Maximum expected distance in meters")
    bluetooth_beacon_uuid: Optional[str] = Field(None, description="Optional Bluetooth beacon UUID")
    auto_connect_enabled: bool = Field(True, description="Enable auto-connection")
    connection_priority: float = Field(0.5, description="Priority score (0.0-1.0)")
    tags: list[str] = Field(default_factory=list, description="Custom tags")


@router.get("/status")
async def get_proximity_display_status(
    include_context: bool = Query(False, description="Include full proximity context")
) -> Dict[str, Any]:
    """
    Get current proximity-display status
    
    Returns basic status including:
    - User proximity (closest device)
    - Available displays
    - Proximity scores
    - Nearest display
    
    Args:
        include_context: If True, includes full ProximityDisplayContext
        
    Returns:
        Status dict with proximity and display information
    """
    try:
        from proximity.proximity_display_bridge import get_proximity_display_bridge
        
        bridge = get_proximity_display_bridge()
        context = await bridge.get_proximity_display_context()
        
        status = {
            "user_proximity": context.user_proximity.to_dict() if context.user_proximity else None,
            "nearest_display": context.nearest_display,
            "proximity_scores": {
                str(k): v for k, v in context.proximity_scores.items()
            },
            "recommended_action": context.recommended_action.value if context.recommended_action else None,
            "display_count": len(context.available_displays),
            "timestamp": context.timestamp.isoformat()
        }
        
        if include_context:
            status["full_context"] = context.to_dict()
        
        return status
        
    except Exception as e:
        logger.error(f"[API] Error getting proximity status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/context")
async def get_proximity_display_context() -> Dict[str, Any]:
    """
    Get full proximity-display context
    
    Returns comprehensive context including:
    - User proximity data
    - All available displays
    - Display location configurations
    - Proximity scores for each display
    - Connection states
    - Recommended action
    
    Returns:
        Full ProximityDisplayContext as dict
    """
    try:
        from proximity.proximity_display_bridge import get_proximity_display_bridge
        
        bridge = get_proximity_display_bridge()
        context = await bridge.get_proximity_display_context()
        
        return context.to_dict()
        
    except Exception as e:
        logger.error(f"[API] Error getting context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/register")
async def register_display_location(request: DisplayLocationRequest) -> Dict[str, Any]:
    """
    Register or update a display location
    
    Allows manual configuration of display locations for proximity mapping.
    
    Args:
        request: DisplayLocationRequest with display metadata
        
    Returns:
        Success message with registered display info
    """
    try:
        from proximity.proximity_display_bridge import get_proximity_display_bridge
        
        bridge = get_proximity_display_bridge()
        
        success = await bridge.register_display_location(
            display_id=request.display_id,
            location_name=request.location_name,
            zone=request.zone,
            expected_proximity_range=(request.min_distance, request.max_distance),
            bluetooth_beacon_uuid=request.bluetooth_beacon_uuid,
            auto_connect_enabled=request.auto_connect_enabled,
            connection_priority=request.connection_priority,
            tags=request.tags
        )
        
        if success:
            return {
                "success": True,
                "message": f"Display location registered: {request.location_name}",
                "display_id": request.display_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to register display location")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Error registering display: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/decision")
async def make_connection_decision() -> Dict[str, Any]:
    """
    Make an intelligent connection decision
    
    Analyzes current proximity and display context to recommend
    a connection action (auto_connect, prompt_user, or ignore).
    
    Returns:
        ConnectionDecision with action, confidence, and reasoning
    """
    try:
        from proximity.proximity_display_bridge import get_proximity_display_bridge
        
        bridge = get_proximity_display_bridge()
        decision = await bridge.make_connection_decision()
        
        if decision:
            return decision.to_dict()
        else:
            return {
                "decision": None,
                "reason": "No proximity data or nearest display available"
            }
        
    except Exception as e:
        logger.error(f"[API] Error making decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scan")
async def trigger_proximity_scan() -> Dict[str, Any]:
    """
    Trigger an immediate Bluetooth proximity scan
    
    Forces a scan for nearby Bluetooth devices and returns results.
    
    Returns:
        List of detected proximity devices
    """
    try:
        from proximity.bluetooth_proximity_service import get_proximity_service
        
        service = get_proximity_service()
        devices = await service.scan_for_devices()
        
        return {
            "device_count": len(devices),
            "devices": [d.to_dict() for d in devices],
            "bluetooth_available": service.bluetooth_available,
            "scan_time": service.last_scan_time.isoformat() if service.last_scan_time else None
        }
        
    except Exception as e:
        logger.error(f"[API] Error scanning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_proximity_display_stats() -> Dict[str, Any]:
    """
    Get comprehensive service statistics
    
    Returns:
        Statistics from all proximity-display components
    """
    try:
        from proximity.proximity_display_bridge import get_proximity_display_bridge
        
        bridge = get_proximity_display_bridge()
        stats = bridge.get_bridge_stats()
        
        return stats
        
    except Exception as e:
        logger.error(f"[API] Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/displays")
async def get_displays_with_proximity() -> Dict[str, Any]:
    """
    Get all displays with proximity scoring
    
    Returns:
        Display list with proximity scores and context
    """
    try:
        from vision.multi_monitor_detector import MultiMonitorDetector
        
        detector = MultiMonitorDetector()
        summary = await detector.get_display_summary(include_proximity=True)
        
        return summary
        
    except Exception as e:
        logger.error(f"[API] Error getting displays: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint
    
    Returns:
        Service health status
    """
    try:
        from proximity.bluetooth_proximity_service import get_proximity_service
        
        service = get_proximity_service()
        bluetooth_ok = await service.check_bluetooth_availability()
        
        return {
            "status": "healthy" if bluetooth_ok else "degraded",
            "bluetooth_available": bluetooth_ok,
            "message": "Proximity-Display API operational"
        }
        
    except Exception as e:
        logger.error(f"[API] Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# ========================================
# PHASE 1.2C + 1.2D ENDPOINTS
# ========================================

@router.post("/connect")
async def connect_display(
    display_id: Optional[int] = Query(None, description="Specific display ID to connect"),
    mode: str = Query("extend", description="Connection mode: 'mirror' or 'extend'"),
    force: bool = Query(False, description="Force connection (bypass debouncing)")
) -> Dict[str, Any]:
    """
    PHASE 1.2D: Manually trigger display connection
    
    Connects to a display via AppleScript (backend only).
    If display_id not provided, connects to nearest display based on proximity.
    
    Args:
        display_id: Specific display ID (optional - uses nearest if not provided)
        mode: Connection mode ('mirror' or 'extend')
        force: Bypass debouncing and user overrides
        
    Returns:
        Connection result with success status
    """
    try:
        from proximity.auto_connection_manager import get_auto_connection_manager, DisplayMode
        from proximity.proximity_display_bridge import get_proximity_display_bridge
        
        manager = get_auto_connection_manager()
        
        # If no display_id provided, use nearest from proximity
        if display_id is None:
            bridge = get_proximity_display_bridge()
            decision = await bridge.make_connection_decision()
            
            if not decision:
                raise HTTPException(status_code=400, detail="No proximity data available. Please provide display_id.")
            
            display_id = decision.display_id
            display_name = decision.display_name
        else:
            display_name = f"Display {display_id}"
        
        # Execute connection
        display_mode = DisplayMode.MIRROR if mode.lower() == "mirror" else DisplayMode.EXTEND
        result = await manager._connect_display(display_id, display_name, display_mode)
        
        return result.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Error connecting display: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/disconnect")
async def disconnect_display(
    display_id: int = Query(..., description="Display ID to disconnect")
) -> Dict[str, Any]:
    """
    PHASE 1.2D: Manually disconnect a display
    
    Args:
        display_id: Display ID to disconnect
        
    Returns:
        Disconnect result
    """
    try:
        from proximity.auto_connection_manager import get_auto_connection_manager
        
        manager = get_auto_connection_manager()
        
        # Register user override (prevent auto-reconnect)
        manager.register_user_override(display_id)
        
        # Execute disconnect
        result = await manager._disconnect_display(display_id, f"Display {display_id}")
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"[API] Error disconnecting display: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auto-connect")
async def evaluate_auto_connect() -> Dict[str, Any]:
    """
    PHASE 1.2D: Evaluate and execute auto-connection based on proximity
    
    Makes connection decision based on current proximity context and
    executes connection if criteria met.
    
    Returns:
        Auto-connection evaluation and result
    """
    try:
        from proximity.proximity_display_bridge import get_proximity_display_bridge
        from proximity.auto_connection_manager import get_auto_connection_manager
        
        bridge = get_proximity_display_bridge()
        manager = get_auto_connection_manager()
        
        # Make connection decision
        decision = await bridge.make_connection_decision()
        
        if not decision:
            return {
                "evaluated": True,
                "action_taken": False,
                "reason": "No proximity data or nearest display available"
            }
        
        # Evaluate and execute
        result = await manager.evaluate_and_execute(decision)
        
        return {
            "evaluated": True,
            "decision": decision.to_dict(),
            "action_taken": result is not None,
            "connection_result": result.to_dict() if result else None
        }
        
    except Exception as e:
        logger.error(f"[API] Error in auto-connect: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/route-command")
async def route_command(
    command: str = Query(..., description="Command to route")
) -> Dict[str, Any]:
    """
    PHASE 1.2C: Route a command to display based on proximity
    
    Routes vision/display commands to the most appropriate display
    based on user proximity context.
    
    Args:
        command: User command string
        
    Returns:
        Routing result with target display and voice response
    """
    try:
        from proximity.proximity_command_router import get_proximity_command_router
        
        router = get_proximity_command_router()
        result = await router.route_command(command)
        
        return result
        
    except Exception as e:
        logger.error(f"[API] Error routing command: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/connection-stats")
async def get_connection_stats() -> Dict[str, Any]:
    """
    PHASE 1.2D: Get auto-connection manager statistics
    
    Returns:
        Connection statistics and history
    """
    try:
        from proximity.auto_connection_manager import get_auto_connection_manager
        
        manager = get_auto_connection_manager()
        return manager.get_manager_stats()
        
    except Exception as e:
        logger.error(f"[API] Error getting connection stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/routing-stats")
async def get_routing_stats() -> Dict[str, Any]:
    """
    PHASE 1.2C: Get command routing statistics
    
    Returns:
        Routing statistics
    """
    try:
        from proximity.proximity_command_router import get_proximity_command_router
        
        router = get_proximity_command_router()
        return router.get_router_stats()
        
    except Exception as e:
        logger.error(f"[API] Error getting routing stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voice-prompt-stats")
async def get_voice_prompt_stats() -> Dict[str, Any]:
    """
    PHASE 1.2C: Get voice prompt statistics
    
    Returns:
        Voice prompt statistics (yes/no acceptance rates)
    """
    try:
        from proximity.voice_prompt_manager import get_voice_prompt_manager
        
        manager = get_voice_prompt_manager()
        return manager.get_prompt_stats()
        
    except Exception as e:
        logger.error(f"[API] Error getting voice prompt stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/display-availability/{display_id}")
async def check_display_availability(display_id: int) -> Dict[str, Any]:
    """
    Check if a specific display is currently available (TV on/off detection)
    
    Args:
        display_id: Display ID to check
        
    Returns:
        Availability status
    """
    try:
        from proximity.display_availability_detector import get_availability_detector
        
        detector = get_availability_detector()
        status = await detector.get_availability_status(display_id)
        
        return status
        
    except Exception as e:
        logger.error(f"[API] Error checking display availability: {e}")
        raise HTTPException(status_code=500, detail=str(e))
