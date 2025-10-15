"""
Display Monitor API
===================

Simple REST API for display monitoring (no proximity detection).

Endpoints:
- POST /api/display-monitor/register - Register a display to monitor
- GET /api/display-monitor/available - Get available displays
- POST /api/display-monitor/connect - Connect to a display
- GET /api/display-monitor/status - Get monitoring status
- POST /api/display-monitor/start - Start monitoring
- POST /api/display-monitor/stop - Stop monitoring

Author: Derek Russell
Date: 2025-10-15
"""

import logging
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/display-monitor", tags=["display-monitor"])


class RegisterDisplayRequest(BaseModel):
    """Request to register a display for monitoring"""
    display_name: str
    auto_prompt: bool = True
    default_mode: str = "extend"


@router.post("/register")
async def register_display(request: RegisterDisplayRequest) -> Dict[str, Any]:
    """
    Register a display to monitor
    
    Args:
        display_name: Display name (e.g., "Living Room TV")
        auto_prompt: Automatically prompt when available
        default_mode: "extend" or "mirror"
        
    Returns:
        Registration result
    """
    try:
        from display import get_display_monitor
        
        monitor = get_display_monitor()
        monitor.register_display(
            display_name=request.display_name,
            auto_prompt=request.auto_prompt,
            default_mode=request.default_mode
        )
        
        return {
            "success": True,
            "message": f"Registered {request.display_name} for monitoring"
        }
        
    except Exception as e:
        logger.error(f"[API] Error registering display: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available")
async def get_available_displays() -> Dict[str, Any]:
    """
    Get currently available displays
    
    Returns:
        List of available displays
    """
    try:
        from display import get_display_monitor
        
        monitor = get_display_monitor()
        
        return {
            "available_displays": list(monitor.available_displays),
            "monitored_displays": list(monitor.monitored_displays.keys())
        }
        
    except Exception as e:
        logger.error(f"[API] Error getting available displays: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/connect")
async def connect_to_display(
    display_name: str,
    mode: str = "extend"
) -> Dict[str, Any]:
    """
    Manually connect to a display
    
    Args:
        display_name: Display name
        mode: "extend" or "mirror"
        
    Returns:
        Connection result
    """
    try:
        from display import get_display_monitor
        
        monitor = get_display_monitor()
        result = await monitor._connect_to_display(display_name, mode)
        
        return result
        
    except Exception as e:
        logger.error(f"[API] Error connecting to display: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_status() -> Dict[str, Any]:
    """
    Get monitoring status
    
    Returns:
        Current status and statistics
    """
    try:
        from display import get_display_monitor
        
        monitor = get_display_monitor()
        stats = monitor.get_stats()
        pending = monitor.get_pending_prompt()
        
        return {
            "stats": stats,
            "pending_prompt": pending
        }
        
    except Exception as e:
        logger.error(f"[API] Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_monitoring() -> Dict[str, Any]:
    """
    Start display monitoring
    
    Returns:
        Start result
    """
    try:
        from display import get_display_monitor
        
        monitor = get_display_monitor()
        await monitor.start_monitoring()
        
        return {
            "success": True,
            "message": "Display monitoring started"
        }
        
    except Exception as e:
        logger.error(f"[API] Error starting monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_monitoring() -> Dict[str, Any]:
    """
    Stop display monitoring
    
    Returns:
        Stop result
    """
    try:
        from display import get_display_monitor
        
        monitor = get_display_monitor()
        await monitor.stop_monitoring()
        
        return {
            "success": True,
            "message": "Display monitoring stopped"
        }
        
    except Exception as e:
        logger.error(f"[API] Error stopping monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))
