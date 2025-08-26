#!/usr/bin/env python3
"""
WebSocket Discovery API
Provides endpoint discovery for the TypeScript dynamic client
"""

from fastapi import APIRouter
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/websocket", tags=["websocket"])


@router.get("/endpoints")
async def discover_websocket_endpoints() -> Dict[str, List[Dict[str, Any]]]:
    """
    Dynamically discover available WebSocket endpoints
    Used by the TypeScript client for auto-discovery
    """
    endpoints = []
    
    # Vision WebSocket
    endpoints.append({
        "path": "/vision/ws/vision",
        "capabilities": ["vision", "monitoring", "analysis", "claude_vision"],
        "priority": 10,
        "description": "Enhanced vision system with Claude AI integration",
        "features": [
            "continuous_monitoring",
            "multi_window_detection", 
            "notification_detection",
            "autonomous_actions"
        ]
    })
    
    # Voice WebSocket (if available)
    try:
        from api.jarvis_voice_api import jarvis_api
        if jarvis_api and jarvis_api.jarvis_available:
            endpoints.append({
                "path": "/voice/ws",
                "capabilities": ["voice", "speech", "jarvis", "commands"],
                "priority": 8,
                "description": "JARVIS voice system with ML routing"
            })
    except:
        pass
    
    # Automation WebSocket
    endpoints.append({
        "path": "/automation/ws",
        "capabilities": ["automation", "tasks", "scheduling"],
        "priority": 7,
        "description": "Task automation and scheduling"
    })
    
    # General WebSocket
    endpoints.append({
        "path": "/ws",
        "capabilities": ["general", "chat", "notifications"],
        "priority": 5,
        "description": "General purpose WebSocket"
    })
    
    return {
        "endpoints": endpoints,
        "discovery_method": "api",
        "dynamic": True
    }


@router.post("/register")
async def register_websocket_endpoint(
    path: str,
    capabilities: List[str],
    priority: int = 5,
    description: str = ""
) -> Dict[str, str]:
    """
    Register a new WebSocket endpoint dynamically
    This allows plugins and extensions to register their own endpoints
    """
    # In a real implementation, this would store in a registry
    logger.info(f"Registered WebSocket endpoint: {path} with capabilities: {capabilities}")
    
    return {
        "status": "registered",
        "path": path,
        "message": "Endpoint registered for discovery"
    }