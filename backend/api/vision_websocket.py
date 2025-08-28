#!/usr/bin/env python3
"""
Vision WebSocket API for real-time workspace monitoring
Enables JARVIS to see and respond to screen events autonomously
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Set
from datetime import datetime
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse

from vision.workspace_analyzer import WorkspaceAnalyzer
from vision.window_detector import WindowDetector
from vision.jarvis_workspace_integration import JARVISWorkspaceIntelligence
from autonomy.autonomous_behaviors import AutonomousBehaviorManager
from autonomy.permission_manager import PermissionManager
from autonomy.context_engine import ContextEngine

logger = logging.getLogger(__name__)

router = APIRouter()

class VisionWebSocketManager:
    """Manages WebSocket connections for vision system"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.monitoring_active = False
        self.workspace_analyzer = WorkspaceAnalyzer()
        self.window_detector = WindowDetector()
        self.jarvis_intelligence = JARVISWorkspaceIntelligence()
        self.behavior_manager = AutonomousBehaviorManager()
        self.permission_manager = PermissionManager()
        self.context_engine = ContextEngine()
        self.last_workspace_state = None
        self.monitoring_interval = 2.0  # seconds
        
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Vision WebSocket connected. Total connections: {len(self.active_connections)}")
        
        # Send initial state
        await self.send_initial_state(websocket)
        
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.discard(websocket)
        logger.info(f"Vision WebSocket disconnected. Total connections: {len(self.active_connections)}")
        
        # Stop monitoring if no connections
        if not self.active_connections:
            self.monitoring_active = False
            
    async def send_initial_state(self, websocket: WebSocket):
        """Send initial workspace state to new connection"""
        try:
            # Get current workspace state
            windows = await self.window_detector.get_all_windows()
            workspace_analysis = await self.workspace_analyzer.analyze_workspace()
            
            initial_state = {
                "type": "initial_state",
                "timestamp": datetime.now().isoformat(),
                "workspace": {
                    "window_count": len(windows),
                    "focused_app": next((w.app_name for w in windows if w.is_focused), None),
                    "notifications": workspace_analysis.important_notifications,
                    "context": workspace_analysis.workspace_context
                },
                "autonomous_mode": await self.jarvis_intelligence.is_autonomous_enabled(),
                "monitoring_active": self.monitoring_active
            }
            
            await websocket.send_json(initial_state)
            
        except Exception as e:
            logger.error(f"Error sending initial state: {e}")
            
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(connection)
                
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
            
    async def start_monitoring(self):
        """Start continuous workspace monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        logger.info("Starting autonomous workspace monitoring")
        
        while self.monitoring_active and self.active_connections:
            try:
                # Get current workspace state
                windows = await self.window_detector.get_all_windows()
                workspace_analysis = await self.workspace_analyzer.analyze_workspace()
                
                # Build workspace state
                workspace_state = {
                    "windows": windows,
                    "analysis": workspace_analysis,
                    "window_count": len(windows),
                    "user_state": "available",  # This could be enhanced
                    "timestamp": datetime.now()
                }
                
                # Get autonomous actions from behavior manager
                autonomous_actions = []
                if await self.jarvis_intelligence.is_autonomous_enabled():
                    autonomous_actions = await self.behavior_manager.process_workspace_state(
                        workspace_state,
                        windows
                    )
                
                # Check for significant changes
                if self._has_significant_change(workspace_state):
                    # Prepare update message
                    update_message = {
                        "type": "workspace_update",
                        "timestamp": datetime.now().isoformat(),
                        "windows": [
                            {
                                "id": w.window_id,
                                "app": w.app_name,
                                "title": w.window_title,
                                "focused": w.is_focused,
                                "visible": w.is_visible
                            } for w in windows[:10]  # Limit to 10 most relevant
                        ],
                        "notifications": workspace_analysis.important_notifications,
                        "suggestions": workspace_analysis.suggestions,
                        "autonomous_actions": [
                            {
                                "type": action.action_type,
                                "target": action.target,
                                "priority": action.priority.name,
                                "reasoning": action.reasoning,
                                "confidence": action.confidence
                            } for action in autonomous_actions[:5]  # Limit to 5 actions
                        ],
                        "stats": {
                            "window_count": len(windows),
                            "notification_count": len(workspace_analysis.important_notifications),
                            "action_count": len(autonomous_actions)
                        }
                    }
                    
                    # Broadcast update
                    await self.broadcast(update_message)
                    
                    # Execute high-confidence autonomous actions
                    if autonomous_actions:
                        await self._execute_autonomous_actions(autonomous_actions)
                
                # Update last state
                self.last_workspace_state = workspace_state
                
                # Wait before next update
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
                
        logger.info("Stopped autonomous workspace monitoring")
        
    def _has_significant_change(self, current_state: Dict[str, Any]) -> bool:
        """Check if workspace has changed significantly"""
        if not self.last_workspace_state:
            return True
            
        # Check for new notifications
        if current_state["analysis"].important_notifications != \
           self.last_workspace_state.get("analysis", {}).get("important_notifications", []):
            return True
            
        # Check for window count change
        if abs(current_state["window_count"] - 
               self.last_workspace_state.get("window_count", 0)) > 2:
            return True
            
        # Check for focus change
        current_focused = next((w.app_name for w in current_state["windows"] if w.is_focused), None)
        last_focused = None
        if self.last_workspace_state and "windows" in self.last_workspace_state:
            last_focused = next((w.app_name for w in self.last_workspace_state["windows"] if w.is_focused), None)
            
        if current_focused != last_focused:
            return True
            
        return False
        
    async def _execute_autonomous_actions(self, actions):
        """Execute high-confidence autonomous actions"""
        for action in actions:
            try:
                # Check permission
                permission, confidence, reason = self.permission_manager.check_permission(action)
                
                if permission is True or (permission is None and action.confidence > 0.9):
                    # Log the action
                    logger.info(f"Executing autonomous action: {action.action_type} on {action.target}")
                    
                    # Broadcast action execution
                    await self.broadcast({
                        "type": "action_executed",
                        "timestamp": datetime.now().isoformat(),
                        "action": {
                            "type": action.action_type,
                            "target": action.target,
                            "reasoning": action.reasoning
                        }
                    })
                    
                    # Here you would actually execute the action
                    # For now, we'll just log it
                    
                    # Record decision if it was auto-approved
                    if permission is None:
                        self.permission_manager.record_decision(action, approved=True)
                        
            except Exception as e:
                logger.error(f"Error executing autonomous action: {e}")

# Global manager instance
vision_manager = VisionWebSocketManager()

@router.websocket("/ws/vision")
async def vision_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time vision updates"""
    await vision_manager.connect(websocket)
    
    try:
        # Start monitoring if not already active
        if not vision_manager.monitoring_active:
            asyncio.create_task(vision_manager.start_monitoring())
            
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_json()
                
                # Handle different message types
                if data.get("type") == "set_monitoring_interval":
                    vision_manager.monitoring_interval = data.get("interval", 2.0)
                    await websocket.send_json({
                        "type": "config_updated",
                        "monitoring_interval": vision_manager.monitoring_interval
                    })
                    
                elif data.get("type") == "request_workspace_analysis":
                    # Immediate workspace analysis
                    windows = await vision_manager.window_detector.get_all_windows()
                    analysis = await vision_manager.workspace_analyzer.analyze_workspace()
                    
                    await websocket.send_json({
                        "type": "workspace_analysis",
                        "timestamp": datetime.now().isoformat(),
                        "analysis": {
                            "focused_task": analysis.focused_task,
                            "context": analysis.workspace_context,
                            "notifications": analysis.important_notifications,
                            "suggestions": analysis.suggestions
                        }
                    })
                    
                elif data.get("type") == "execute_action":
                    # Manual action execution request
                    action_data = data.get("action", {})
                    # Process action execution
                    await websocket.send_json({
                        "type": "action_result",
                        "success": True,
                        "action": action_data
                    })
                    
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON received"
                })
                
    except WebSocketDisconnect:
        vision_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        vision_manager.disconnect(websocket)

@router.get("/vision/status")
async def get_vision_status():
    """Get current vision system status"""
    return JSONResponse({
        "monitoring_active": vision_manager.monitoring_active,
        "connected_clients": len(vision_manager.active_connections),
        "monitoring_interval": vision_manager.monitoring_interval,
        "autonomous_enabled": await vision_manager.jarvis_intelligence.is_autonomous_enabled()
    })

@router.post("/vision/start_monitoring")
async def start_monitoring():
    """Manually start vision monitoring"""
    if not vision_manager.monitoring_active:
        asyncio.create_task(vision_manager.start_monitoring())
        return JSONResponse({"status": "monitoring_started"})
    return JSONResponse({"status": "already_monitoring"})

@router.post("/vision/stop_monitoring")
async def stop_monitoring():
    """Manually stop vision monitoring"""
    vision_manager.monitoring_active = False
    return JSONResponse({"status": "monitoring_stopped"})