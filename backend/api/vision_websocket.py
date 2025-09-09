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
import os

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse

# Import the actual vision analyzer we have
try:
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    VISION_ANALYZER_AVAILABLE = True
except ImportError:
    VISION_ANALYZER_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter()

class VisionWebSocketManager:
    """Manages WebSocket connections for vision system"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.monitoring_active = False
        self.monitoring_task = None
        
        # Don't create a new analyzer here - will get from app state
        self._vision_analyzer = None
        self.monitoring_interval = 2.0  # seconds
        
    @property
    def vision_analyzer(self):
        """Get vision analyzer from app state if not already set"""
        if self._vision_analyzer is None:
            try:
                from main import app
                if hasattr(app.state, 'vision_analyzer'):
                    self._vision_analyzer = app.state.vision_analyzer
                    logger.info("Using vision analyzer from app state")
            except Exception as e:
                logger.error(f"Failed to get vision analyzer from app state: {e}")
                
                # Fallback: create our own if needed
                if VISION_ANALYZER_AVAILABLE:
                    api_key = os.getenv('ANTHROPIC_API_KEY')
                    if api_key:
                        self._vision_analyzer = ClaudeVisionAnalyzer(api_key, enable_realtime=True)
                        logger.info("Created fallback vision analyzer for WebSocket")
        return self._vision_analyzer
    
    @vision_analyzer.setter
    def vision_analyzer(self, value):
        """Allow setting vision analyzer"""
        self._vision_analyzer = value
        
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
        if len(self.active_connections) == 0:
            asyncio.create_task(self.stop_monitoring())
            
    async def send_initial_state(self, websocket: WebSocket):
        """Send initial state to newly connected client"""
        try:
            initial_state = {
                "type": "initial_state",
                "monitoring_active": self.monitoring_active,
                "vision_available": self.vision_analyzer is not None,
                "workspace": {
                    'window_count': 0,
                    'focused_app': None,
                    'notifications': [],
                    'notification_details': {
                        'badges': 0,
                        'messages': 0,
                        'meetings': 0,
                        'alerts': 0
                    }
                },
                "timestamp": datetime.now().isoformat()
            }
            
            await websocket.send_json(initial_state)
        except Exception as e:
            logger.error(f"Error sending initial state: {e}")
            
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
            
    async def start_monitoring(self):
        """Start continuous workspace monitoring"""
        if self.monitoring_active:
            logger.info("Monitoring already active")
            return
            
        self.monitoring_active = True
        logger.info("Starting vision monitoring")
        
        # Start video streaming if vision analyzer is available
        if self.vision_analyzer:
            result = await self.vision_analyzer.start_video_streaming()
            if result.get('success'):
                logger.info("Video streaming started successfully")
                await self.broadcast_message({
                    'type': 'monitoring_started',
                    'capture_method': result.get('metrics', {}).get('capture_method', 'unknown'),
                    'message': 'Screen monitoring started - macOS recording indicator should be visible'
                })
            else:
                logger.error(f"Failed to start video streaming: {result.get('error')}")
                await self.broadcast_message({
                    'type': 'error',
                    'message': f"Failed to start monitoring: {result.get('error')}"
                })
                self.monitoring_active = False
                return
        
        # Start monitoring loop
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active and self.active_connections:
            try:
                if self.vision_analyzer:
                    # Analyze video stream for changes
                    analysis = await self.vision_analyzer.analyze_video_stream(
                        "Monitor the screen and describe any changes or important elements",
                        duration_seconds=self.monitoring_interval
                    )
                    
                    if analysis.get('success'):
                        # Parse analysis for workspace info
                        workspace_data = {
                            'window_count': 0,  # Will be updated from analysis
                            'focused_app': None,
                            'notifications': [],
                            'notification_details': {
                                'badges': 0,
                                'messages': 0,
                                'meetings': 0,
                                'alerts': 0
                            }
                        }
                        
                        # Extract workspace info from analysis text if available
                        analysis_text = analysis.get('analysis', '')
                        if 'window' in analysis_text.lower():
                            # Simple extraction - can be improved
                            import re
                            window_match = re.search(r'(\d+)\s*window', analysis_text.lower())
                            if window_match:
                                workspace_data['window_count'] = int(window_match.group(1))
                        
                        # Send update to clients
                        await self.broadcast_message({
                            'type': 'workspace_update',
                            'workspace': workspace_data,
                            'analysis': analysis_text,
                            'frames_analyzed': analysis.get('frames_analyzed', 0),
                            'timestamp': datetime.now().isoformat()
                        })
                    else:
                        logger.error(f"Failed to analyze video stream: {analysis.get('error')}")
                
                # Wait before next analysis
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
                
    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        if not self.monitoring_active:
            return
            
        logger.info("Stopping vision monitoring")
        self.monitoring_active = False
        
        # Stop video streaming
        if self.vision_analyzer:
            await self.vision_analyzer.stop_video_streaming()
            
        # Cancel monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None
            
        await self.broadcast_message({
            'type': 'monitoring_stopped',
            'message': 'Screen monitoring stopped'
        })

# Create global manager instance
vision_manager = VisionWebSocketManager()

@router.websocket("/ws/vision")
async def vision_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time vision updates"""
    await vision_manager.connect(websocket)
    
    try:
        # Start monitoring when client connects
        asyncio.create_task(vision_manager.start_monitoring())
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_json()
                
                # Handle different message types
                if data.get("type") == "set_monitoring_interval":
                    vision_manager.monitoring_interval = data.get("interval", 2.0)
                    await websocket.send_json({"type": "config_updated", "interval": vision_manager.monitoring_interval})
                    
                elif data.get("type") == "request_workspace_analysis":
                    # Trigger immediate analysis
                    if vision_manager.vision_analyzer:
                        screenshot = await vision_manager.vision_analyzer.take_screenshot()
                        if screenshot:
                            analysis = await vision_manager.vision_analyzer.analyze_image(
                                screenshot,
                                "Describe what you see on the screen in detail"
                            )
                            await websocket.send_json({
                                "type": "workspace_analysis",
                                "analysis": analysis,
                                "timestamp": datetime.now().isoformat()
                            })
                            
                elif data.get("type") == "stop_monitoring":
                    await vision_manager.stop_monitoring()
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                
    except WebSocketDisconnect:
        pass
    finally:
        vision_manager.disconnect(websocket)
        
@router.post("/start_monitoring")
async def start_monitoring():
    """Manually start vision monitoring"""
    if not vision_manager.monitoring_active:
        asyncio.create_task(vision_manager.start_monitoring())
        return JSONResponse({"status": "monitoring_started"})
    return JSONResponse({"status": "already_monitoring"})

@router.post("/stop_monitoring")
async def stop_monitoring():
    """Manually stop vision monitoring"""
    await vision_manager.stop_monitoring()
    return JSONResponse({"status": "monitoring_stopped"})

@router.get("/monitoring_status")
async def get_monitoring_status():
    """Get current monitoring status"""
    return JSONResponse({
        "monitoring_active": vision_manager.monitoring_active,
        "connections": len(vision_manager.active_connections),
        "vision_available": vision_manager.vision_analyzer is not None
    })