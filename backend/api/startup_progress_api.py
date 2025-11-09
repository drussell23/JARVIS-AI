#!/usr/bin/env python3
"""
Startup Progress WebSocket API
Real-time broadcast of JARVIS startup/restart progress to loading page
"""

import asyncio
import logging
from typing import Dict, List, Set
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


class StartupProgressManager:
    """Manages WebSocket connections for startup progress updates"""

    def __init__(self):
        self.connections: Set[WebSocket] = set()
        self.current_status: Dict = {
            "stage": "idle",
            "message": "System idle",
            "progress": 0,
            "timestamp": datetime.now().isoformat(),
        }
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        """Add a new WebSocket connection"""
        await websocket.accept()
        async with self._lock:
            self.connections.add(websocket)
        logger.info(f"Startup progress client connected. Total: {len(self.connections)}")

        # Send current status immediately
        try:
            await websocket.send_json(self.current_status)
        except Exception as e:
            logger.error(f"Failed to send initial status: {e}")

    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        async with self._lock:
            self.connections.discard(websocket)
        logger.info(f"Startup progress client disconnected. Remaining: {len(self.connections)}")

    async def broadcast_progress(
        self,
        stage: str,
        message: str,
        progress: int,
        details: Dict = None,
        metadata: Dict = None,
    ):
        """
        Broadcast progress update to all connected clients

        Args:
            stage: Current stage (e.g., "detecting", "killing", "starting")
            message: Human-readable message
            progress: Progress percentage (0-100)
            details: Optional additional data
            metadata: Optional stage metadata (icon, label, sublabel for dynamic UI)
        """
        status = {
            "stage": stage,
            "message": message,
            "progress": progress,
            "timestamp": datetime.now().isoformat(),
        }

        if details:
            status["details"] = details

        if metadata:
            status["metadata"] = metadata

        self.current_status = status

        # Broadcast to all connected clients
        disconnected = []
        async with self._lock:
            for websocket in self.connections:
                try:
                    await websocket.send_json(status)
                except Exception as e:
                    logger.warning(f"Failed to send to client: {e}")
                    disconnected.append(websocket)

        # Clean up disconnected clients
        if disconnected:
            async with self._lock:
                for ws in disconnected:
                    self.connections.discard(ws)

    async def broadcast_complete(self, success: bool = True, redirect_url: str = None):
        """Broadcast completion status"""
        status = {
            "stage": "complete" if success else "failed",
            "message": "System ready!" if success else "Startup failed",
            "progress": 100 if success else 0,
            "timestamp": datetime.now().isoformat(),
            "success": success,
        }

        if redirect_url:
            status["redirect_url"] = redirect_url

        self.current_status = status

        # Broadcast to all clients
        async with self._lock:
            for websocket in list(self.connections):
                try:
                    await websocket.send_json(status)
                except:
                    pass


# Global instance
startup_progress_manager = StartupProgressManager()


@router.websocket("/ws/startup-progress")
async def startup_progress_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time startup progress updates"""
    await startup_progress_manager.connect(websocket)

    try:
        # Keep connection alive and listen for pings
        while True:
            try:
                data = await websocket.receive_text()
                # Client can send "ping" to keep connection alive
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                break
    except Exception as e:
        logger.error(f"Startup progress WebSocket error: {e}")
    finally:
        await startup_progress_manager.disconnect(websocket)


# HTTP endpoint for polling fallback
@router.get("/api/startup-progress")
async def get_startup_progress():
    """HTTP endpoint for polling-based progress updates (fallback for WebSocket)"""
    return startup_progress_manager.current_status


# Convenience function for use in start_system.py
def get_startup_progress_manager() -> StartupProgressManager:
    """Get the global startup progress manager instance"""
    return startup_progress_manager
