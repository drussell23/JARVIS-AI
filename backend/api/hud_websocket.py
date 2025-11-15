"""
JARVIS macOS Native HUD WebSocket Endpoint
Real-time bidirectional communication between Python backend and Swift HUD
Dynamic state synchronization with zero hardcoding
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set, Optional
import asyncio
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

# Active HUD connections
active_hud_connections: Set[WebSocket] = set()

# Shared state for HUD clients
hud_state = {
    "status": "offline",
    "message": "System initializing...",
    "transcript": [],
    "reactor_state": "idle",
    "last_update": None
}


class HUDConnectionManager:
    """
    Manages WebSocket connections for macOS HUD clients
    Features:
    - Multi-client support
    - State synchronization
    - Broadcast messaging
    - Health monitoring
    """

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.client_info: Dict[WebSocket, dict] = {}

    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None):
        """Accept and register new HUD connection"""
        await websocket.accept()
        self.active_connections.add(websocket)

        client_info = {
            "id": client_id or f"hud-{len(self.active_connections)}",
            "connected_at": datetime.now().isoformat(),
            "client_type": "macos-hud"
        }
        self.client_info[websocket] = client_info

        logger.info(f"✓ HUD client connected: {client_info['id']}")
        logger.info(f"  Total HUD clients: {len(self.active_connections)}")

        # Send current state to new client
        await self.send_state(websocket, hud_state)

    async def disconnect(self, websocket: WebSocket):
        """Remove disconnected client"""
        if websocket in self.active_connections:
            client_info = self.client_info.get(websocket, {})
            self.active_connections.remove(websocket)
            if websocket in self.client_info:
                del self.client_info[websocket]

            logger.info(f"✗ HUD client disconnected: {client_info.get('id', 'unknown')}")
            logger.info(f"  Remaining clients: {len(self.active_connections)}")

    async def send_state(self, websocket: WebSocket, state: dict):
        """Send state update to specific client"""
        try:
            message = {
                "type": "state",
                "data": state,
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending state to client: {e}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected HUD clients"""
        if not self.active_connections:
            return

        message["timestamp"] = datetime.now().isoformat()

        # Send to all clients concurrently
        disconnected_clients = []

        for websocket in self.active_connections:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected_clients.append(websocket)

        # Clean up disconnected clients
        for websocket in disconnected_clients:
            await self.disconnect(websocket)

    async def update_state(self, updates: dict):
        """Update global HUD state and broadcast to clients"""
        hud_state.update(updates)
        hud_state["last_update"] = datetime.now().isoformat()

        await self.broadcast({
            "type": "state_update",
            "updates": updates
        })

    async def send_transcript(self, speaker: str, text: str):
        """Send transcript message to all HUD clients"""
        transcript_entry = {
            "speaker": speaker,
            "text": text,
            "timestamp": datetime.now().isoformat()
        }

        # Add to state
        hud_state["transcript"].append(transcript_entry)

        # Keep only last 10 messages
        if len(hud_state["transcript"]) > 10:
            hud_state["transcript"] = hud_state["transcript"][-10:]

        # Broadcast
        await self.broadcast({
            "type": "transcript",
            "data": transcript_entry
        })

    async def set_reactor_state(self, state: str):
        """Update arc reactor state (idle, listening, processing, speaking)"""
        hud_state["reactor_state"] = state
        await self.broadcast({
            "type": "reactor_state",
            "state": state
        })

    async def set_status(self, status: str, message: str = ""):
        """Update system status"""
        await self.update_state({
            "status": status,
            "message": message
        })


# Global manager instance
hud_manager = HUDConnectionManager()


@router.websocket("/ws/hud")
async def hud_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for macOS HUD clients
    Handles bidirectional communication with real-time state sync
    """
    client_id = None

    try:
        await hud_manager.connect(websocket)

        while True:
            # Receive messages from HUD
            data = await websocket.receive_json()

            message_type = data.get("type")

            if message_type == "connect":
                # Initial connection message
                client_id = data.get("client_id")
                version = data.get("version", "unknown")
                logger.info(f"HUD client info: version={version}")

                # Send welcome message
                await websocket.send_json({
                    "type": "welcome",
                    "message": "Connected to JARVIS backend",
                    "server_version": "1.0.0"
                })

            elif message_type == "ping":
                # Health check
                await websocket.send_json({"type": "pong"})

            elif message_type == "command":
                # Command from HUD
                command_text = data.get("text", "")
                logger.info(f"📱 Command from HUD: {command_text}")

                # Add to transcript
                await hud_manager.send_transcript("USER", command_text)

                # Process command (integrate with existing JARVIS command system)
                # This would connect to your existing voice/command processing
                # For now, echo back
                await hud_manager.send_transcript("JARVIS", f"Received: {command_text}")

            elif message_type == "request_state":
                # Client requesting current state
                await hud_manager.send_state(websocket, hud_state)

            else:
                logger.warning(f"Unknown message type from HUD: {message_type}")

    except WebSocketDisconnect:
        logger.info("HUD client disconnected normally")
    except Exception as e:
        logger.error(f"HUD WebSocket error: {e}")
    finally:
        await hud_manager.disconnect(websocket)


# Helper functions for other parts of the system to update HUD

async def update_hud_status(status: str, message: str = ""):
    """Update HUD status from anywhere in the system"""
    await hud_manager.set_status(status, message)


async def send_hud_transcript(speaker: str, text: str):
    """Send transcript to HUD from anywhere in the system"""
    await hud_manager.send_transcript(speaker, text)


async def set_hud_reactor_state(state: str):
    """Update arc reactor state from anywhere in the system"""
    await hud_manager.set_reactor_state(state)


async def broadcast_to_hud(message: dict):
    """Broadcast custom message to all HUD clients"""
    await hud_manager.broadcast(message)


async def send_loading_progress(progress: int, message: str):
    """
    Send loading progress update to HUD during system startup
    Used by start_system.py to show real-time boot progress

    Args:
        progress: Progress percentage (0-100)
        message: Status message describing current step
    """
    await hud_manager.broadcast({
        "type": "loading_progress",
        "progress": progress,
        "message": message
    })


async def send_loading_complete(success: bool = True):
    """
    Send completion signal to HUD - triggers transition to main HUD
    Called by start_system.py when backend is fully ready

    Args:
        success: Whether startup completed successfully
    """
    await hud_manager.broadcast({
        "type": "loading_complete",
        "success": success,
        "progress": 100,
        "message": "JARVIS is ready!" if success else "Startup failed"
    })
