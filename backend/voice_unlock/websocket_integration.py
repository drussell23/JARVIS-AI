#!/usr/bin/env python3
"""
Voice Unlock WebSocket Integration for JARVIS
============================================

This module integrates the Voice Unlock system with JARVIS's main WebSocket server.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)


class VoiceUnlockWebSocketHandler:
    """Handles Voice Unlock WebSocket messages within JARVIS"""
    
    def __init__(self):
        self.clients: Dict[str, WebSocketServerProtocol] = {}
        self.objc_daemon_connection: Optional[WebSocketServerProtocol] = None
        
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        
        if path == "/voice-unlock":
            # This is the Objective-C daemon connecting
            self.objc_daemon_connection = websocket
            logger.info(f"Voice Unlock daemon connected from {client_id}")
            
            try:
                await self._handle_daemon_messages(websocket)
            finally:
                self.objc_daemon_connection = None
                logger.info("Voice Unlock daemon disconnected")
                
        else:
            # Regular client connection
            self.clients[client_id] = websocket
            logger.info(f"Client connected: {client_id}")
            
            try:
                await self._handle_client_messages(websocket, client_id)
            finally:
                del self.clients[client_id]
                logger.info(f"Client disconnected: {client_id}")
    
    async def _handle_daemon_messages(self, websocket: WebSocketServerProtocol):
        """Handle messages from the Objective-C daemon"""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_daemon_message(data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from daemon: {message}")
                except Exception as e:
                    logger.error(f"Error processing daemon message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            pass
    
    async def _handle_client_messages(self, websocket: WebSocketServerProtocol, client_id: str):
        """Handle messages from JARVIS clients"""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data.get('type') == 'voice_unlock_command':
                        await self._forward_to_daemon(data)
                        
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from client {client_id}: {message}")
                except Exception as e:
                    logger.error(f"Error processing client message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            pass
    
    async def _process_daemon_message(self, data: Dict[str, Any]):
        """Process messages from the daemon"""
        msg_type = data.get('type')
        
        if msg_type == 'status':
            # Broadcast status to all clients
            await self._broadcast_to_clients({
                'type': 'voice_unlock_status',
                'data': data.get('status', {})
            })
            
        elif msg_type == 'authentication':
            # Authentication result
            result = data.get('payload', {})
            if result.get('authenticated'):
                await self._broadcast_to_clients({
                    'type': 'voice_unlock_success',
                    'user': result.get('user_id'),
                    'confidence': result.get('confidence')
                })
            else:
                await self._broadcast_to_clients({
                    'type': 'voice_unlock_failed',
                    'reason': result.get('reason', 'Authentication failed')
                })
                
        elif msg_type == 'screen_state':
            # Screen state update
            await self._broadcast_to_clients({
                'type': 'screen_state_changed',
                'locked': data.get('payload', {}).get('locked', False)
            })
    
    async def _forward_to_daemon(self, data: Dict[str, Any]):
        """Forward commands to the Objective-C daemon"""
        if self.objc_daemon_connection:
            try:
                await self.objc_daemon_connection.send(json.dumps(data))
            except Exception as e:
                logger.error(f"Failed to forward to daemon: {e}")
        else:
            logger.warning("No daemon connection available")
    
    async def _broadcast_to_clients(self, data: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        message = json.dumps(data)
        
        # Send to all clients
        disconnected = []
        for client_id, websocket in self.clients.items():
            try:
                await websocket.send(message)
            except Exception as e:
                logger.error(f"Failed to send to client {client_id}: {e}")
                disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            del self.clients[client_id]


def integrate_with_jarvis_websocket(websocket_handler):
    """
    Integrate Voice Unlock with JARVIS's main WebSocket handler
    
    This function should be called from the main JARVIS WebSocket server
    to add Voice Unlock functionality.
    """
    voice_handler = VoiceUnlockWebSocketHandler()
    
    # Add Voice Unlock routes to the main handler
    async def voice_unlock_handler(websocket, path):
        if path.startswith('/voice-unlock'):
            await voice_handler.handle_connection(websocket, path)
        else:
            # Pass through to main handler
            await websocket_handler(websocket, path)
    
    return voice_unlock_handler


async def start_standalone_server(host: str = 'localhost', port: int = 8765):
    """Start a standalone Voice Unlock WebSocket server for testing"""
    handler = VoiceUnlockWebSocketHandler()
    
    async def handle_connection(websocket, path):
        await handler.handle_connection(websocket, path)
    
    logger.info(f"Starting Voice Unlock WebSocket server on {host}:{port}")
    await websockets.serve(handle_connection, host, port)


if __name__ == "__main__":
    # Run standalone server for testing
    logging.basicConfig(level=logging.INFO)
    asyncio.run(start_standalone_server())