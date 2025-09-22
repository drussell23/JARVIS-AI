#!/usr/bin/env python3
"""
WebSocket Server Bridge for JARVIS Voice Unlock Daemon
======================================================

Provides WebSocket API for the Objective-C daemon.
"""

import asyncio
import websockets
import json
import logging
import subprocess
import os
from datetime import datetime
from typing import Optional, Dict, Any, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceUnlockWebSocketServer:
    """WebSocket server that interfaces with the Objective-C daemon"""
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.daemon_process = None
        self.enrolled_users_file = os.path.expanduser("~/.jarvis/voice_unlock/enrolled_users.json")
        
    async def register_client(self, websocket):
        """Register a new WebSocket client"""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
    async def unregister_client(self, websocket):
        """Unregister a WebSocket client"""
        self.clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
        
    async def handle_message(self, websocket, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming WebSocket message"""
        msg_type = message.get("type")
        command = message.get("command")
        parameters = message.get("parameters", {})
        
        logger.info(f"Received: type={msg_type}, command={command}")
        
        if msg_type == "command":
            if command == "handshake":
                return {
                    "type": "handshake",
                    "success": True,
                    "version": "1.0",
                    "daemon": "JARVIS Voice Unlock"
                }
                
            elif command == "get_status":
                # Get daemon status
                is_running = self.check_daemon_running()
                enrolled_users = self.get_enrolled_users()
                
                return {
                    "type": "status",
                    "success": True,
                    "status": {
                        "isMonitoring": is_running,
                        "isScreenLocked": False,  # Would need actual check
                        "enrolledUser": enrolled_users[0] if enrolled_users else "none",
                        "failedAttempts": 0,
                        "state": 1 if is_running else 0
                    }
                }
                
            elif command == "start_monitoring":
                # Start daemon if not running
                if not self.check_daemon_running():
                    success = self.start_daemon()
                    return {
                        "type": "command_response",
                        "command": command,
                        "success": success,
                        "message": "Monitoring started" if success else "Failed to start monitoring"
                    }
                else:
                    return {
                        "type": "command_response",
                        "command": command,
                        "success": True,
                        "message": "Monitoring already active"
                    }
                    
            elif command == "stop_monitoring":
                # Stop daemon
                success = self.stop_daemon()
                return {
                    "type": "command_response",
                    "command": command,
                    "success": success,
                    "message": "Monitoring stopped" if success else "Failed to stop monitoring"
                }
                
            else:
                return {
                    "type": "error",
                    "message": f"Unknown command: {command}",
                    "success": False
                }
        
        return {
            "type": "error",
            "message": f"Unknown message type: {msg_type}",
            "success": False
        }
        
    def check_daemon_running(self) -> bool:
        """Check if the Voice Unlock daemon is running"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "JARVISVoiceUnlockDaemon"],
                capture_output=True
            )
            return result.returncode == 0
        except:
            return False
            
    def get_enrolled_users(self) -> list:
        """Get list of enrolled user names"""
        try:
            if os.path.exists(self.enrolled_users_file):
                with open(self.enrolled_users_file) as f:
                    users = json.load(f)
                    return [user.get("name", "unknown") for user in users.values()]
        except:
            pass
        return []
        
    def start_daemon(self) -> bool:
        """Start the Voice Unlock daemon"""
        try:
            daemon_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "bin/JARVISVoiceUnlockDaemon"
            )
            
            if not os.path.exists(daemon_path):
                logger.error(f"Daemon not found at {daemon_path}")
                return False
                
            self.daemon_process = subprocess.Popen(
                [daemon_path, "--debug"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Give it time to start
            asyncio.get_event_loop().call_later(1.0, self.check_daemon_started)
            
            return True
        except Exception as e:
            logger.error(f"Failed to start daemon: {e}")
            return False
            
    def check_daemon_started(self):
        """Check if daemon started successfully"""
        if self.daemon_process and self.daemon_process.poll() is None:
            logger.info("Daemon started successfully")
        else:
            logger.error("Daemon failed to start")
            
    def stop_daemon(self) -> bool:
        """Stop the Voice Unlock daemon"""
        try:
            subprocess.run(["pkill", "-f", "JARVISVoiceUnlockDaemon"])
            return True
        except:
            return False
            
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections"""
        logger.info(f"New connection on path: {path}")
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self.handle_message(websocket, data)
                    await websocket.send(json.dumps(response))
                    
                except json.JSONDecodeError:
                    error_response = {
                        "type": "error",
                        "message": "Invalid JSON",
                        "success": False
                    }
                    await websocket.send(json.dumps(error_response))
                    
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    error_response = {
                        "type": "error",
                        "message": str(e),
                        "success": False
                    }
                    await websocket.send(json.dumps(error_response))
                    
        except websockets.exceptions.ConnectionClosed:
            pass
            
        finally:
            await self.unregister_client(websocket)
            
    async def start(self):
        """Start the WebSocket server"""
        logger.info(f"Starting Voice Unlock WebSocket server on port {self.port}")
        
        async with websockets.serve(self.websocket_handler, "localhost", self.port):
            logger.info(f"Server listening on ws://localhost:{self.port}")
            await asyncio.Future()  # Run forever


def main():
    """Main entry point"""
    server = VoiceUnlockWebSocketServer(port=8765)
    asyncio.run(server.start())


if __name__ == "__main__":
    main()