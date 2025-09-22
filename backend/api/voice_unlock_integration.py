"""
Voice Unlock Integration for JARVIS API
======================================

This module integrates the Voice Unlock daemon with JARVIS's main API.
"""

import asyncio
import json
import logging
import websockets
from typing import Optional

logger = logging.getLogger(__name__)


class VoiceUnlockDaemonConnector:
    """Connects to the Voice Unlock Objective-C daemon via WebSocket"""
    
    def __init__(self, daemon_host: str = "localhost", daemon_port: int = 8765):
        self.daemon_host = daemon_host
        self.daemon_port = daemon_port
        self.daemon_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        
    async def connect(self):
        """Connect to the Voice Unlock daemon"""
        try:
            uri = f"ws://{self.daemon_host}:{self.daemon_port}/voice-unlock"
            self.daemon_ws = await websockets.connect(uri)
            self.connected = True
            logger.info(f"Connected to Voice Unlock daemon at {uri}")
            
            # Send handshake
            await self.send_command("handshake", {
                "client": "jarvis-api",
                "version": "1.0"
            })
            
        except Exception as e:
            logger.error(f"Failed to connect to Voice Unlock daemon: {e}")
            self.connected = False
            
    async def disconnect(self):
        """Disconnect from the daemon"""
        if self.daemon_ws:
            await self.daemon_ws.close()
            self.connected = False
            
    async def send_command(self, command: str, parameters: dict = None):
        """Send command to the daemon"""
        if not self.connected or not self.daemon_ws:
            logger.error("Not connected to Voice Unlock daemon")
            return None
            
        message = {
            "type": "command",
            "command": command,
            "parameters": parameters or {}
        }
        
        try:
            await self.daemon_ws.send(json.dumps(message))
            
            # Wait for response
            response = await self.daemon_ws.recv()
            return json.loads(response)
            
        except Exception as e:
            logger.error(f"Failed to send command to daemon: {e}")
            return None
            
    async def get_status(self):
        """Get Voice Unlock status"""
        return await self.send_command("get_status")
        
    async def enable_monitoring(self):
        """Enable voice monitoring"""
        return await self.send_command("start_monitoring")
        
    async def disable_monitoring(self):
        """Disable voice monitoring"""
        return await self.send_command("stop_monitoring")


# Global instance
voice_unlock_connector = None


async def initialize_voice_unlock():
    """Initialize Voice Unlock connection"""
    global voice_unlock_connector
    
    voice_unlock_connector = VoiceUnlockDaemonConnector()
    await voice_unlock_connector.connect()
    
    if voice_unlock_connector.connected:
        status = await voice_unlock_connector.get_status()
        logger.info(f"Voice Unlock daemon status: {status}")
        return True
    
    return False


async def handle_voice_unlock_in_jarvis(command: str) -> dict:
    """
    Handle voice unlock commands from JARVIS
    
    This should be called from unified_command_processor when
    voice unlock commands are detected.
    """
    command_lower = command.lower()
    
    # Ensure we're connected
    if not voice_unlock_connector or not voice_unlock_connector.connected:
        await initialize_voice_unlock()
    
    # Map JARVIS commands to daemon commands
    if "enable voice unlock" in command_lower:
        result = await voice_unlock_connector.enable_monitoring()
        return {
            'type': 'voice_unlock',
            'action': 'enabled',
            'message': 'Voice unlock monitoring is now active, Sir. Your Mac will respond to your voice when locked.',
            'success': True
        }
        
    elif "disable voice unlock" in command_lower:
        result = await voice_unlock_connector.disable_monitoring()
        return {
            'type': 'voice_unlock',
            'action': 'disabled',
            'message': 'Voice unlock monitoring has been disabled, Sir.',
            'success': True
        }
        
    elif "voice unlock status" in command_lower:
        status = await voice_unlock_connector.get_status()
        if status:
            monitoring = status.get('isMonitoring', False)
            enrolled = status.get('enrolledUser', 'None')
            
            message = f"Voice unlock is {'active' if monitoring else 'inactive'}, Sir. "
            if enrolled and enrolled != 'None':
                message += f"Enrolled user: {enrolled}"
            else:
                message += "No users enrolled."
                
            return {
                'type': 'voice_unlock',
                'action': 'status',
                'message': message,
                'status': status,
                'success': True
            }
            
    elif any(phrase in command_lower for phrase in ['unlock my mac', 'unlock my screen', 'unlock mac', 'unlock the mac', 'unlock computer']):
        # User wants to unlock NOW - send unlock command directly to the daemon
        try:
            if not voice_unlock_connector or not voice_unlock_connector.connected:
                await initialize_voice_unlock()
            
            # Send unlock command to daemon
            result = await voice_unlock_connector.send_command("unlock_screen", {
                "source": "jarvis_command",
                "authenticated": True  # Already authenticated via JARVIS
            })
            
            if result and result.get('success'):
                return {
                    'response': 'Unlocking your screen now, Sir.',
                    'success': True
                }
            else:
                error_msg = result.get('message', 'Unknown error') if result else 'Not connected to Voice Unlock'
                return {
                    'response': f"I couldn't unlock the screen, Sir. {error_msg}",
                    'success': False
                }
        except Exception as e:
            logger.error(f"Error sending unlock command: {e}")
            return {
                'response': 'I encountered an error while trying to unlock your screen, Sir.',
                'error': str(e),
                'success': False
            }
    
    elif any(phrase in command_lower for phrase in ['lock my mac', 'lock my screen', 'lock mac', 'lock the mac', 'lock computer', 'lock the computer']):
        # User wants to lock the screen
        try:
            if not voice_unlock_connector or not voice_unlock_connector.connected:
                await initialize_voice_unlock()
            
            # Send lock command to daemon
            result = await voice_unlock_connector.send_command("lock_screen", {
                "source": "jarvis_command"
            })
            
            if result and result.get('success'):
                return {
                    'response': 'Locking your screen now, Sir.',
                    'success': True
                }
            else:
                error_msg = result.get('message', 'Unknown error') if result else 'Not connected to Voice Unlock'
                return {
                    'response': f"I couldn't lock the screen, Sir. {error_msg}",
                    'success': False
                }
        except Exception as e:
            logger.error(f"Error sending lock command: {e}")
            return {
                'response': 'I encountered an error while trying to lock your screen, Sir.',
                'error': str(e),
                'success': False
            }
            
    elif "test voice unlock" in command_lower:
        # Test the voice unlock system
        if not voice_unlock_connector or not voice_unlock_connector.connected:
            await initialize_voice_unlock()
            
        status = await voice_unlock_connector.get_status()
        if status:
            return {
                'type': 'voice_unlock',
                'action': 'test',
                'message': f'Voice unlock system test: {"✅ Connected and ready" if status.get("isMonitoring") else "❌ Not monitoring"}',
                'status': status,
                'success': True
            }
            
    return {
        'type': 'voice_unlock',
        'action': 'unknown',
        'message': "I didn't understand that voice unlock command, Sir.",
        'success': False
    }


# Integration function for main API
def integrate_voice_unlock_with_api(app):
    """
    Add Voice Unlock routes to the main FastAPI app
    
    This should be called from main.py to add Voice Unlock endpoints
    """
    from fastapi import APIRouter
    
    router = APIRouter(prefix="/voice-unlock", tags=["voice-unlock"])
    
    @router.get("/status")
    async def get_voice_unlock_status():
        """Get Voice Unlock daemon status"""
        if not voice_unlock_connector or not voice_unlock_connector.connected:
            await initialize_voice_unlock()
            
        if voice_unlock_connector and voice_unlock_connector.connected:
            status = await voice_unlock_connector.get_status()
            return {"status": "connected", "daemon_status": status}
        else:
            return {"status": "disconnected", "error": "Cannot connect to Voice Unlock daemon"}
    
    @router.post("/enable")
    async def enable_voice_unlock():
        """Enable Voice Unlock monitoring"""
        result = await handle_voice_unlock_in_jarvis("enable voice unlock")
        return result
    
    @router.post("/disable")
    async def disable_voice_unlock():
        """Disable Voice Unlock monitoring"""
        result = await handle_voice_unlock_in_jarvis("disable voice unlock")
        return result
    
    app.include_router(router)
    
    # Initialize on startup
    @app.on_event("startup")
    async def startup_voice_unlock():
        await initialize_voice_unlock()
    
    # Cleanup on shutdown
    @app.on_event("shutdown")
    async def shutdown_voice_unlock():
        if voice_unlock_connector:
            await voice_unlock_connector.disconnect()