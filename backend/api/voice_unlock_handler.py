"""
Voice Unlock Command Handler
===========================

Handles natural language commands for voice unlock functionality.
"""

import logging
from typing import Dict, Any, Optional
import asyncio

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.voice_unlock_integration import voice_unlock_connector, initialize_voice_unlock, handle_voice_unlock_in_jarvis

logger = logging.getLogger(__name__)

# Global instance
_connector = None


async def ensure_daemon_connection():
    """Ensure we're connected to the Voice Unlock daemon"""
    global _connector
    
    if not _connector or not _connector.connected:
        from api.voice_unlock_integration import VoiceUnlockDaemonConnector
        _connector = VoiceUnlockDaemonConnector()
        try:
            await _connector.connect()
            logger.info("Connected to Voice Unlock daemon")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Voice Unlock daemon: {e}")
            return False
    return True


async def handle_voice_unlock_command(command: str, websocket=None) -> Dict[str, Any]:
    """Handle voice unlock related commands"""
    
    logger.info(f"[VOICE UNLOCK] Handling command: {command}")
    
    # Use the integration handler that connects to the Objective-C daemon
    return await handle_voice_unlock_in_jarvis(command)
    


# Handler class for unified command processor
class VoiceUnlockHandler:
    """Handler for voice unlock commands"""
    
    async def handle_command(self, command: str, websocket=None) -> Dict[str, Any]:
        """Process voice unlock command"""
        return await handle_voice_unlock_command(command, websocket)
    
    async def process_command(self, command: str, websocket=None) -> Dict[str, Any]:
        """Process voice unlock command (backward compatibility)"""
        return await self.handle_command(command, websocket)


# Singleton instance
voice_unlock_handler = VoiceUnlockHandler()

# Function to get the handler instance
def get_voice_unlock_handler():
    """Get the voice unlock handler instance"""
    return voice_unlock_handler

# For backward compatibility with the unified command processor
async def process_command(command: str, websocket=None) -> Dict[str, Any]:
    """Process voice unlock command (called by UnifiedCommandProcessor)"""
    return await handle_voice_unlock_command(command, websocket)