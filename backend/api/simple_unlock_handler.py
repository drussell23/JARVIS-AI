#!/usr/bin/env python3
"""
Simple Unlock Handler
====================

Direct unlock functionality without complex state management.
Just unlock the screen when asked.
"""

import asyncio
import logging
import websockets
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)

VOICE_UNLOCK_WS_URL = "ws://localhost:8765/voice-unlock"


async def handle_unlock_command(command: str) -> Dict[str, Any]:
    """Handle unlock/lock screen commands directly"""
    
    command_lower = command.lower()
    
    # Determine action
    if any(phrase in command_lower for phrase in ['unlock my screen', 'unlock screen', 'unlock the screen']):
        action = "unlock_screen"
        response_text = "Unlocking your screen now, Sir."
    elif any(phrase in command_lower for phrase in ['lock my screen', 'lock screen', 'lock the screen']):
        action = "lock_screen"
        response_text = "Locking your screen now, Sir."
    else:
        return {
            'success': False,
            'response': "I didn't understand that screen command, Sir."
        }
    
    # Execute the action
    try:
        # Create a fresh WebSocket connection for this command
        async with websockets.connect(VOICE_UNLOCK_WS_URL, ping_interval=20) as ws:
            # Send command
            cmd_msg = {
                "type": "command",
                "command": action,
                "parameters": {
                    "source": "jarvis_command",
                    "authenticated": True
                }
            }
            
            await ws.send(json.dumps(cmd_msg))
            logger.info(f"[SIMPLE UNLOCK] Sent {action} command")
            
            # Wait for response with timeout
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=30.0)  # Increase timeout for unlock operation
                result = json.loads(response)
                
                if result.get('success'):
                    logger.info(f"[SIMPLE UNLOCK] {action} succeeded")
                    return {
                        'success': True,
                        'response': response_text,
                        'type': 'voice_unlock',
                        'action': action
                    }
                else:
                    error_msg = result.get('message', 'Unknown error')
                    logger.error(f"[SIMPLE UNLOCK] {action} failed: {error_msg}")
                    return {
                        'success': False,
                        'response': f"I couldn't {action.replace('_', ' ')}, Sir. {error_msg}",
                        'type': 'voice_unlock',
                        'action': action
                    }
                    
            except asyncio.TimeoutError:
                logger.error(f"[SIMPLE UNLOCK] Timeout waiting for {action} response")
                return {
                    'success': False,
                    'response': f"The {action.replace('_', ' ')} operation timed out, Sir.",
                    'type': 'voice_unlock',
                    'action': action
                }
                
    except (ConnectionRefusedError, OSError) as e:
        logger.error(f"[SIMPLE UNLOCK] Voice unlock daemon not running: {e}")
        return {
            'success': False,
            'response': "The Voice Unlock service isn't running, Sir. Shall I start it?",
            'type': 'voice_unlock',
            'action': action
        }
    except Exception as e:
        logger.error(f"[SIMPLE UNLOCK] Error: {e}")
        return {
            'success': False,
            'response': f"I encountered an error: {str(e)}",
            'type': 'voice_unlock', 
            'action': action
        }