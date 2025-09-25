#!/usr/bin/env python3
"""
Direct Unlock Handler
====================

Provides direct screen unlock functionality
"""

import asyncio
import logging
import websockets
import json
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

VOICE_UNLOCK_WS_URL = "ws://localhost:8765/voice-unlock"


async def unlock_screen_direct(reason: str = "User request") -> bool:
    """Directly unlock the screen using WebSocket connection"""
    try:
        # Connect to voice unlock daemon
        logger.info("Connecting to voice unlock daemon for direct unlock")
        
        async with websockets.connect(VOICE_UNLOCK_WS_URL) as websocket:
            # Send unlock command (Voice Unlock expects this format)
            unlock_command = {
                "type": "command",
                "command": "unlock_screen"
            }
            
            await websocket.send(json.dumps(unlock_command))
            logger.info(f"Sent unlock command: {unlock_command}")
            
            # Wait for response (longer timeout for unlock operation)
            response = await asyncio.wait_for(websocket.recv(), timeout=20.0)
            result = json.loads(response)
            
            logger.info(f"Unlock response: {result}")
            
            # Check for command_response type (what Voice Unlock actually sends)
            if result.get("type") == "command_response":
                success = result.get("success", False)
                logger.info(f"Unlock success: {success}, message: {result.get('message')}")
                return success
            elif result.get("type") == "unlock_result":
                return result.get("success", False)
            else:
                logger.error(f"Unexpected response type: {result.get('type')}")
                logger.error(f"Full response: {result}")
                return False
                
    except asyncio.TimeoutError:
        logger.error("Timeout waiting for unlock response")
        return False
    except Exception as e:
        logger.error(f"Error in direct unlock: {e}")
        return False


async def check_screen_locked_direct() -> bool:
    """Check if screen is locked using Context Intelligence"""
    try:
        # Use Context Intelligence screen state detector for accurate detection
        from context_intelligence.core.screen_state import ScreenStateDetector, ScreenState
        
        logger.info("[DIRECT UNLOCK] Checking screen lock via Context Intelligence")
        detector = ScreenStateDetector()
        state = await detector.get_screen_state()
        
        is_locked = state.state == ScreenState.LOCKED
        logger.info(f"[DIRECT UNLOCK] Screen state: {state.state.value} (confidence: {state.confidence:.2f})")
        logger.info(f"[DIRECT UNLOCK] Screen locked: {is_locked}")
        
        # Also try Voice Unlock daemon for comparison (but don't rely on it)
        try:
            async with websockets.connect(VOICE_UNLOCK_WS_URL) as websocket:
                status_command = {"type": "command", "command": "get_status"}
                await websocket.send(json.dumps(status_command))
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                result = json.loads(response)
                daemon_locked = result.get("status", {}).get("isScreenLocked", False)
                logger.debug(f"[DIRECT UNLOCK] Voice Unlock daemon reports: {daemon_locked} (ignored)")
        except:
            pass
        
        return is_locked
        
    except Exception as e:
        logger.error(f"Error checking screen lock: {e}")
        # Fallback to system check
        return check_screen_locked_system()


def check_screen_locked_system() -> bool:
    """Check screen lock state using system API"""
    try:
        logger.info("[DIRECT UNLOCK] Checking screen lock via system API")
        import subprocess
        result = subprocess.run(['python', '-c', '''
import Quartz
session_dict = Quartz.CGSessionCopyCurrentDictionary()
if session_dict:
    locked = session_dict.get("CGSSessionScreenIsLocked", False)
    print("true" if locked else "false")
else:
    print("false")
'''], capture_output=True, text=True)
        
        is_locked = result.stdout.strip().lower() == "true"
        logger.info(f"[DIRECT UNLOCK] Screen locked from system: {is_locked}")
        return is_locked
        
    except Exception as e:
        logger.error(f"Error in system screen check: {e}")
        return False