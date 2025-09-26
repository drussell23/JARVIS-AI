#!/usr/bin/env python3
"""
Manual Unlock Handler
=====================

Handles manual "unlock my screen" commands without policy restrictions
"""

import logging
from typing import Dict, Any

from api.direct_unlock_handler_fixed import unlock_screen_direct

logger = logging.getLogger(__name__)


async def handle_manual_unlock(command: str, websocket=None) -> Dict[str, Any]:
    """Handle manual unlock request directly without policy checks"""
    logger.info("[MANUAL UNLOCK] User requested manual screen unlock")
    
    # Send immediate feedback
    if websocket:
        await websocket.send_json({
            "type": "response",
            "text": "I'll unlock your screen right away, Sir.",
            "command_type": "manual_unlock",
            "speak": True,
            "intermediate": True
        })
    
    # Perform unlock
    success = await unlock_screen_direct("Manual user request - bypass quiet hours")
    
    if success:
        return {
            "success": True,
            "response": "I've unlocked your screen, Sir.",
            "command_type": "manual_unlock"
        }
    else:
        return {
            "success": False,
            "response": "I couldn't unlock the screen, Sir. Please check if the Voice Unlock daemon is running.",
            "command_type": "manual_unlock"
        }
