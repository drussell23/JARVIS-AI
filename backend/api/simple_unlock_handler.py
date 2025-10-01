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


async def handle_unlock_command(command: str, jarvis_instance=None) -> Dict[str, Any]:
    """Handle unlock/lock screen commands directly with dynamic responses"""
    
    command_lower = command.lower()
    
    # Determine action
    if any(phrase in command_lower for phrase in ['unlock my screen', 'unlock screen', 'unlock the screen']):
        action = "unlock_screen"
        # Default response if JARVIS instance not available
        response_text = "Unlocking your screen now, Sir."
    elif any(phrase in command_lower for phrase in ['lock my screen', 'lock screen', 'lock the screen']):
        action = "lock_screen"
        # Default response if JARVIS instance not available
        response_text = "Locking your screen now, Sir."
    else:
        return {
            'success': False,
            'response': "I didn't understand that screen command, Sir."
        }
    
    # Generate dynamic, contextual responses instead of hardcoded ones
    import random
    from datetime import datetime
    
    # Get time-based context
    hour = datetime.now().hour
    time_context = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"
    
    if action == "unlock_screen":
        # Dynamic unlock responses
        unlock_responses = [
            f"Right away, Sir. Unlocking your screen now.",
            f"Accessing your system now, Sir.",
            f"Certainly, Sir. Screen unlock in progress.",
            f"Of course, Sir. Unlocking for you.",
            f"Immediately, Sir. Your screen is being unlocked.",
            f"As you wish, Sir. Unlocking now.",
            f"At once, Sir. Accessing your desktop."
        ]
        
        # Add time-based variations
        if time_context == "morning":
            unlock_responses.extend([
                f"Good morning, Sir. Unlocking your screen now.",
                f"Starting your {time_context} with screen access, Sir."
            ])
        elif time_context == "evening":
            unlock_responses.extend([
                f"Welcome back this {time_context}, Sir. Unlocking now.",
                f"Continuing your {time_context} work, Sir. Screen unlocked."
            ])
            
        response_text = random.choice(unlock_responses)
        
    else:  # lock_screen
        # Dynamic lock responses
        lock_responses = [
            f"Securing your system now, Sir.",
            f"Right away, Sir. Locking your screen.",
            f"Of course, Sir. Your screen is being locked.",
            f"Certainly, Sir. Securing your desktop.",
            f"At your command, Sir. Screen lock engaged.",
            f"Immediately, Sir. Your system is secured.",
            f"As requested, Sir. Locking now."
        ]
        
        # Add time-based variations
        if time_context == "evening":
            lock_responses.extend([
                f"Securing your system for the {time_context}, Sir.",
                f"Have a pleasant {time_context}, Sir. Screen locked."
            ])
        elif hour >= 17:  # Late afternoon/evening
            lock_responses.extend([
                f"Securing your workstation, Sir. Enjoy your {time_context}."
            ])
            
        response_text = random.choice(lock_responses)
    
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
        logger.warning(f"[SIMPLE UNLOCK] Voice unlock daemon not running: {e}")

        # For lock_screen, execute directly without the daemon
        if action == "lock_screen":
            logger.info("[SIMPLE UNLOCK] Falling back to direct macOS lock command")
            import subprocess
            try:
                # Use pmset to lock the screen (works on all macOS versions)
                subprocess.run([
                    "pmset", "displaysleepnow"
                ], check=True)

                logger.info("[SIMPLE UNLOCK] Screen locked successfully via direct command")
                return {
                    'success': True,
                    'response': response_text,
                    'type': 'screen_lock',
                    'action': action,
                    'method': 'direct'
                }
            except subprocess.CalledProcessError as lock_error:
                logger.error(f"[SIMPLE UNLOCK] Failed to lock screen: {lock_error}")
                return {
                    'success': False,
                    'response': "I couldn't lock your screen, Sir. Please check system permissions.",
                    'type': 'screen_lock',
                    'action': action
                }
        else:
            # For unlock, we need the daemon
            return {
                'success': False,
                'response': "The Voice Unlock service isn't running, Sir. I can't unlock without it.",
                'type': 'voice_unlock',
                'action': action,
                'requires_daemon': True
            }

    except Exception as e:
        logger.error(f"[SIMPLE UNLOCK] Error: {e}")
        return {
            'success': False,
            'response': f"I encountered an error: {str(e)}",
            'type': 'voice_unlock',
            'action': action
        }