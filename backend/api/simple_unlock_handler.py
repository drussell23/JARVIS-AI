#!/usr/bin/env python3
"""
Simple Unlock Handler
====================

Direct unlock functionality without complex state management.
Just unlock the screen when asked.

Now integrated with AdvancedAsyncPipeline for non-blocking operations.
"""

import asyncio
import logging
import websockets
import json
import os
from typing import Dict, Any

# Import async pipeline
from core.async_pipeline import get_async_pipeline

logger = logging.getLogger(__name__)

VOICE_UNLOCK_WS_URL = "ws://localhost:8765/voice-unlock"

# Global pipeline instance
_pipeline = None

def _get_pipeline():
    """Get or create the async pipeline instance"""
    global _pipeline
    if _pipeline is None:
        _pipeline = get_async_pipeline()
        _register_pipeline_stages()
    return _pipeline

def _register_pipeline_stages():
    """Register simple unlock handler pipeline stages"""
    global _pipeline

    _pipeline.register_stage(
        "unlock_caffeinate",
        _caffeinate_async,
        timeout=3.0,
        retry_count=1,
        required=False
    )
    _pipeline.register_stage(
        "unlock_applescript",
        _applescript_unlock_async,
        timeout=15.0,
        retry_count=1,
        required=True
    )
    logger.info("✅ Simple unlock handler pipeline stages registered")

async def _caffeinate_async(context):
    """Async pipeline handler for waking display"""
    from api.jarvis_voice_api import async_subprocess_run

    try:
        # Ensure command is properly formatted
        caffeinate_cmd = ['caffeinate', '-u', '-t', '1']
        stdout, stderr, returncode = await async_subprocess_run(
            caffeinate_cmd,
            timeout=2.0
        )
        context.metadata["caffeinate_success"] = (returncode == 0)
        context.metadata["success"] = True
        logger.debug(f"Caffeinate result: returncode={returncode}")
    except Exception as e:
        logger.warning(f"Caffeinate failed: {e}")
        context.metadata["caffeinate_success"] = False
        context.metadata["error"] = str(e)
        # Don't fail the entire pipeline just because caffeinate failed
        context.metadata["success"] = True  # Allow continuation

async def _applescript_unlock_async(context):
    """Async pipeline handler for AppleScript unlock"""
    from api.jarvis_voice_api import async_osascript

    script = context.metadata.get("script", "")
    timeout = context.metadata.get("timeout", 10.0)

    try:
        stdout, stderr, returncode = await async_osascript(script, timeout=timeout)
        context.metadata["returncode"] = returncode
        context.metadata["stdout"] = stdout.decode() if stdout else ""
        context.metadata["stderr"] = stderr.decode() if stderr else ""
        context.metadata["success"] = (returncode == 0)
    except Exception as e:
        context.metadata["success"] = False
        context.metadata["error"] = str(e)


def _escape_password_for_applescript(password: str) -> str:
    """Escape special characters in password for AppleScript"""
    escaped = password.replace('\\', '\\\\')  # Escape backslashes
    escaped = escaped.replace('"', '\\"')     # Escape double quotes
    return escaped


async def _perform_direct_unlock(password: str) -> bool:
    """
    Perform direct screen unlock using AppleScript and password WITHOUT pipeline to avoid loops

    Args:
        password: The user's Mac password from keychain

    Returns:
        bool: True if unlock succeeded, False otherwise
    """
    try:
        logger.info("[DIRECT UNLOCK] Starting unlock sequence (direct, no pipeline)")

        # Import all required functions at the beginning
        from api.jarvis_voice_api import async_subprocess_run, async_osascript

        # Wake the display first directly
        try:
            await async_subprocess_run(
                ['caffeinate', '-u', '-t', '1'],
                timeout=2.0
            )
        except Exception as e:
            logger.debug(f"Caffeinate failed (non-critical): {e}")

        await asyncio.sleep(1)

        # Wake script - move mouse and activate loginwindow
        wake_script = '''
        tell application "System Events"
            -- Wake the display by moving mouse
            do shell script "caffeinate -u -t 2"
            delay 0.5

            -- Click on the user profile to show password field
            click at {720, 860}
            delay 1

            -- Make sure loginwindow is frontmost
            set frontmost of process "loginwindow" to true
            delay 0.5

            -- Sometimes need to click again to ensure password field is active
            click at {720, 500}
            delay 0.5

            -- Clear any existing text
            keystroke "a" using command down
            delay 0.1
            key code 51
            delay 0.2
        end tell
        '''

        # Execute wake script directly (no pipeline)
        try:
            stdout, stderr, returncode = await async_osascript(wake_script, timeout=10.0)
            if returncode != 0:
                logger.debug(f"Wake script failed: {stderr}")
        except Exception as e:
            logger.debug(f"Wake script error: {e}")

        await asyncio.sleep(0.5)

        # Escape password for AppleScript
        escaped_password = _escape_password_for_applescript(password)
        logger.info(f"[DIRECT UNLOCK] Typing password ({len(password)} characters)")

        # Type password using System Events
        password_script = f'''
        tell application "System Events"
            tell process "loginwindow"
                set frontmost to true
                delay 0.3

                -- Type the password
                keystroke "{escaped_password}"
                delay 0.5

                -- Press return to unlock
                keystroke return
                delay 1
            end tell
        end tell
        '''

        # Execute password script directly (no pipeline)
        stdout, stderr, returncode = await async_osascript(password_script, timeout=15.0)

        if returncode == 0:
            # Wait a bit for unlock to complete
            await asyncio.sleep(1.5)

            # Verify unlock by checking screen state
            try:
                from voice_unlock.objc.server.screen_lock_detector import is_screen_locked
                is_locked = is_screen_locked()

                if not is_locked:
                    logger.info("[DIRECT UNLOCK] Unlock verified successful")
                    return True
                else:
                    logger.warning("[DIRECT UNLOCK] Screen still locked after attempt")
                    return False
            except:
                # If we can't verify, assume success if no errors
                logger.info("[DIRECT UNLOCK] Unlock completed (verification unavailable)")
                return True
        else:
            logger.error(f"[DIRECT UNLOCK] AppleScript error: {stderr}")
            return False

    except Exception as e:
        logger.error(f"[DIRECT UNLOCK] Error during unlock: {e}")
        return False


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
                
    except Exception as e:
        # This catches websockets connection errors and other exceptions
        logger.warning(f"[SIMPLE UNLOCK] Voice unlock daemon not running: {e}")

        # For lock_screen, use the MacOS controller which has async pipeline integration
        if action == "lock_screen":
            logger.info("[SIMPLE UNLOCK] Routing lock command to MacOS controller with async pipeline")

            try:
                from system_control.macos_controller import MacOSController
                controller = MacOSController()
                success, message = await controller.lock_screen()
                lock_success = success
                logger.info(f"[SIMPLE UNLOCK] Lock result: {success}, {message}")
            except Exception as e:
                logger.error(f"MacOS controller lock failed: {e}")
                lock_success = False

            # Method 3: Start screensaver as last resort
            if not lock_success:
                try:
                    subprocess.run(['open', '-a', 'ScreenSaverEngine'], check=True)
                    lock_success = True
                    logger.info("[SIMPLE UNLOCK] Started screensaver")
                except Exception as e:
                    logger.debug(f"ScreenSaver method failed: {e}")

            if lock_success:
                return {
                    'success': True,
                    'response': response_text,
                    'type': 'screen_lock',
                    'action': action,
                    'method': 'direct'
                }
            else:
                logger.error("[SIMPLE UNLOCK] All lock methods failed")
                return {
                    'success': False,
                    'response': "I couldn't lock your screen, Sir. Please use Control+Command+Q manually.",
                    'type': 'screen_lock',
                    'action': action
                }
        else:
            # For unlock - try direct methods if daemon not available
            logger.info(f"[SIMPLE UNLOCK] Attempting unlock without daemon")

            # Check if screen is actually locked first
            try:
                from voice_unlock.objc.server.screen_lock_detector import is_screen_locked
                if not is_screen_locked():
                    return {
                        'success': True,
                        'response': "Your screen is already unlocked, Sir.",
                        'type': 'screen_unlock',
                        'action': action
                    }
            except:
                pass

            # Method 1: Try to retrieve password from keychain and unlock
            unlock_success = False
            unlock_method = None

            try:
                logger.info("[SIMPLE UNLOCK] Attempting to retrieve password from keychain")

                # Try to get password from keychain
                result = subprocess.run([
                    'security', 'find-generic-password',
                    '-s', 'com.jarvis.voiceunlock',
                    '-a', 'unlock_token',
                    '-w'  # Print only the password
                ], capture_output=True, text=True)
                if result.returncode == 0:
                    password = result.stdout.strip()
                    logger.info("[SIMPLE UNLOCK] Password retrieved from keychain, attempting unlock")

                    # Perform unlock using the password
                    unlock_result = await _perform_direct_unlock(password)

                    if unlock_result:
                        logger.info("[SIMPLE UNLOCK] Direct unlock successful")
                        return {
                            'success': True,
                            'response': response_text,
                            'type': 'screen_unlock',
                            'action': action,
                            'method': 'keychain_direct'
                        }
                    else:
                        logger.error("[SIMPLE UNLOCK] Direct unlock failed")

                else:
                    logger.warning(f"[SIMPLE UNLOCK] Password not in keychain: {result.stderr}")

            except Exception as e:
                logger.debug(f"Keychain unlock failed: {e}")

            # If keychain method failed, provide helpful message about setup
            return {
                'success': False,
                'response': "I cannot unlock your screen without the Voice Unlock daemon, Sir. To enable automatic unlocking, please run: ./backend/voice_unlock/enable_screen_unlock.sh",
                'type': 'voice_unlock',
                'action': action,
                'requires_daemon': True,
                'setup_instructions': {
                    'command': './backend/voice_unlock/enable_screen_unlock.sh',
                    'description': 'This will securely store your password and start the unlock service'
                }
            }

    except Exception as e:
        logger.error(f"[SIMPLE UNLOCK] Error: {e}")
        return {
            'success': False,
            'response': f"I encountered an error: {str(e)}",
            'type': 'voice_unlock',
            'action': action
        }