#!/usr/bin/env python3
"""
Simple Unlock Handler
====================

Direct unlock functionality without complex state management.
Just unlock the screen when asked.

Now integrated with AdvancedAsyncPipeline for non-blocking operations.
"""

import asyncio
import json
import logging
import subprocess
from typing import Any, Dict, List, Tuple

import websockets

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
        "unlock_caffeinate", _caffeinate_async, timeout=3.0, retry_count=1, required=False
    )
    _pipeline.register_stage(
        "unlock_applescript", _applescript_unlock_async, timeout=15.0, retry_count=1, required=True
    )
    logger.info("✅ Simple unlock handler pipeline stages registered")


async def _caffeinate_async(context):
    """Async pipeline handler for waking display"""
    from api.jarvis_voice_api import async_subprocess_run

    try:
        # Ensure command is properly formatted
        caffeinate_cmd = ["caffeinate", "-u", "-t", "1"]
        stdout, stderr, returncode = await async_subprocess_run(caffeinate_cmd, timeout=2.0)
        context.metadata["caffeinate_success"] = returncode == 0
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
        context.metadata["success"] = returncode == 0
    except Exception as e:
        context.metadata["success"] = False
        context.metadata["error"] = str(e)


def _escape_password_for_applescript(password: str) -> str:
    """Escape special characters in password for AppleScript"""
    escaped = password.replace("\\", "\\\\")  # Escape backslashes
    escaped = escaped.replace('"', '\\"')  # Escape double quotes
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
        from api.jarvis_voice_api import async_osascript, async_subprocess_run

        # Wake the display first directly
        try:
            await async_subprocess_run(["caffeinate", "-u", "-t", "1"], timeout=2.0)
        except Exception as e:
            logger.debug(f"Caffeinate failed (non-critical): {e}")

        await asyncio.sleep(1)

        # Wake script - move mouse and activate loginwindow
        wake_script = """
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
        """

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
        password_script = f"""
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
        """

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
    """
    Enhanced unlock/lock screen command handler with dynamic response generation,
    advanced command parsing, and intelligent fallback mechanisms.

    Features:
    - Dynamic response generation using AI
    - Advanced command parsing with context awareness
    - Multiple unlock/lock methods with intelligent fallback
    - Context-aware error handling and user guidance
    - Performance monitoring and adaptive behavior
    """

    # Initialize response generator for dynamic responses
    try:
        from voice.dynamic_response_generator import get_response_generator

        response_gen = get_response_generator()
    except ImportError:
        response_gen = None
        logger.warning("Dynamic response generator not available, using fallback responses")

    # Advanced command parsing and intent detection
    command_analysis = await _analyze_unlock_command(command, jarvis_instance)

    if not command_analysis["is_valid"]:
        return await _generate_command_error_response(command_analysis, response_gen)

    action = command_analysis["action"]
    context = command_analysis["context"]

    # Generate dynamic response based on context
    response_text = await _generate_contextual_response(action, context, response_gen)

    # Execute the action with multiple fallback methods
    result = await _execute_screen_action(action, context, jarvis_instance)

    # Enhance result with dynamic response and context
    return await _enhance_result_with_context(result, response_text, action, context, response_gen)


async def _analyze_unlock_command(command: str, jarvis_instance=None) -> Dict[str, Any]:
    """
    Advanced command analysis - SPEED OPTIMIZED.
    Fast pattern matching without heavy regex.
    """

    command_lower = command.lower().strip()

    # FAST PATH: Simple keyword matching (no regex overhead)
    is_unlock = "unlock" in command_lower
    is_lock = "lock" in command_lower and not is_unlock

    # Determine action
    if is_unlock:
        action = "unlock_screen"
    elif is_lock:
        action = "lock_screen"
    else:
        return {
            "is_valid": False,
            "action": None,
            "context": {
                "confidence": 0.0,
                "reason": "no_valid_intent_detected",
                "suggestions": ["Try 'unlock my screen' or 'lock my screen'"],
            },
        }

    # Fast context analysis
    context = await _analyze_command_context(command, action, jarvis_instance)

    return {"is_valid": True, "action": action, "context": context}


async def _analyze_command_context(
    command: str, action: str, jarvis_instance=None
) -> Dict[str, Any]:
    """Analyze command context for urgency, politeness, and user state - SPEED OPTIMIZED."""

    command_lower = command.lower()

    # FAST PATH: Quick single-pass analysis
    is_urgent = "now" in command_lower or "quickly" in command_lower or "fast" in command_lower
    is_polite = "please" in command_lower

    # Minimal context for speed
    return {
        "urgency": "high" if is_urgent else "normal",
        "politeness": "polite" if is_polite else "direct",
        "user_state": "normal",  # Skip complex state detection for speed
        "time_context": "day",  # Skip time calculation for speed
        "screen_state": "unknown",  # Skip screen check for speed
        "command_original": command,
        "confidence": 0.95,
    }


async def _generate_contextual_response(action: str, context: Dict[str, Any], response_gen) -> str:
    """Generate dynamic, contextual response based on action and context - SPEED OPTIMIZED."""

    # FAST PATH: Use pre-generated responses for speed (no dynamic generation overhead)
    # This eliminates the response_gen overhead while keeping contextual awareness

    if action == "unlock_screen" and context.get("screen_state") == "unlocked":
        return "Your screen is already unlocked."

    # Use fast fallback responses (no random selection overhead)
    if context.get("urgency") == "high":
        return "Right away! " + (
            "Unlocking your screen now."
            if action == "unlock_screen"
            else "Locking your screen now."
        )

    # Default fast responses
    return "Of course, Sir. " + (
        "Unlocking for you." if action == "unlock_screen" else "Locking for you."
    )


def _generate_fallback_response(action: str, context: Dict[str, Any]) -> str:
    """Generate intelligent fallback responses based on context."""

    responses = {
        "unlock_screen": {
            "unlocked": [
                "Your screen is already unlocked.",
                "The screen is already accessible.",
                "No need to unlock - you're already in.",
            ],
            "locked": {
                "urgent": [
                    "Right away! Unlocking your screen now.",
                    "Immediately unlocking for you.",
                    "Quickly accessing your screen.",
                ],
                "polite": [
                    "Of course, I'll unlock that for you.",
                    "Certainly, unlocking your screen.",
                    "I'd be happy to unlock that for you.",
                ],
                "tired": [
                    "I'll unlock your screen so you can rest.",
                    "Let me get that unlocked for you.",
                    "Unlocking now - you can relax.",
                ],
                "normal": [
                    "Unlocking your screen.",
                    "I'll unlock that for you.",
                    "Accessing your screen now.",
                ],
            },
        },
        "lock_screen": {
            "urgent": [
                "Securing your screen immediately.",
                "Right away - locking now.",
                "Quickly securing your screen.",
            ],
            "polite": [
                "Of course, I'll lock your screen.",
                "Certainly, securing that for you.",
                "I'd be happy to lock that for you.",
            ],
            "leaving": [
                "I'll lock your screen before you go.",
                "Securing your screen for your departure.",
                "Locking up as you requested.",
            ],
            "normal": [
                "Locking your screen.",
                "I'll secure that for you.",
                "Protecting your screen now.",
            ],
        },
    }

    # Select appropriate response category
    if action == "unlock_screen":
        if context["screen_state"] == "unlocked":
            category = "unlocked"
        else:
            if context["urgency"] == "high":
                category = "urgent"
            elif context["politeness"] == "polite":
                category = "polite"
            elif context["user_state"] == "tired":
                category = "tired"
            else:
                category = "normal"
    else:  # lock_screen
        if context["urgency"] == "high":
            category = "urgent"
        elif context["politeness"] == "polite":
            category = "polite"
        elif context["user_state"] == "leaving":
            category = "leaving"
        else:
            category = "normal"

    # Get response options and select one
    import random

    response_options = responses[action][category]
    base_response = random.choice(response_options)  # nosec B311 # UI responses, not cryptographic

    # Add time-based context occasionally
    if random.random() < 0.3:  # nosec B311 # UI variation, not cryptographic
        time_additions = {
            "morning": "Good morning! ",
            "afternoon": "Good afternoon! ",
            "evening": "Good evening! ",
            "night": "Good evening! ",
        }
        base_response = time_additions.get(context["time_context"], "") + base_response

    return base_response


def _generate_command_suggestions(command: str) -> List[str]:
    """Generate helpful suggestions for unclear commands."""

    suggestions = [
        "Try saying 'unlock my screen' or 'lock my screen'",
        "You can say 'unlock' or 'lock' followed by 'screen'",
        "Commands like 'unlock my screen' or 'lock my screen' work well",
        "Try 'unlock screen' or 'lock screen' for screen control",
    ]

    # Check if command contains screen-related words
    if "screen" in command.lower():
        suggestions.extend(
            [
                "For screen control, try 'unlock screen' or 'lock screen'",
                "Screen commands: 'unlock my screen' or 'lock my screen'",
            ]
        )

    return suggestions[:3]  # Return top 3 suggestions


async def _execute_screen_action(
    action: str, context: Dict[str, Any], jarvis_instance=None
) -> Dict[str, Any]:
    """Execute screen action with multiple fallback methods and intelligent routing."""

    logger.info(f"[ENHANCED UNLOCK] Executing {action} with context: {context}")

    # Pass jarvis_instance to context for audio/speaker extraction
    if jarvis_instance:
        context["jarvis_instance"] = jarvis_instance

    # Try primary method first (WebSocket daemon)
    try:
        result = await _try_websocket_method(action, context)
        if result["success"]:
            result["method"] = "websocket_daemon"
            return result
    except Exception as e:
        logger.debug(f"WebSocket method failed: {e}")

    # Try secondary methods based on action type
    if action == "lock_screen":
        result = await _try_lock_methods(context)
    else:  # unlock_screen
        result = await _try_unlock_methods(context)

    return result


async def _try_websocket_method(action: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Try WebSocket daemon method with enhanced error handling."""

    async with websockets.connect(VOICE_UNLOCK_WS_URL, ping_interval=20) as ws:
        cmd_msg = {
            "type": "command",
            "command": action,
            "parameters": {
                "source": "jarvis_enhanced_command",
                "authenticated": True,
                "context": context,
            },
        }

        await ws.send(json.dumps(cmd_msg))
        logger.info(f"[ENHANCED UNLOCK] Sent {action} command via WebSocket")

        # Dynamic timeout based on urgency
        timeout = 15.0 if context["urgency"] == "high" else 30.0

        try:
            response = await asyncio.wait_for(ws.recv(), timeout=timeout)
            result = json.loads(response)

            return {
                "success": result.get("success", False),
                "message": result.get("message", ""),
                "data": result.get("data", {}),
            }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "message": f"Operation timed out after {timeout} seconds",
                "error_type": "timeout",
            }


async def _try_lock_methods(context: Dict[str, Any]) -> Dict[str, Any]:
    """Try multiple lock methods with intelligent fallback."""

    methods = [
        ("macos_controller", _try_macos_controller_lock),
        ("screensaver", _try_screensaver_lock),
        ("system_command", _try_system_lock_command),
    ]

    for method_name, method_func in methods:
        try:
            success, message = await method_func(context)
            if success:
                return {"success": True, "message": message, "method": method_name}
        except Exception as e:
            logger.debug(f"Lock method {method_name} failed: {e}")

    return {
        "success": False,
        "message": "All lock methods failed",
        "error_type": "all_methods_failed",
    }


async def _try_unlock_methods(context: Dict[str, Any]) -> Dict[str, Any]:
    """Try multiple unlock methods with intelligent fallback."""

    # Check if screen is already unlocked
    try:
        from voice_unlock.objc.server.screen_lock_detector import is_screen_locked

        if not is_screen_locked():
            return {
                "success": True,
                "message": "Screen was already unlocked",
                "method": "already_unlocked",
            }
    except:
        pass

    # Extract audio data from jarvis_instance if available
    # This enables voice verification for unlock commands
    jarvis_instance = context.get("jarvis_instance")
    if jarvis_instance:
        # Try to get audio data from recent voice interaction
        if hasattr(jarvis_instance, "last_audio_data"):
            context["audio_data"] = jarvis_instance.last_audio_data
            logger.debug("✅ Audio data extracted for voice verification")

        # Try to get speaker name from recent identification
        if hasattr(jarvis_instance, "last_speaker_name"):
            context["speaker_name"] = jarvis_instance.last_speaker_name
            logger.debug(f"✅ Speaker name extracted: {jarvis_instance.last_speaker_name}")

    methods = [
        ("keychain_direct", _try_keychain_unlock),
        ("manual_unlock", _try_manual_unlock_fallback),
    ]

    for method_name, method_func in methods:
        try:
            success, message = await method_func(context)
            if success:
                return {"success": True, "message": message, "method": method_name}
        except Exception as e:
            logger.debug(f"Unlock method {method_name} failed: {e}")

    return {
        "success": False,
        "message": "All unlock methods failed",
        "error_type": "all_methods_failed",
    }


async def _try_macos_controller_lock(context: Dict[str, Any]) -> Tuple[bool, str]:
    """Try MacOS controller lock method."""
    from system_control.macos_controller import MacOSController

    controller = MacOSController()
    return await controller.lock_screen()


async def _try_screensaver_lock(context: Dict[str, Any]) -> Tuple[bool, str]:
    """Try screensaver lock method."""
    subprocess.run(["open", "-a", "ScreenSaverEngine"], check=True)
    return True, "Screensaver started"


async def _try_system_lock_command(context: Dict[str, Any]) -> Tuple[bool, str]:
    """Try system command lock method."""
    subprocess.run(["pmset", "displaysleepnow"], check=True)
    return True, "Display sleep activated"


async def _try_keychain_unlock(context: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Try keychain-based unlock method with voice verification.

    Security Flow:
    1. Verify speaker identity from audio (if available)
    2. Check if speaker is the device owner
    3. Retrieve password from keychain
    4. Perform unlock
    """
    # Step 1: Voice verification (if audio data is available in context)
    audio_data = context.get("audio_data")
    speaker_name = context.get("speaker_name")

    if audio_data:
        # Verify speaker using voice biometrics
        try:
            from voice.speaker_recognition import get_speaker_recognition_engine

            speaker_engine = get_speaker_recognition_engine()
            await speaker_engine.initialize()

            # Identify speaker if not already identified
            if not speaker_name:
                speaker_name, confidence = await speaker_engine.identify_speaker(audio_data)
                logger.info(
                    f"🔐 Speaker identified for unlock: {speaker_name} (confidence: {confidence:.2f})"
                )
            else:
                # Verify claimed speaker
                is_verified, confidence = await speaker_engine.verify_speaker(
                    audio_data, speaker_name
                )
                logger.info(
                    f"🔐 Speaker verification for unlock: {speaker_name} - {'✅ VERIFIED' if is_verified else '❌ FAILED'} (confidence: {confidence:.2f})"
                )

                if not is_verified:
                    logger.warning(
                        f"🚫 Voice verification failed for {speaker_name} - unlock denied"
                    )
                    return (
                        False,
                        f"Voice verification failed (confidence: {confidence:.2%}). Unlock denied for security.",
                    )

            # Check if speaker is the device owner
            if not speaker_engine.is_owner(speaker_name):
                logger.warning(f"🚫 Non-owner {speaker_name} attempted unlock - denied")
                return (
                    False,
                    f"Only the device owner can unlock the screen via voice. User '{speaker_name}' does not have unlock privileges.",
                )

            # Get security clearance
            security_clearance = speaker_engine.get_security_clearance(speaker_name)
            if security_clearance != "admin":
                logger.warning(
                    f"🚫 User {speaker_name} lacks admin clearance for unlock (level: {security_clearance})"
                )
                return (
                    False,
                    f"Insufficient security clearance for screen unlock. Required: admin, Current: {security_clearance}",
                )

            logger.info(f"✅ Voice verification passed for owner: {speaker_name}")

        except Exception as e:
            logger.error(f"Voice verification error: {e}")
            # For security, fail closed - don't allow unlock if verification fails
            return False, f"Voice verification system error: {e}"
    else:
        # No audio data provided - this is a security risk for voice unlock
        # Only allow if explicitly bypassed in context (for manual/admin unlocks)
        if not context.get("bypass_voice_verification", False):
            logger.warning("🚫 No audio data for voice verification - unlock denied")
            return False, "Voice verification required for screen unlock. No audio data provided."
        else:
            logger.info("⚠️  Voice verification bypassed by admin flag")

    # Step 2: Retrieve password from keychain
    result = subprocess.run(
        [
            "security",
            "find-generic-password",
            "-s",
            "com.jarvis.voiceunlock",
            "-a",
            "unlock_token",
            "-w",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        password = result.stdout.strip()

        # Step 3: Perform unlock
        unlock_result = await _perform_direct_unlock(password)

        if unlock_result:
            logger.info(f"🔓 Screen unlocked successfully by {speaker_name or 'verified user'}")
            return True, f"Screen unlocked by {speaker_name or 'verified user'}"
        else:
            return False, "Keychain unlock failed - unable to unlock screen"

    return False, "Password not found in keychain"


async def _try_manual_unlock_fallback(context: Dict[str, Any]) -> Tuple[bool, str]:
    """Try manual unlock fallback method."""
    # This would trigger a manual unlock process
    return False, "Manual unlock not implemented"


async def _generate_command_error_response(
    analysis: Dict[str, Any], response_gen
) -> Dict[str, Any]:
    """Generate helpful error response for invalid commands."""

    if response_gen:
        try:
            error_response = response_gen.get_error_message(
                "invalid_command", "I didn't understand that screen command"
            )
        except:
            error_response = "I didn't understand that screen command."
    else:
        error_response = "I didn't understand that screen command."

    return {
        "success": False,
        "response": error_response,
        "type": "command_error",
        "suggestions": analysis["context"]["suggestions"],
        "confidence": analysis["context"]["confidence"],
    }


async def _enhance_result_with_context(
    result: Dict[str, Any], response_text: str, action: str, context: Dict[str, Any], response_gen
) -> Dict[str, Any]:
    """Enhance result with dynamic response and contextual information."""

    # Add dynamic response
    result["response"] = response_text

    # CRITICAL: Add action field for fast path compatibility
    result["action"] = action

    # Add type for voice_unlock compatibility
    if "type" not in result:
        result["type"] = "voice_unlock" if action == "unlock_screen" else "screen_lock"

    # Add contextual metadata
    result["enhanced_context"] = {
        "action": action,
        "urgency": context["urgency"],
        "politeness": context["politeness"],
        "user_state": context["user_state"],
        "time_context": context["time_context"],
        "execution_method": result.get("method", "unknown"),
    }

    # Add performance metrics
    result["performance"] = {
        "command_confidence": context["confidence"],
        "response_type": "dynamic" if response_gen else "fallback",
    }

    # Add helpful information for failures
    if not result["success"]:
        result["troubleshooting"] = await _generate_troubleshooting_info(action, result, context)

    return result


async def _generate_troubleshooting_info(
    action: str, result: Dict[str, Any], context: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate helpful troubleshooting information for failed operations."""

    troubleshooting = {
        "action": action,
        "error_type": result.get("error_type", "unknown"),
        "suggestions": [],
    }

    if action == "unlock_screen":
        troubleshooting["suggestions"] = [
            "Make sure your password is stored in the keychain",
            "Try running: ./backend/voice_unlock/enable_screen_unlock.sh",
            "Check if the Voice Unlock daemon is running",
            "Verify your screen is actually locked",
        ]
    else:  # lock_screen
        troubleshooting["suggestions"] = [
            "Try using Control+Command+Q manually",
            "Check if Screen Time restrictions are enabled",
            "Verify you have permission to lock the screen",
            "Try restarting the system control services",
        ]

    # Add context-specific suggestions
    if context["urgency"] == "high":
        troubleshooting["suggestions"].insert(0, "For urgent requests, try manual methods first")

    return troubleshooting
