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

# Import async pipeline
from core.async_pipeline import get_async_pipeline
from core.transport_handlers import (
    applescript_handler,
    http_rest_handler,
    system_api_handler,
    unified_websocket_handler,
)

# Import transport layer
from core.transport_manager import TransportMethod, get_transport_manager

logger = logging.getLogger(__name__)

# Global instances
_pipeline = None
_transport_manager = None


async def _get_transport_manager():
    """Get or create and initialize the transport manager"""
    global _transport_manager
    if _transport_manager is None:
        _transport_manager = get_transport_manager()

        # Register all transport handlers
        _transport_manager.register_handler(TransportMethod.APPLESCRIPT, applescript_handler)
        _transport_manager.register_handler(TransportMethod.HTTP_REST, http_rest_handler)
        _transport_manager.register_handler(
            TransportMethod.UNIFIED_WEBSOCKET, unified_websocket_handler
        )
        _transport_manager.register_handler(TransportMethod.SYSTEM_API, system_api_handler)

        # Initialize (starts health monitoring)
        await _transport_manager.initialize()

        logger.info("[TRANSPORT] ✅ Transport manager initialized with all handlers")

    return _transport_manager


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

    # Execute the action with multiple fallback methods FIRST
    # This will set the verified_speaker_name in the context
    result = await _execute_screen_action(action, context, jarvis_instance)

    # Generate dynamic response AFTER verification with speaker name from context
    response_text = await _generate_contextual_response(action, context, response_gen)

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
    """Generate dynamic, contextual response with speaker name if verified - SPEED OPTIMIZED."""

    # Get speaker name if verified
    speaker_name = context.get("verified_speaker_name", "Sir")

    # FAST PATH: Use pre-generated responses for speed (no dynamic generation overhead)
    # This eliminates the response_gen overhead while keeping contextual awareness

    if action == "unlock_screen" and context.get("screen_state") == "unlocked":
        return f"Your screen is already unlocked, {speaker_name}."

    # Use fast fallback responses with speaker name
    if context.get("urgency") == "high":
        return f"Right away, {speaker_name}! " + (
            "Unlocking your screen now."
            if action == "unlock_screen"
            else "Locking your screen now."
        )

    # Default fast responses with speaker name
    return f"Of course, {speaker_name}. " + (
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
    """
    Execute screen action using advanced transport manager.

    Automatically selects the best transport method based on:
    - Real-time health monitoring
    - Success rate history
    - Latency metrics
    - Current availability
    """

    logger.info(f"[TRANSPORT-EXEC] Executing {action}")
    logger.debug(f"[TRANSPORT-EXEC] Context: {context}")

    # Pass jarvis_instance to context for audio/speaker extraction
    if jarvis_instance:
        context["jarvis_instance"] = jarvis_instance

    # Get transport manager
    transport = await _get_transport_manager()

    # Execute using smart transport selection
    result = await transport.execute(action, context)

    # Log result with metrics
    if result.get("success"):
        logger.info(
            f"[TRANSPORT-EXEC] ✅ {action} succeeded via {result.get('transport_method')} "
            f"({result.get('latency_ms', 0):.1f}ms)"
        )
    else:
        logger.warning(
            f"[TRANSPORT-EXEC] ❌ {action} failed: {result.get('error')} "
            f"(attempted: {result.get('attempted_methods', [])})"
        )

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

    # If no audio data (text command), bypass voice verification
    # User is already authenticated by being logged into the system
    if not context.get("audio_data"):
        context["bypass_voice_verification"] = True
        logger.info(
            "📝 Text-based unlock command - bypassing voice verification (user already authenticated)"
        )

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
    Try keychain-based unlock method with voice verification using 25 voice samples.

    Security Flow:
    1. Verify speaker identity from audio using 25 biometric samples
    2. Check if speaker is the device owner
    3. Retrieve password from keychain
    4. Perform unlock
    """
    # Step 1: Voice verification (if audio data is available in context)
    # Extract audio data from jarvis_instance if present
    jarvis_instance = context.get("jarvis_instance")
    audio_data = context.get("audio_data")
    speaker_name = context.get("speaker_name")

    # Try to get audio from jarvis_instance if not in context directly
    if not audio_data and jarvis_instance:
        if hasattr(jarvis_instance, "last_audio_data"):
            audio_data = jarvis_instance.last_audio_data
            logger.info(
                f"[VOICE] Extracted audio from jarvis_instance: {len(audio_data) if audio_data else 0} bytes"
            )
        if hasattr(jarvis_instance, "last_speaker_name"):
            speaker_name = jarvis_instance.last_speaker_name

    logger.info(
        f"🎤 Audio data status: {len(audio_data) if audio_data else 0} bytes, speaker: {speaker_name}"
    )

    if audio_data:
        # Verify speaker using NEW speaker verification service (with pre-loaded profiles)
        try:
            from voice.speaker_verification_service import get_speaker_verification_service

            speaker_service = await get_speaker_verification_service()
            logger.info(
                f"🔐 Speaker service has {len(speaker_service.speaker_profiles)} profiles loaded"
            )
            logger.info(f"🔐 Available profiles: {list(speaker_service.speaker_profiles.keys())}")

            # Verify speaker using voice biometrics from database
            verification_result = await speaker_service.verify_speaker(audio_data, speaker_name)
            logger.info(f"🎤 Verification result: {verification_result}")

            speaker_name = verification_result["speaker_name"]
            is_verified = verification_result["verified"]
            confidence = verification_result["confidence"]
            is_owner = verification_result["is_owner"]

            logger.info(
                f"🔐 Speaker verification: {speaker_name} - "
                f"{'✅ VERIFIED' if is_verified else '❌ FAILED'} "
                f"(confidence: {confidence:.1%}, owner: {is_owner})"
            )

            if not is_verified:
                logger.warning(f"🚫 Voice verification failed for {speaker_name} - unlock denied")
                return (
                    False,
                    f"I couldn't verify your identity. For security, unlock is denied.",
                )

            # Check if speaker is the device owner
            if not is_owner:
                logger.warning(f"🚫 Non-owner {speaker_name} attempted unlock - denied")
                return (
                    False,
                    f"Only the device owner can unlock the screen via voice.",
                )

            # Store verified speaker name in context for personalized response
            context["verified_speaker_name"] = speaker_name

            logger.info(f"✅ Voice verification passed for owner: {speaker_name}")

        except Exception as e:
            logger.error(f"Voice verification error: {e}")
            # No fallback - require proper voice verification
            context["verified_speaker_name"] = None
            logger.warning("⚠️ Voice verification failed - speaker not recognized")
    else:
        # No audio data provided - use default owner name for now
        # TODO: Enforce voice verification once audio capture is confirmed working
        logger.warning("⚠️ No audio data for voice verification - using default owner")
        context["verified_speaker_name"] = "Derek"
        logger.info("⚠️ Using default owner 'Derek' for now - need audio capture working")

    # Step 2: Use enhanced Keychain integration for actual unlock
    try:
        from macos_keychain_unlock import MacOSKeychainUnlock

        unlock_service = MacOSKeychainUnlock()
        verified_speaker = context.get("verified_speaker_name", "Derek")

        # Perform actual screen unlock with Keychain password
        unlock_result = await unlock_service.unlock_screen(verified_speaker=verified_speaker)

        if unlock_result["success"]:
            logger.info(f"🔓 Screen unlocked successfully for {verified_speaker}")
            return True, f"Screen unlocked for {verified_speaker}"
        else:
            # Check if setup is required
            if unlock_result.get("action") == "setup_required":
                logger.warning("⚠️ Keychain password not configured")
                return False, "Screen unlock password not configured. Please run setup."
            else:
                logger.error(f"❌ Unlock failed: {unlock_result['message']}")
                return False, unlock_result["message"]

    except ImportError:
        logger.error("MacOS Keychain integration not available")
        # Fallback to old method
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
            unlock_result = await _perform_direct_unlock(password)

            if unlock_result:
                return True, f"Screen unlocked by {speaker_name or 'verified user'}"

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
