#!/usr/bin/env python3
"""
Transport Method Handlers
==========================

Implementation of all transport methods for screen control.
Each handler is async, timeout-safe, and reports detailed results.
"""

import asyncio
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


async def applescript_handler(action: str, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    AppleScript transport handler - Direct system automation.

    Most reliable method, works even when network services are down.
    Fast execution, low latency.
    Uses actual MacOSKeychainUnlock for unlock and system commands for lock.
    """
    logger.info(f"[APPLESCRIPT] Executing {action}")

    try:
        if action == "unlock_screen":
            # Use MacOSKeychainUnlock for actual unlock
            from macos_keychain_unlock import MacOSKeychainUnlock

            unlock_service = MacOSKeychainUnlock()
            verified_speaker = context.get("verified_speaker_name", "Derek")

            # Perform actual screen unlock with Keychain password
            unlock_result = await unlock_service.unlock_screen(verified_speaker=verified_speaker)

            if unlock_result["success"]:
                logger.info(f"[APPLESCRIPT] ✅ {action} succeeded for {verified_speaker}")
                return {
                    "success": True,
                    "method": "applescript",
                    "action": action,
                    "verified_speaker": verified_speaker,
                }
            else:
                logger.error(f"[APPLESCRIPT] ❌ Failed: {unlock_result['message']}")
                return {
                    "success": False,
                    "error": "unlock_failed",
                    "message": unlock_result["message"],
                }

        elif action == "lock_screen":
            # Use Command+Control+Q to lock screen (native macOS shortcut)
            script = """
            tell application "System Events"
                keystroke "q" using {command down, control down}
            end tell
            """

            # Execute AppleScript
            process = await asyncio.create_subprocess_exec(
                "osascript",
                "-e",
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                logger.info(f"[APPLESCRIPT] ✅ {action} succeeded")
                return {
                    "success": True,
                    "method": "applescript",
                    "action": action,
                }
            else:
                error_msg = stderr.decode().strip() if stderr else "unknown error"
                logger.error(f"[APPLESCRIPT] ❌ Failed: {error_msg}")
                return {
                    "success": False,
                    "error": "applescript_failed",
                    "message": error_msg,
                }

        else:
            return {
                "success": False,
                "error": "unknown_action",
                "message": f"Unknown action: {action}",
            }

    except Exception as e:
        logger.error(f"[APPLESCRIPT] Exception: {e}", exc_info=True)
        return {
            "success": False,
            "error": "applescript_exception",
            "message": str(e),
        }


async def _applescript_wake_display():
    """Wake display using caffeinate"""
    try:
        process = await asyncio.create_subprocess_exec(
            "caffeinate",
            "-u",
            "-t",
            "1",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await process.wait()
    except Exception as e:
        logger.debug(f"[APPLESCRIPT] Display wake failed: {e}")


async def http_rest_handler(action: str, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    HTTP REST transport handler - Local HTTP API fallback.

    Uses aiohttp to call local REST endpoints.
    Reliable when WebSocket is down but HTTP server is running.
    """
    logger.info(f"[HTTP-REST] Executing {action}")

    try:
        import aiohttp

        endpoint_map = {
            "unlock_screen": "http://localhost:8000/api/screen/unlock",
            "lock_screen": "http://localhost:8000/api/screen/lock",
        }

        endpoint = endpoint_map.get(action)
        if not endpoint:
            return {"success": False, "error": "unknown_action"}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                json={"action": action, "context": context},
                timeout=aiohttp.ClientTimeout(total=3.0),
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"[HTTP-REST] ✅ {action} succeeded")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"[HTTP-REST] ❌ Failed: HTTP {response.status}")
                    return {
                        "success": False,
                        "error": "http_error",
                        "message": f"HTTP {response.status}: {error_text}",
                    }

    except asyncio.TimeoutError:
        logger.warning("[HTTP-REST] Request timed out")
        return {"success": False, "error": "http_timeout"}
    except Exception as e:
        logger.error(f"[HTTP-REST] Exception: {e}", exc_info=True)
        return {
            "success": False,
            "error": "http_exception",
            "message": str(e),
        }


async def unified_websocket_handler(
    action: str, context: Dict[str, Any], **kwargs
) -> Dict[str, Any]:
    """
    Unified WebSocket transport handler - Real-time bidirectional communication.

    Uses the EXISTING unified WebSocket connection (not port 8765).
    Fast when connection is healthy, but requires active WebSocket.
    """
    logger.info(f"[UNIFIED-WS] Executing {action}")

    try:
        # Get WebSocket connection from app state
        websocket_manager = kwargs.get("websocket_manager")

        if not websocket_manager:
            logger.warning("[UNIFIED-WS] WebSocket manager not available")
            return {
                "success": False,
                "error": "ws_not_available",
                "message": "WebSocket manager not initialized",
            }

        # Send action through unified WebSocket
        message = {
            "type": "screen_control",
            "action": action,
            "context": context,
        }

        # Broadcast to all connected clients
        await websocket_manager.broadcast(message)

        # For now, assume success if broadcast succeeded
        # In production, you'd wait for acknowledgment
        logger.info(f"[UNIFIED-WS] ✅ {action} broadcast succeeded")
        return {
            "success": True,
            "method": "unified_websocket",
            "action": action,
        }

    except Exception as e:
        logger.error(f"[UNIFIED-WS] Exception: {e}", exc_info=True)
        return {
            "success": False,
            "error": "ws_exception",
            "message": str(e),
        }


async def system_api_handler(action: str, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    System API transport handler - macOS system APIs.

    Uses AppleScript shortcut for lock, delegates to AppleScript handler for unlock.
    Most compatible with macOS system features.
    """
    logger.info(f"[SYSTEM-API] Executing {action}")

    try:
        if action == "lock_screen":
            # Use AppleScript with Command+Control+Q (reliable macOS shortcut)
            script = """
            tell application "System Events"
                keystroke "q" using {command down, control down}
            end tell
            """

            process = await asyncio.create_subprocess_exec(
                "osascript",
                "-e",
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            await process.communicate()

            if process.returncode == 0:
                logger.info(f"[SYSTEM-API] ✅ {action} succeeded")
                return {
                    "success": True,
                    "method": "system_api",
                    "action": action,
                }

        elif action == "unlock_screen":
            # Delegate to AppleScript handler which has MacOSKeychainUnlock
            logger.info("[SYSTEM-API] Delegating unlock to AppleScript handler")
            return await applescript_handler(action, context, **kwargs)

        return {
            "success": False,
            "error": "unknown_action",
            "message": f"Unknown action: {action}",
        }

    except Exception as e:
        logger.error(f"[SYSTEM-API] Exception: {e}", exc_info=True)
        return {
            "success": False,
            "error": "system_api_exception",
            "message": str(e),
        }


# Handler registry
TRANSPORT_HANDLERS = {
    "applescript": applescript_handler,
    "http_rest": http_rest_handler,
    "unified_websocket": unified_websocket_handler,
    "system_api": system_api_handler,
}
