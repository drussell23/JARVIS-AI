#!/usr/bin/env python3
"""
Direct Unlock Handler - FIXED
=============================

Provides direct screen unlock functionality with correct message format
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
        logger.info(
            "[DIRECT UNLOCK] Connecting to voice unlock daemon for direct unlock"
        )

        async with websockets.connect(
            VOICE_UNLOCK_WS_URL, ping_interval=20
        ) as websocket:
            # Send unlock command with CORRECT format
            unlock_command = {
                "type": "command",
                "command": "unlock_screen",
                "parameters": {
                    "source": "context_handler",
                    "reason": reason,
                    "authenticated": True,
                },
            }

            await websocket.send(json.dumps(unlock_command))
            logger.info(f"[DIRECT UNLOCK] Sent unlock command: {unlock_command}")

            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            result = json.loads(response)

            logger.info(f"[DIRECT UNLOCK] Unlock response: {result}")

            # Check for correct response format
            if (
                result.get("type") == "command_response"
                and result.get("command") == "unlock_screen"
            ):
                success = result.get("success", False)
                message = result.get("message", "")
                logger.info(
                    f"[DIRECT UNLOCK] Unlock {'succeeded' if success else 'failed'}: {message}"
                )
                return success
            else:
                logger.error(f"[DIRECT UNLOCK] Unexpected response: {result}")
                return False

    except asyncio.TimeoutError:
        logger.error("[DIRECT UNLOCK] Timeout waiting for unlock response")
        return False
    except websockets.exceptions.ConnectionRefusedError:
        logger.error("[DIRECT UNLOCK] Voice unlock daemon not running on port 8765")
        return False
    except Exception as e:
        logger.error(f"[DIRECT UNLOCK] Error in direct unlock: {e}")
        return False


async def check_screen_locked_direct() -> bool:
    """Check if screen is locked via direct WebSocket"""
    try:
        logger.info("[DIRECT UNLOCK] Checking screen lock status via WebSocket")
        async with websockets.connect(
            VOICE_UNLOCK_WS_URL, ping_interval=20
        ) as websocket:
            # Get status with correct format
            status_command = {"type": "command", "command": "get_status"}
            await websocket.send(json.dumps(status_command))

            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            result = json.loads(response)
            logger.info(f"[DIRECT UNLOCK] Voice unlock status: {result}")

            if result.get("type") == "status" and result.get("success"):
                status = result.get("status", {})
                is_locked = status.get("isScreenLocked", False)
                logger.info(f"[DIRECT UNLOCK] Screen locked from daemon: {is_locked}")
                return is_locked

        return False

    except websockets.exceptions.ConnectionRefusedError:
        logger.warning(
            "[DIRECT UNLOCK] Voice unlock daemon not running, checking via system"
        )
        return check_screen_locked_system()
    except Exception as e:
        logger.error(f"[DIRECT UNLOCK] Error checking screen lock: {e}")
        # Fallback to system check
        return check_screen_locked_system()


def check_screen_locked_system() -> bool:
    """Check screen lock state using system API"""
    try:
        logger.info("[DIRECT UNLOCK] Checking screen lock via system API")
        import subprocess

        # Use a more reliable method to check screen lock
        check_script = """
import Quartz
import sys

try:
    # Get the current session dictionary
    session_dict = Quartz.CGSessionCopyCurrentDictionary()
    if session_dict:
        # Check multiple indicators
        screen_locked = session_dict.get("CGSSessionScreenIsLocked", False)
        screen_saver = session_dict.get("CGSSessionScreenSaverIsActive", False)
        on_console = session_dict.get("kCGSSessionOnConsoleKey", True)
        
        # Screen is considered locked if locked flag is True or screensaver is active
        is_locked = bool(screen_locked or screen_saver)
        print("true" if is_locked else "false")
    else:
        # If we can't get session dict, assume unlocked
        print("false")
except Exception as e:
    print("false")
    sys.exit(1)
"""

        result = subprocess.run(
            ["python3", "-c", check_script], capture_output=True, text=True, timeout=5
        )

        is_locked = result.stdout.strip().lower() == "true"
        logger.info(f"[DIRECT UNLOCK] Screen locked from system: {is_locked}")
        return is_locked

    except subprocess.TimeoutExpired:
        logger.error("[DIRECT UNLOCK] Timeout checking screen lock state")
        return False
    except Exception as e:
        logger.error(f"[DIRECT UNLOCK] Error in system screen check: {e}")
        return False


async def test_screen_lock_context():
    """Test function to verify screen lock detection and unlock"""
    print("\n🔍 Testing Screen Lock Context Detection")
    print("=" * 50)

    # Check if screen is locked
    is_locked = await check_screen_locked_direct()
    print(f"Screen is {'LOCKED' if is_locked else 'UNLOCKED'}")

    if is_locked:
        print("\n🔓 Attempting to unlock screen...")
        success = await unlock_screen_direct("Testing context awareness")
        if success:
            print("✅ Screen unlocked successfully!")
        else:
            print("❌ Failed to unlock screen")
    else:
        print("\n💡 Lock your screen (Cmd+Ctrl+Q) and run this test again")

    return is_locked


if __name__ == "__main__":
    # Run test
    asyncio.run(test_screen_lock_context())

