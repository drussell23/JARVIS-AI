"""
Voice Unlock Integration for JARVIS API
======================================

This module integrates the Voice Unlock daemon with JARVIS's main API.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import websockets

# Add voice_unlock to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "voice_unlock"))

# Import async pipeline
from core.async_pipeline import get_async_pipeline

logger = logging.getLogger(__name__)

# Import proximity auth if available
proximity_auth_available = False
proximity_auth_manager = None

try:
    from jarvis_proximity_integration import handle_proximity_voice_command

    proximity_auth_available = True
    logger.info("✅ Proximity + Voice Auth module available")
except ImportError:
    logger.info("ℹ️  Proximity + Voice Auth module not available")


class VoiceUnlockDaemonConnector:
    """Connects to the Voice Unlock Objective-C daemon via WebSocket"""

    def __init__(self, daemon_host: str = "localhost", daemon_port: int = 8765):
        self.daemon_host = daemon_host
        self.daemon_port = daemon_port
        self.daemon_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False

        # Initialize async pipeline
        self.pipeline = get_async_pipeline()
        self._register_pipeline_stages()

    def _register_pipeline_stages(self):
        """Register voice unlock pipeline stages"""
        self.pipeline.register_stage(
            "voice_unlock_connect", self._connect_async, timeout=5.0, retry_count=2, required=True
        )
        self.pipeline.register_stage(
            "voice_unlock_command",
            self._send_command_async,
            timeout=10.0,
            retry_count=1,
            required=True,
        )
        logger.info("✅ Voice unlock pipeline stages registered")

    async def _connect_async(self, context):
        """Async pipeline handler for connecting to daemon"""
        uri = context.metadata.get("uri")
        try:
            ws = await websockets.connect(uri, timeout=2.0)
            context.metadata["websocket"] = ws
            context.metadata["success"] = True
            context.metadata["connected"] = True
        except Exception as e:
            context.metadata["success"] = False
            context.metadata["error"] = str(e)
            logger.error(f"Failed to connect to Voice Unlock daemon: {e}")

    async def _send_command_async(self, context):
        """Async pipeline handler for sending commands to daemon"""
        command = context.metadata.get("command")
        parameters = context.metadata.get("parameters", {})
        ws = context.metadata.get("websocket")

        if not ws:
            context.metadata["success"] = False
            context.metadata["error"] = "No websocket connection"
            return

        message = {"type": "command", "command": command, "parameters": parameters}

        try:
            await ws.send(json.dumps(message))
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            result = json.loads(response)

            context.metadata["response"] = result
            context.metadata["success"] = result.get("success", False)
        except asyncio.TimeoutError:
            context.metadata["success"] = False
            context.metadata["error"] = "Command timeout"
        except Exception as e:
            context.metadata["success"] = False
            context.metadata["error"] = str(e)

    async def connect(self):
        """Connect to the Voice Unlock daemon via async pipeline"""
        try:
            uri = f"ws://{self.daemon_host}:{self.daemon_port}/voice-unlock"

            # Route through async pipeline
            result = await self.pipeline.process_async(
                text="Connect to Voice Unlock daemon",
                metadata={"uri": uri, "stage": "voice_unlock_connect"},
            )

            metadata = result.get("metadata", {})
            if metadata.get("success"):
                self.daemon_ws = metadata.get("websocket")
                self.connected = True
                logger.info(f"Connected to Voice Unlock daemon at {uri}")

                # Send handshake
                await self.send_command("handshake", {"client": "jarvis-api", "version": "1.0"})
            else:
                self.connected = False
                logger.error(f"Failed to connect: {metadata.get('error')}")

        except Exception as e:
            logger.error(f"Failed to connect to Voice Unlock daemon: {e}")
            self.connected = False

    async def disconnect(self):
        """Disconnect from the daemon"""
        if self.daemon_ws:
            await self.daemon_ws.close()
            self.connected = False

    async def send_command(self, command: str, parameters: dict = None):
        """Send command to the daemon via async pipeline"""
        if not self.connected or not self.daemon_ws:
            logger.error("Not connected to Voice Unlock daemon")
            return None

        try:
            # Route through async pipeline
            result = await self.pipeline.process_async(
                text=f"Send voice unlock command: {command}",
                metadata={
                    "command": command,
                    "parameters": parameters or {},
                    "websocket": self.daemon_ws,
                    "stage": "voice_unlock_command",
                },
            )

            metadata = result.get("metadata", {})
            if metadata.get("success"):
                return metadata.get("response")
            else:
                logger.error(f"Command failed: {metadata.get('error')}")
                return None

        except Exception as e:
            logger.error(f"Failed to send command to daemon: {e}")
            return None

    async def get_status(self):
        """Get Voice Unlock status"""
        return await self.send_command("get_status")

    async def enable_monitoring(self):
        """Enable voice monitoring"""
        return await self.send_command("start_monitoring")

    async def disable_monitoring(self):
        """Disable voice monitoring"""
        return await self.send_command("stop_monitoring")


# Global instance
voice_unlock_connector = None


async def initialize_voice_unlock():
    """Initialize Voice Unlock connection"""
    global voice_unlock_connector

    voice_unlock_connector = VoiceUnlockDaemonConnector()
    await voice_unlock_connector.connect()

    if voice_unlock_connector.connected:
        status = await voice_unlock_connector.get_status()
        logger.info(f"Voice Unlock daemon status: {status}")
        return True

    return False


async def handle_voice_unlock_in_jarvis(command: str, jarvis_instance=None) -> dict:
    """
    Handle voice unlock commands from JARVIS

    This should be called from unified_command_processor when
    voice unlock commands are detected.
    """
    command_lower = command.lower()

    # Ensure we're connected
    if not voice_unlock_connector or not voice_unlock_connector.connected:
        await initialize_voice_unlock()

    # Map JARVIS commands to daemon commands
    if "enable voice unlock" in command_lower:
        result = await voice_unlock_connector.enable_monitoring()
        return {
            "type": "voice_unlock",
            "action": "enabled",
            "message": "Voice unlock monitoring is now active, Sir. Your Mac will respond to your voice when locked.",
            "success": True,
        }

    elif "disable voice unlock" in command_lower:
        result = await voice_unlock_connector.disable_monitoring()
        return {
            "type": "voice_unlock",
            "action": "disabled",
            "message": "Voice unlock monitoring has been disabled, Sir.",
            "success": True,
        }

    elif "voice unlock status" in command_lower:
        status = await voice_unlock_connector.get_status()
        if status:
            monitoring = status.get("isMonitoring", False)
            enrolled = status.get("enrolledUser", "None")

            message = f"Voice unlock is {'active' if monitoring else 'inactive'}, Sir. "
            if enrolled and enrolled != "None":
                message += f"Enrolled user: {enrolled}"
            else:
                message += "No users enrolled."

            return {
                "type": "voice_unlock",
                "action": "status",
                "message": message,
                "status": status,
                "success": True,
            }

    elif any(
        phrase in command_lower
        for phrase in [
            "unlock my mac",
            "unlock my screen",
            "unlock mac",
            "unlock the mac",
            "unlock computer",
        ]
    ):
        # User wants to unlock NOW - use simple handler with voice verification
        from .simple_unlock_handler import handle_unlock_command

        # Pass audio data and speaker info for voice verification
        # Note: jarvis_instance should contain audio_data and speaker_name if available
        return await handle_unlock_command(command, jarvis_instance)

    elif any(
        phrase in command_lower
        for phrase in [
            "lock my mac",
            "lock my screen",
            "lock mac",
            "lock the mac",
            "lock computer",
            "lock the computer",
        ]
    ):
        # User wants to lock the screen - use simple handler to avoid WebSocket conflicts
        from .simple_unlock_handler import handle_unlock_command

        return await handle_unlock_command(command, jarvis_instance)

    elif "test voice unlock" in command_lower:
        # Test the voice unlock system
        if not voice_unlock_connector or not voice_unlock_connector.connected:
            await initialize_voice_unlock()

        status = await voice_unlock_connector.get_status()
        if status:
            return {
                "type": "voice_unlock",
                "action": "test",
                "message": f'Voice unlock system test: {"✅ Connected and ready" if status.get("isMonitoring") else "❌ Not monitoring"}',
                "status": status,
                "success": True,
            }

    # Proximity + Voice Auth commands
    elif proximity_auth_available and any(
        phrase in command_lower
        for phrase in [
            "proximity auth",
            "proximity unlock",
            "enable proximity",
            "check proximity",
            "enroll my voice",
            "voice biometric",
        ]
    ):
        if "enroll" in command_lower:
            # Voice enrollment would need audio data from the user
            return {
                "type": "proximity_auth",
                "action": "enroll_prompt",
                "message": 'Sir, I need you to say "Hello JARVIS, this is [your name]" three times for voice enrollment.',
                "requires_audio": True,
                "success": True,
            }

        elif "check proximity" in command_lower:
            result = await handle_proximity_voice_command("check_proximity")

            if result["success"]:
                return {
                    "type": "proximity_auth",
                    "action": "proximity_check",
                    "message": f"Your Apple Watch proximity score is {result['proximity_score']:.0f}%, Sir.",
                    "proximity_score": result["proximity_score"],
                    "success": True,
                }
            else:
                return {
                    "type": "proximity_auth",
                    "action": "proximity_check",
                    "message": f"Apple Watch not detected in proximity, Sir. {result['reason']}",
                    "success": False,
                }

        elif "enable proximity" in command_lower or "proximity unlock" in command_lower:
            status = await handle_proximity_voice_command("status")

            if status["initialized"]:
                return {
                    "type": "proximity_auth",
                    "action": "status",
                    "message": "Proximity + Voice authentication is active, Sir. Your Apple Watch and voice pattern provide dual-factor security.",
                    "status": status,
                    "success": True,
                }
            else:
                return {
                    "type": "proximity_auth",
                    "action": "initialization",
                    "message": "Initializing Proximity + Voice authentication system, Sir...",
                    "success": False,
                }

        elif "proximity auth status" in command_lower:
            status = await handle_proximity_voice_command("status")

            enrolled = status.get("voice_enrolled", False)
            proximity = status.get("proximity_service_running", False)

            message = f"Proximity + Voice Auth Status:\n"
            message += (
                f"• Apple Watch Detection: {'✅ Running' if proximity else '❌ Not running'}\n"
            )
            message += f"• Voice Enrollment: {'✅ Complete' if enrolled else '❌ Not enrolled'}\n"

            if "thresholds" in status:
                message += f"• Security Thresholds: Proximity {status['thresholds']['proximity_threshold']}%, Voice {status['thresholds']['voice_threshold']}%"

            return {
                "type": "proximity_auth",
                "action": "status",
                "message": message,
                "status": status,
                "success": True,
            }

    return {
        "type": "voice_unlock",
        "action": "unknown",
        "message": "I didn't understand that voice unlock command, Sir.",
        "success": False,
    }


# Integration function for main API
def integrate_voice_unlock_with_api(app):
    """
    Add Voice Unlock routes to the main FastAPI app

    This should be called from main.py to add Voice Unlock endpoints
    """
    from fastapi import APIRouter

    router = APIRouter(prefix="/voice-unlock", tags=["voice-unlock"])

    @router.get("/status")
    async def get_voice_unlock_status():
        """Get Voice Unlock daemon status"""
        if not voice_unlock_connector or not voice_unlock_connector.connected:
            await initialize_voice_unlock()

        if voice_unlock_connector and voice_unlock_connector.connected:
            status = await voice_unlock_connector.get_status()
            return {"status": "connected", "daemon_status": status}
        else:
            return {"status": "disconnected", "error": "Cannot connect to Voice Unlock daemon"}

    @router.post("/enable")
    async def enable_voice_unlock():
        """Enable Voice Unlock monitoring"""
        result = await handle_voice_unlock_in_jarvis("enable voice unlock")
        return result

    @router.post("/disable")
    async def disable_voice_unlock():
        """Disable Voice Unlock monitoring"""
        result = await handle_voice_unlock_in_jarvis("disable voice unlock")
        return result

    app.include_router(router)

    # Initialize on startup
    @app.on_event("startup")
    async def startup_voice_unlock():
        await initialize_voice_unlock()

    # Cleanup on shutdown
    @app.on_event("shutdown")
    async def shutdown_voice_unlock():
        if voice_unlock_connector:
            await voice_unlock_connector.disconnect()
