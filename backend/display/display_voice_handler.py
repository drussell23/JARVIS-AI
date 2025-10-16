#!/usr/bin/env python3
"""
Display Monitor Voice Handler
==============================

Voice integration wrapper for display monitoring system.
Bridges the display monitor with JARVIS voice system.

Author: Derek Russell
Date: 2025-10-15
Version: 1.0
"""

import asyncio
import logging
import subprocess
from typing import Optional, Dict, Any
import os

logger = logging.getLogger(__name__)


class DisplayVoiceHandler:
    """
    Voice handler for display monitoring

    Integrates with JARVIS voice systems:
    - voice_engine.py (if available)
    - voice_integration_handler.py (if available)
    - Fallback to macOS say command
    """

    def __init__(self, voice_engine=None, voice_integration=None):
        """
        Initialize voice handler

        Args:
            voice_engine: JARVIS voice engine instance
            voice_integration: JARVIS voice integration handler instance
        """
        self.voice_engine = voice_engine
        self.voice_integration = voice_integration

        # Voice settings from environment
        self.voice_enabled = os.getenv('JARVIS_VOICE_ENABLED', 'true').lower() == 'true'
        self.voice_rate = float(os.getenv('JARVIS_VOICE_RATE', '1.0'))
        self.voice_name = os.getenv('JARVIS_VOICE_NAME', 'Samantha')

        logger.info(f"[DISPLAY VOICE] Initialized (enabled={self.voice_enabled})")

    async def speak(self, message: str, priority: str = "normal"):
        """
        Speak a message using available voice systems

        Args:
            message: Message to speak
            priority: Priority level (low, normal, high, urgent)
        """
        if not self.voice_enabled:
            logger.debug(f"[DISPLAY VOICE] Voice disabled, skipping: {message}")
            return

        logger.info(f"[DISPLAY VOICE] Speaking: {message}")

        # For display monitor, always use macOS say for immediate feedback
        # This ensures the announcement is heard right when the TV is detected
        await self._speak_with_say(message)

    async def _try_jarvis_voice(self, message: str, priority: str) -> bool:
        """
        Try to use JARVIS voice systems

        Returns:
            True if successful, False if not available
        """
        # For display monitor, we want immediate audio feedback
        # Skip voice_integration (which queues notifications) and use macOS say directly
        # This ensures users hear the announcement immediately when TV is detected

        # Try voice_engine if available
        if self.voice_engine:
            try:
                # Check if voice_engine has speak method
                if hasattr(self.voice_engine, 'speak'):
                    await self.voice_engine.speak(message)
                    logger.debug("[DISPLAY VOICE] Used voice_engine")
                    return True
                elif hasattr(self.voice_engine, 'text_to_speech'):
                    await self.voice_engine.text_to_speech(message)
                    logger.debug("[DISPLAY VOICE] Used voice_engine.text_to_speech")
                    return True
            except Exception as e:
                logger.warning(f"[DISPLAY VOICE] voice_engine error: {e}")

        return False

    async def _speak_with_say(self, message: str):
        """
        Speak using macOS say command

        Args:
            message: Message to speak
        """
        try:
            # Build say command
            cmd = ['say']

            # Add voice name
            if self.voice_name:
                cmd.extend(['-v', self.voice_name])

            # Add rate
            if self.voice_rate != 1.0:
                rate = int(175 * self.voice_rate)  # 175 is default rate
                cmd.extend(['-r', str(rate)])

            # Add message
            cmd.append(message)

            # Run in background (non-blocking)
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )

            logger.debug(f"[DISPLAY VOICE] Used macOS say command (voice={self.voice_name}, rate={self.voice_rate})")

            # Don't wait for completion (fire and forget)
            # await process.wait()

        except FileNotFoundError:
            logger.error("[DISPLAY VOICE] say command not found (not on macOS?)")
        except Exception as e:
            logger.error(f"[DISPLAY VOICE] say command error: {e}")

    def set_enabled(self, enabled: bool):
        """Enable or disable voice"""
        self.voice_enabled = enabled
        logger.info(f"[DISPLAY VOICE] Voice {'enabled' if enabled else 'disabled'}")

    def set_voice_name(self, voice_name: str):
        """Set voice name for say command"""
        self.voice_name = voice_name
        logger.info(f"[DISPLAY VOICE] Voice name set to: {voice_name}")

    def set_voice_rate(self, rate: float):
        """Set voice rate (1.0 = normal)"""
        self.voice_rate = rate
        logger.info(f"[DISPLAY VOICE] Voice rate set to: {rate}")

    def get_available_voices(self) -> list:
        """Get list of available macOS voices"""
        try:
            result = subprocess.run(
                ['say', '-v', '?'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                voices = []
                for line in result.stdout.split('\n'):
                    if line.strip():
                        # Parse voice name (first word)
                        parts = line.split()
                        if parts:
                            voices.append(parts[0])
                return voices

            return []

        except Exception as e:
            logger.error(f"[DISPLAY VOICE] Error getting voices: {e}")
            return []

    async def test_voice(self, test_message: str = "JARVIS display monitoring is online, sir."):
        """
        Test voice output

        Args:
            test_message: Message to test with
        """
        logger.info("[DISPLAY VOICE] Testing voice...")
        await self.speak(test_message, priority="normal")


# Factory function
def create_voice_handler(voice_engine=None, voice_integration=None) -> DisplayVoiceHandler:
    """
    Create voice handler with automatic detection of JARVIS voice systems

    Args:
        voice_engine: Optional voice engine instance
        voice_integration: Optional voice integration instance

    Returns:
        DisplayVoiceHandler instance
    """
    # Try to auto-detect if not provided
    if voice_engine is None:
        try:
            from engines.voice_engine import VoiceEngine
            voice_engine = VoiceEngine()
            logger.info("[DISPLAY VOICE] Auto-detected VoiceEngine")
        except:
            pass

    # Don't auto-detect voice_integration for display monitor
    # We want immediate audio feedback via macOS say, not queued notifications
    # if voice_integration is None:
    #     try:
    #         from vision.voice_integration_handler import VoiceIntegrationHandler
    #         voice_integration = VoiceIntegrationHandler()
    #         logger.info("[DISPLAY VOICE] Auto-detected VoiceIntegrationHandler")
    #     except:
    #         pass

    return DisplayVoiceHandler(voice_engine, voice_integration=None)


if __name__ == "__main__":
    # Test the voice handler
    async def test():
        logging.basicConfig(level=logging.INFO)

        handler = create_voice_handler()

        print("Available voices:")
        voices = handler.get_available_voices()
        for voice in voices[:10]:  # Show first 10
            print(f"  - {voice}")

        print("\nTesting voice output...")
        await handler.test_voice()

        print("\nTesting with different message...")
        await handler.speak("Sir, your Living Room TV is now available. Would you like to extend your display?")

    asyncio.run(test())
