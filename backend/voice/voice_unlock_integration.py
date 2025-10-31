"""
Voice Unlock Integration Service
==================================

Integrates the new SpeechBrain-based voice enrollment system with
the existing macOS screen unlock infrastructure.

This bridges:
- New: SpeechBrain speaker verification with Cloud SQL voiceprints
- Existing: Mac unlock automation, keychain, screen lock detection
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))

from intelligence.learning_database import JARVISLearningDatabase
from voice.speaker_verification_service import SpeakerVerificationService
from voice_unlock.objc.server.screen_lock_detector import is_screen_locked
from voice_unlock.services.keychain_service import KeychainService

logger = logging.getLogger(__name__)


class VoiceUnlockIntegration:
    """
    Integrated voice unlock service combining new voice system with existing unlock infrastructure

    Features:
    - Real speaker verification using SpeechBrain ECAPA-TDNN embeddings
    - Cloud SQL voiceprint storage
    - macOS screen lock detection
    - Keychain password retrieval
    - Learning database integration for audit trail
    """

    def __init__(self):
        self.speaker_verification = None
        self.keychain_service = None
        self.learning_db = None
        self.initialized = False

        # Configuration
        self.unlock_confidence_threshold = 0.85  # Higher threshold for security
        self.general_confidence_threshold = 0.75  # For general speaker identification

        # Statistics
        self.stats = {
            "total_unlock_attempts": 0,
            "successful_unlocks": 0,
            "failed_verifications": 0,
            "rejected_low_confidence": 0,
            "screen_already_unlocked": 0,
        }

    async def initialize(self):
        """Initialize all services"""
        if self.initialized:
            return

        logger.info("🚀 Initializing Voice Unlock Integration...")

        try:
            # Initialize speaker verification with Cloud SQL
            self.speaker_verification = SpeakerVerificationService()
            await self.speaker_verification.initialize()
            logger.info("✅ Speaker verification initialized")

            # Initialize keychain service
            self.keychain_service = KeychainService()
            logger.info("✅ Keychain service initialized")

            # Initialize learning database
            self.learning_db = JARVISLearningDatabase()
            await self.learning_db.initialize()
            logger.info("✅ Learning database initialized")

            self.initialized = True
            logger.info("✅ Voice Unlock Integration ready")

        except Exception as e:
            logger.error(f"Failed to initialize Voice Unlock Integration: {e}", exc_info=True)
            raise

    async def verify_and_unlock(
        self, audio_data: bytes, speaker_name: str, command_text: Optional[str] = None
    ) -> Tuple[bool, str, float]:
        """
        Verify speaker and unlock screen if authorized

        Args:
            audio_data: Audio bytes from microphone
            speaker_name: Expected speaker name (e.g., "Derek J. Russell")
            command_text: Optional command text that triggered unlock

        Returns:
            Tuple of (success, message, confidence)
        """
        if not self.initialized:
            await self.initialize()

        self.stats["total_unlock_attempts"] += 1

        try:
            # Step 1: Check if screen is already unlocked
            if not is_screen_locked():
                logger.info("📱 Screen is already unlocked")
                self.stats["screen_already_unlocked"] += 1
                return True, "Screen is already unlocked", 1.0

            # Step 2: Verify speaker identity
            logger.info(f"🎤 Verifying speaker: {speaker_name}")
            result = await self.speaker_verification.verify_speaker(
                audio_data, speaker_name
            )

            # Extract values from result dict
            is_verified = result["verified"]
            confidence = result["confidence"]
            speaker_id = result.get("speaker_id")

            logger.info(
                f"🔐 Verification result: {speaker_name}, "
                f"Confidence: {confidence:.2%}, Verified: {is_verified}"
            )

            # Step 3: Check confidence threshold for unlock
            if confidence < self.unlock_confidence_threshold:
                logger.warning(
                    f"❌ Confidence too low for unlock: {confidence:.2%} < "
                    f"{self.unlock_confidence_threshold:.2%}"
                )
                self.stats["rejected_low_confidence"] += 1

                # Record failed attempt
                await self._record_unlock_attempt(
                    speaker_name=speaker_name,
                    confidence=confidence,
                    success=False,
                    reason="Low confidence",
                    command_text=command_text,
                )

                return False, f"Voice verification confidence too low: {confidence:.2%}", confidence

            if not is_verified:
                logger.warning(f"❌ Speaker not verified: {speaker_name}")
                self.stats["failed_verifications"] += 1

                # Record failed attempt
                await self._record_unlock_attempt(
                    speaker_name=speaker_name,
                    confidence=confidence,
                    success=False,
                    reason="Speaker not verified",
                    command_text=command_text,
                )

                return False, "Speaker verification failed", confidence

            # Step 4: Retrieve password from keychain
            logger.info("🔑 Retrieving unlock password from keychain...")
            password = self._get_unlock_password()

            if not password:
                logger.error("❌ Failed to retrieve unlock password")
                return False, "Failed to retrieve unlock password", confidence

            # Step 5: Execute unlock
            logger.info("🔓 Executing screen unlock...")
            unlock_success = await self._execute_unlock(password)

            if unlock_success:
                logger.info(f"✅ Screen unlocked successfully for {speaker_name}")
                self.stats["successful_unlocks"] += 1

                # Record successful unlock
                await self._record_unlock_attempt(
                    speaker_name=speaker_name,
                    confidence=confidence,
                    success=True,
                    reason="Successful unlock",
                    command_text=command_text,
                )

                return True, f"Screen unlocked. Welcome, {speaker_name}!", confidence
            else:
                logger.error("❌ Unlock execution failed")
                return False, "Unlock execution failed", confidence

        except Exception as e:
            logger.error(f"Error during verify and unlock: {e}", exc_info=True)
            return False, f"Error: {str(e)}", 0.0

    def _get_unlock_password(self) -> Optional[str]:
        """Retrieve unlock password from keychain"""
        try:
            # Try to get password from keychain
            password = self.keychain_service.get_password(
                service="com.jarvis.voiceunlock", account="unlock_token"
            )

            if password:
                return password

            # Fallback: Try to get from environment variable (for testing)
            import os

            password = os.environ.get("JARVIS_UNLOCK_PASSWORD")

            if password:
                logger.warning(
                    "Using password from environment variable (not recommended for production)"
                )
                return password

            logger.error("No unlock password found in keychain or environment")
            return None

        except Exception as e:
            logger.error(f"Error retrieving unlock password: {e}", exc_info=True)
            return None

    async def _execute_unlock(self, password: str) -> bool:
        """Execute the actual screen unlock"""
        try:
            # Use AppleScript to type password and unlock
            unlock_script = f"""
            tell application "System Events"
                -- Wake display
                do shell script "caffeinate -u -t 1"
                delay 0.5

                -- Activate loginwindow if needed
                set frontApp to name of first process whose frontmost is true
                if frontApp contains "loginwindow" then
                    -- Type password
                    keystroke "{password}"
                    delay 0.2
                    keystroke return
                    delay 1.5
                end if
            end tell
            """

            import subprocess

            result = subprocess.run(
                ["osascript", "-e", unlock_script], capture_output=True, text=True, timeout=5
            )

            if result.returncode != 0:
                logger.error(f"Unlock script failed: {result.stderr}")
                return False

            # Verify unlock
            await asyncio.sleep(0.5)
            if not is_screen_locked():
                logger.info("✅ Unlock verified successful")
                return True
            else:
                logger.warning("⚠️ Screen still locked after unlock attempt")
                return False

        except Exception as e:
            logger.error(f"Error executing unlock: {e}", exc_info=True)
            return False

    async def _record_unlock_attempt(
        self,
        speaker_name: str,
        confidence: float,
        success: bool,
        reason: str,
        command_text: Optional[str] = None,
    ):
        """Record unlock attempt in learning database for audit trail"""
        try:
            if self.learning_db:
                # Record in conversation history
                await self.learning_db.record_interaction(
                    user_query=command_text or "unlock screen",
                    jarvis_response=reason,
                    response_type="voice_unlock",
                    confidence_score=confidence,
                    success=success,
                    metadata={
                        "speaker_name": speaker_name,
                        "unlock_attempt": True,
                        "reason": reason,
                    },
                )
        except Exception as e:
            logger.error(f"Error recording unlock attempt: {e}", exc_info=True)

    async def identify_speaker(self, audio_data: bytes) -> Tuple[Optional[str], float, bool]:
        """
        Identify speaker from audio without unlocking

        Args:
            audio_data: Audio bytes from microphone

        Returns:
            Tuple of (speaker_name, confidence, is_owner)
        """
        if not self.initialized:
            await self.initialize()

        try:
            # Use general threshold for identification
            result = await self.speaker_verification.identify_speaker(audio_data)

            if result:
                speaker_name = result.get("speaker_name")
                confidence = result.get("confidence", 0.0)
                is_owner = result.get("is_primary_user", False)

                return speaker_name, confidence, is_owner
            else:
                return None, 0.0, False

        except Exception as e:
            logger.error(f"Error identifying speaker: {e}", exc_info=True)
            return None, 0.0, False

    def get_stats(self) -> dict:
        """Get unlock statistics"""
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful_unlocks"] / self.stats["total_unlock_attempts"]
                if self.stats["total_unlock_attempts"] > 0
                else 0.0
            ),
        }

    async def cleanup(self):
        """Cleanup resources"""
        if self.speaker_verification:
            await self.speaker_verification.cleanup()
        if self.learning_db:
            await self.learning_db.close()
        self.initialized = False
        logger.info("🧹 Voice Unlock Integration cleaned up")


# Singleton instance
_voice_unlock_integration = None


async def get_voice_unlock_integration() -> VoiceUnlockIntegration:
    """Get or create singleton instance"""
    global _voice_unlock_integration

    if _voice_unlock_integration is None:
        _voice_unlock_integration = VoiceUnlockIntegration()
        await _voice_unlock_integration.initialize()

    return _voice_unlock_integration
